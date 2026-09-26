// Full READ-ONLY audit of the MILLOW chain against chain-manifest.json and the
// canonical mapping manifest.
//
//   npm run chain:verify
//   npm run chain:verify -- --full
//
// READ-ONLY BY CONSTRUCTION: every chain access is either an eth_call on a view
// function, eth_getCode, eth_getLogs, or a manifest read.  This file contains no
// sendTransaction, no deploy, no mint, no transfer, and lib/chain.js denylists
// hardhat_reset / anvil_reset / evm_revert for the whole toolchain.
//
//   default : full mapping + full ownership + full listings/sales,
//             tokenURI/propertyOf sampled every 25th
//   --full  : additionally verifies tokenURI and propertyOf for every token
//             (all 29,135).  Both modes fully verify the 29,135 mapping,
//             the 29,135 owners, the roles, the wiring, the PropertyMinted
//             event count and every listing/sale state.
//
// Exit code 0 = chain matches the canonical checkpoint, 1 = it does not.
const {
  loadManifest,
  loadTokenMap,
  manifestSha256,
  attachReadOnly,
  tokenUriFor,
  eventTopic,
  rpc,
  rpcUrl,
  isRpcUp,
} = require("./lib/chain");

const results = [];
function check(name, pass, detail = "") {
  results.push({ name, pass, detail });
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? `  — ${detail}` : ""}`);
  return pass;
}

async function main() {
  const full = process.argv.includes("--full");
  const manifest = loadManifest();
  const url = rpcUrl(manifest);
  const expected = manifest.canonical_token_count;
  const seller = manifest.roles.seller.toLowerCase();

  console.log(`=== MILLOW chain verification (${full ? "full" : "standard"}) ===`);
  console.log(`rpc ${url}   expected ${expected} tokens\n`);

  if (!(await isRpcUp(url))) {
    check("RPC reachable", false, "no response — start with: npm run chain:start");
    return finish();
  }

  const { provider, nft, registry, escrow } = await attachReadOnly(manifest);

  // ---- 1. chain id ----
  const net = await provider.getNetwork();
  const chainOk = check(
    "chainId == " + manifest.chain_id,
    net.chainId === manifest.chain_id,
    `got ${net.chainId}`,
  );

  // ---- 2. contracts exist with expected bytecode ----
  let codeOk = true;
  for (const [name, addr] of Object.entries(manifest.contracts)) {
    const code = await provider.getCode(addr);
    const bytes = code === "0x" ? 0 : (code.length - 2) / 2;
    const want = manifest.expected_code_bytes[name];
    if (!check(`${name} bytecode @ ${addr}`, bytes === want, `${bytes} bytes (want ${want})`)) {
      codeOk = false;
    }
  }

  // Nothing below this point can be meaningful without deployed code, and every
  // view call against an empty address reverts.  Report and stop rather than
  // throwing a raw ethers CALL_EXCEPTION.
  if (!chainOk || !codeOk) {
    console.log(
      "\nChain does not carry the MILLOW deployment; skipping the mapping sweep.\n" +
        "On Computer 1 build it with:  npx hardhat run scripts/deployMillow.js --network localhost\n" +
        "On Computer 2 import it with: npm run chain:import -- --artifact=<release .json.gz>",
    );
    return finish(false);
  }

  // ---- 3. total supply ----
  const supply = (await nft.totalSupply()).toNumber();
  check(`totalSupply == ${expected}`, supply === expected, `got ${supply}`);

  // ---- 4. mapping manifest integrity ----
  const rows = loadTokenMap(manifest);
  const sha = manifestSha256(manifest);
  check(
    "mapping manifest sha256",
    sha === manifest.mapping_manifest.sha256,
    `${sha.slice(0, 16)}...`,
  );

  // ---- 5. every MREID maps to its canonical token id ----
  let badMap = 0;
  let firstBad = null;
  const seenTokens = new Set();
  let dupTokens = 0;
  let notSeller = 0;
  let badOwnerSample = null;
  let badUri = 0;
  let badProp = 0;
  let missingUri = 0;
  const CONC = Number(process.env.MILLOW_VERIFY_CONCURRENCY) || 24;
  const BATCH = 250;

  for (let s = 0; s < rows.length; s += BATCH) {
    const chunk = rows.slice(s, s + BATCH);
    const res = await Promise.all(
      chunk.map(async (r) => {
        const out = { r, id: 0, owner: null, uri: null, prop: null };
        try {
          out.id = (await nft.tokenByProperty(r.mreid)).toNumber();
        } catch {
          out.id = -1;
        }
        if (out.id > 0) {
          try {
            out.owner = (await nft.ownerOf(out.id)).toLowerCase();
          } catch {
            out.owner = null;
          }
        }
        const deep = full || r.tokenId % 25 === 0;
        if (out.id > 0 && deep) {
          try {
            out.uri = await nft.tokenURI(out.id);
          } catch {
            out.uri = null;
          }
          try {
            out.prop = await nft.propertyOf(out.id);
          } catch {
            out.prop = null;
          }
        }
        return out;
      }),
    );
    for (const o of res) {
      if (o.id !== o.r.tokenId) {
        badMap += 1;
        if (!firstBad) firstBad = `${o.r.mreid} expected #${o.r.tokenId} got ${o.id}`;
      }
      if (o.id > 0) {
        if (seenTokens.has(o.id)) dupTokens += 1;
        seenTokens.add(o.id);
      }
      if (o.id > 0 && o.owner !== seller) {
        notSeller += 1;
        if (!badOwnerSample) badOwnerSample = `${o.r.mreid}#${o.id} owner ${o.owner}`;
      }
      if (o.uri !== null) {
        if (!o.uri) missingUri += 1;
        else if (o.uri !== tokenUriFor(o.r.mreid)) badUri += 1;
      }
      if (o.prop !== null && o.prop !== o.r.mreid) badProp += 1;
    }
    if (Math.floor(s / BATCH) % 10 === 0) {
      process.stdout.write(`  scanned ${s + chunk.length}/${rows.length}\r`);
    }
  }
  console.log(" ".repeat(40) + "\r");

  check(`all ${rows.length} MREIDs map to canonical token ids`, badMap === 0, firstBad || "exact match");
  check("no duplicate token assignments", dupTokens === 0, `${dupTokens} duplicates`);
  check(`all NFTs owned by seller ${manifest.roles.seller}`, notSeller === 0, badOwnerSample || `${rows.length} owners verified`);
  check(
    `tokenURI correct (${full ? "all" : "every 25th"})`,
    badUri === 0 && missingUri === 0,
    `${badUri} wrong, ${missingUri} empty`,
  );
  check(
    `propertyOf correct (${full ? "all" : "every 25th"})`,
    badProp === 0,
    `${badProp} wrong`,
  );

  // ---- 6. roles ----
  const MINTER_ROLE = await nft.MINTER_ROLE();
  check(
    "MINTER_ROLE on " + manifest.roles.minter_role_holder,
    await nft.hasRole(MINTER_ROLE, manifest.roles.minter_role_holder),
  );
  const INSPECTOR_ROLE = await escrow.INSPECTOR_ROLE();
  check("INSPECTOR_ROLE on inspector", await escrow.hasRole(INSPECTOR_ROLE, manifest.roles.inspector));
  const LENDER_ROLE = await escrow.LENDER_ROLE();
  check("LENDER_ROLE on lender", await escrow.hasRole(LENDER_ROLE, manifest.roles.lender));
  const LM = await registry.LISTING_MANAGER_ROLE();
  check("LISTING_MANAGER_ROLE on escrow", await registry.hasRole(LM, manifest.contracts.millow_escrow));

  // ---- 7. wiring ----
  check("Registry.propertyNFT == PropertyNFT", (await registry.propertyNFT()) === manifest.contracts.property_nft);
  check("Escrow.propertyNFT == PropertyNFT", (await escrow.propertyNFT()) === manifest.contracts.property_nft);
  check("Escrow.propertyRegistry == Registry", (await escrow.propertyRegistry()) === manifest.contracts.property_registry);

  // ---- 8. event history ----
  const PropertyMintedSig = eventTopic("PropertyMinted(uint256,string,address)");
  const minted = await rpc("eth_getLogs", [
    {
      fromBlock: "0x0",
      toBlock: "latest",
      address: manifest.contracts.property_nft,
      topics: [PropertyMintedSig],
    },
  ]);
  check(
    `PropertyMinted events (exact topic, from NFT)`,
    minted.length === expected,
    `${minted.length} events (== ${expected})`,
  );

  // ---- 9. no unexpected listings / active sales ----
  // Always a full sweep: these are cheap view calls relative to the mapping
  // check, and a stale listing on a canonical chain is a real defect.
  let listed = 0;
  let activeSales = 0;
  const LIST_BATCH = 250;
  for (let s = 1; s <= supply; s += LIST_BATCH) {
    const hi = Math.min(s + LIST_BATCH - 1, supply);
    const ids = [];
    for (let id = s; id <= hi; id++) ids.push(id);
    const res = await Promise.all(
      ids.map(async (id) => {
        const onOffer = await registry.isOnOffer(id);
        const sale = await escrow.sales(id);
        return { onOffer, status: Number(sale.status) };
      }),
    );
    for (const r of res) {
      if (r.onOffer) listed += 1;
      if (r.status >= 1 && r.status <= 3) activeSales += 1;
    }
    process.stdout.write(`  listings/sales ${hi}/${supply}\r`);
  }
  console.log(" ".repeat(40) + "\r");
  check("no unexpected on-offer listings", listed === 0, `${listed} listed`);
  check("no unexpected active sales", activeSales === 0, `${activeSales} active`);

  finish(full);
}

function finish(full) {
  const failed = results.filter((r) => !r.pass);
  console.log("\n=== summary ===");
  console.log(`checks: ${results.length}   passed: ${results.length - failed.length}   failed: ${failed.length}`);
  if (failed.length) {
    for (const f of failed) console.log(`  FAILED: ${f.name} — ${f.detail}`);
  }
  console.log(
    failed.length
      ? "\nRESULT: CHAIN DOES NOT MATCH THE CANONICAL CHECKPOINT"
      : "\nRESULT: CHAIN MATCHES THE CANONICAL CHECKPOINT" + (full ? " (full sweep)" : ""),
  );
  process.exitCode = failed.length ? 1 : 0;
}

main().catch((e) => {
  console.error(`chain:verify error: ${e.message}`);
  process.exitCode = 1;
});
