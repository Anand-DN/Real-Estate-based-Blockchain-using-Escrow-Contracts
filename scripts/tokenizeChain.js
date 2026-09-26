// Computer 1 builder: mints the canonical 29,135-property MILLOW catalogue in
// CANONICAL TOKEN ORDER so the resulting state is bit-reproducible.
//
//   npx hardhat run scripts/deployMillow.js --network localhost   (once)
//   npm run chain:tokenize
//   npm run chain:verify
//   npm run chain:export
//
// Determinism: PropertyNFT.mintProperty assigns token ids from an incrementing
// counter in EXECUTION ORDER, so token ids follow NONCE order, not dataset
// order.  This script therefore mints strictly in manifest order
// (data/processed/millow_token_map.csv, which encodes the 1,396 intentional
// permutations) using consecutive nonces inside a single block per batch.
// 29,135 MREIDs must therefore never be assumed to satisfy
// tokenId == numeric suffix of the MREID.
//
// Resume: token ids are assigned sequentially, so a rebuild is only
// deterministic while the already-minted prefix matches the manifest exactly.
// The script walks the manifest from token 1, accepts the longest correct
// prefix, and continues from the first divergence.  An inconsistent chain is a
// hard error — it is never "fixed" by re-minting, because that would corrupt
// the mapping.
//
// Safety: refuses to run if the MILLOW contracts are absent (deploy them with
// deployMillow.js first), never calls a reset method, and never mints a property
// that already has a token.
const fs = require("fs");
const path = require("path");
const {
  loadManifest,
  loadTokenMap,
  loadArtifact,
  tokenUriFor,
  CHAIN_DIR,
  rpc,
  rpcUrl,
  isRpcUp,
} = require("./lib/chain");

const BATCH = Number(process.env.MILLOW_TOKENIZE_BATCH) > 0
  ? Number(process.env.MILLOW_TOKENIZE_BATCH)
  : 100;
const GAS_MINT = 400000;
const GAS_TRANSFER = 120000;

const rpcUrlOverride = () => process.env.MILLOW_RPC_URL || null;

// Pure resume planner, exported for testing.  `lookup(mreid)` must resolve to the
// token id already minted for that MREID, or 0 when it is not minted.
//
// Token ids are assigned sequentially, so a rebuild stays deterministic only
// while the already-minted prefix matches the manifest exactly.  This walks the
// manifest from token 1 and accepts the longest correct prefix; anything else is
// a hard error, because token ids cannot be repaired by minting.
async function planTokenization({ total, supply0, lookup, concurrency = 24 }) {
  if (supply0 > total) {
    throw new Error(
      `Chain already holds ${supply0} tokens, more than the canonical ${total}.`,
    );
  }
  // On a fresh chain nothing is minted, so the correct prefix is 0 by
  // definition.  Seeding with supply0 (not total) is what makes a fresh chain
  // start at row 0 instead of being mistaken for "already complete".
  let matchedPrefix = supply0;

  if (supply0 > 0) {
    outer: for (let s = 0; s < total; s += concurrency) {
      const chunk = [];
      for (let i = s; i < Math.min(s + concurrency, total); i++) chunk.push(i);
      const got = await Promise.all(
        chunk.map(async (i) => {
          try {
            return { i, id: await lookup(i) };
          } catch {
            return { i, id: 0 };
          }
        }),
      );
      for (const g of got) {
        if (g.id !== g.i + 1) {
          matchedPrefix = g.i;
          break outer;
        }
      }
    }
    if (matchedPrefix !== supply0) {
      throw new Error(
        `Inconsistent chain: the longest prefix matching the canonical manifest is ` +
          `${matchedPrefix} tokens, but totalSupply is ${supply0}.\n` +
          "Token ids are sequential, so this chain cannot be repaired by minting. " +
          "Start from an empty chain and re-run deploy + tokenize.",
      );
    }
  }

  return {
    resumeAt: matchedPrefix,
    todo: total - matchedPrefix,
    complete: matchedPrefix >= total,
  };
}

async function main() {
  const manifest = loadManifest();
  const url = rpcUrlOverride() || rpcUrl(manifest);
  const { ethers } = require("ethers");
  const expected = manifest.canonical_token_count;
  const seller = manifest.roles.seller;
  const minterAddr = manifest.roles.minter_role_holder;

  if (!(await isRpcUp(url))) {
    throw new Error(`No chain at ${url}. Start it: npm run chain:start`);
  }

  const provider = new ethers.providers.JsonRpcProvider(url, undefined, {
    staticNetwork: true,
  });
  const net = await provider.getNetwork();
  if (net.chainId !== manifest.chain_id) {
    throw new Error(`chainId ${net.chainId}, expected ${manifest.chain_id}`);
  }

  // Contracts must already exist. This script never deploys.
  const nftDef = loadArtifact("PropertyNFT", "PropertyNFT");
  const nft = new ethers.Contract(manifest.contracts.property_nft, nftDef.abi, provider);
  const code = await provider.getCode(manifest.contracts.property_nft);
  if (code === "0x") {
    throw new Error(
      "PropertyNFT is not deployed at " + manifest.contracts.property_nft + ".\n" +
        "Deploy the MILLOW layer first: npx hardhat run scripts/deployMillow.js --network localhost",
    );
  }
  const supply0 = (await nft.totalSupply()).toNumber();
  console.log(`chain        : ${url} (chainId ${net.chainId})`);
  console.log(`PropertyNFT  : ${manifest.contracts.property_nft}`);
  console.log(`existing supply: ${supply0} / ${expected}`);

  const rows = loadTokenMap(manifest);
  const minterSigner = provider.getSigner(minterAddr);
  const MINTER_ROLE = await nft.MINTER_ROLE();
  if (!(await nft.hasRole(MINTER_ROLE, minterAddr))) {
    throw new Error(`MINTER_ROLE is not held by ${minterAddr}. Re-run deployMillow.js.`);
  }

  // ---- find the longest correct prefix, then plan the remaining work ----
  const lookup = async (i) =>
    (await nft.tokenByProperty(rows[i].mreid)).toNumber();
  const plan = await planTokenization({ total: rows.length, supply0, lookup });

  if (plan.complete) {
    console.log(
      `Nothing to do: the chain already holds the full canonical catalogue ` +
        `(${supply0}/${expected}).`,
    );
    return;
  }
  if (supply0 > 0) {
    console.log(`verified prefix: all ${supply0} existing tokens match the manifest`);
  }
  const resumeAt = plan.resumeAt;
  const todo = rows.slice(resumeAt);
  console.log(`to mint       : ${todo.length} (tokens ${resumeAt + 1}..${rows.length})`);
  console.log(`batch         : ${BATCH} per block\n`);

  const started = Date.now();
  for (let i = 0; i < todo.length; i += BATCH) {
    const batch = todo.slice(i, i + BATCH);

    // Re-check: never mint something that already has a token.
    for (const r of batch) {
      const existing = (await nft.tokenByProperty(r.mreid)).toNumber();
      if (existing !== 0) {
        throw new Error(
          `${r.mreid} already has token #${existing}, expected #${r.tokenId}. Refusing to continue.`,
        );
      }
    }

    // ---- mint pass: consecutive nonces, one block ----
    await rpc("evm_setAutomine", [false], url);
    let nonce = await minterSigner.getTransactionCount();
    for (const r of batch) {
      const data = nft.interface.encodeFunctionData("mintProperty", [
        r.mreid,
        tokenUriFor(r.mreid),
      ]);
      await minterSigner.sendTransaction({
        to: nft.address,
        data,
        nonce: nonce++,
        gasLimit: GAS_MINT,
      });
    }
    await rpc("evm_mine", [], url);
    await rpc("evm_setAutomine", [true], url);

    // ---- verify the batch got exactly the canonical ids ----
    for (const r of batch) {
      const id = (await nft.tokenByProperty(r.mreid)).toNumber();
      if (id !== r.tokenId) {
        throw new Error(
          `Token id divergence: ${r.mreid} got #${id}, canonical is #${r.tokenId}.`,
        );
      }
    }

    // ---- transfer pass: deliver to the canonical seller ----
    await rpc("evm_setAutomine", [false], url);
    nonce = await minterSigner.getTransactionCount();
    for (const r of batch) {
      const data = nft.interface.encodeFunctionData("transferFrom", [
        minterAddr,
        seller,
        r.tokenId,
      ]);
      await minterSigner.sendTransaction({
        to: nft.address,
        data,
        nonce: nonce++,
        gasLimit: GAS_TRANSFER,
      });
    }
    await rpc("evm_mine", [], url);
    await rpc("evm_setAutomine", [true], url);

    // ---- confirm ownership ----
    for (const r of batch) {
      const owner = (await nft.ownerOf(r.tokenId)).toLowerCase();
      if (owner !== seller.toLowerCase()) {
        throw new Error(`Token #${r.tokenId} owner is ${owner}, expected the seller.`);
      }
    }

    const done = resumeAt + i + batch.length;
    const pct = ((done / rows.length) * 100).toFixed(1);
    const el = ((Date.now() - started) / 1000).toFixed(0);
    process.stdout.write(
      `[${done}/${rows.length}] ${pct}%  ${el}s  block ${await provider.getBlockNumber()}\n`,
    );
  }

  const finalSupply = (await nft.totalSupply()).toNumber();
  console.log(`\ndone in ${((Date.now() - started) / 1000).toFixed(0)}s`);
  console.log(`totalSupply: ${finalSupply} / ${expected}`);
  console.log(`state file : ${path.join(CHAIN_DIR, "state.json")} (rewritten every --state-interval)`);
  console.log("\nNext: npm run chain:verify   then   npm run chain:export");
}

// Only mint when run as a command.  Requiring this file (e.g. from a test)
// exposes planTokenization without touching a chain.
if (require.main === module) {
  main().catch((e) => {
    console.error(`\nchain:tokenize failed: ${e.message}`);
    process.exitCode = 1;
  });
}

module.exports = { planTokenization };
