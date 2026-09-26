// Exports the live chain state as a portable, distributable artifact.
//
//   npm run chain:export
//
// This is the Computer 1 -> Computer 2 hand-off step.  It performs exactly one
// RPC that touches the node, and that call only *reads* the chain:
//     anvil_dumpState      (returns hex-encoded gzip of the full state)
//
// The chain is not modified: no transaction is sent, no block is mined.
//
// Output: .chain/millow-anvil-state-<count>.json.gz  plus a .sha256 sidecar.
// The artifact is intentionally kept out of git (see .gitignore); publish it as
// a GitHub Release asset.
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const {
  loadManifest,
  CHAIN_DIR,
  ensureChainDir,
  rpc,
  sha256File,
} = require("./lib/chain");

async function main() {
  const manifest = loadManifest();
  const count = manifest.canonical_token_count;

  // ---- validate BEFORE dumping ----
  // A foreign, partial, or wrong-chain node must be refused before we spend
  // time serialising ~14 MB of state, and before any artifact is written.
  // This also catches a Hardhat node, which has no anvil_dumpState at all.
  const { attachReadOnly } = require("./lib/chain");
  const { provider, nft } = await attachReadOnly(manifest);
  const net = await provider.getNetwork();
  if (net.chainId !== manifest.chain_id) {
    throw new Error(
      `Refusing to export: chainId is ${net.chainId}, expected ${manifest.chain_id}`,
    );
  }
  for (const [name, addr] of Object.entries(manifest.contracts)) {
    const c = await provider.getCode(addr);
    const bytes = c === "0x" ? 0 : (c.length - 2) / 2;
    const want = manifest.expected_code_bytes[name];
    if (bytes !== want) {
      throw new Error(
        `Refusing to export: ${name} at ${addr} has ${bytes} bytes of code, expected ${want}. ` +
          "This is not the canonical MILLOW deployment.",
      );
    }
  }
  const supply = (await nft.totalSupply()).toNumber();
  if (supply !== count) {
    throw new Error(
      `Refusing to export: totalSupply is ${supply}, expected canonical ${count}. ` +
        "Run `npm run chain:verify` first; a partial chain must not be published.",
    );
  }

  ensureChainDir();

  // Record what is actually serving the chain right now, not what the manifest
  // hopes is installed.  web3_clientVersion is the running binary's own report.
  const clientVersion = await rpc("web3_clientVersion").catch(() => null);
  const liveBlock = parseInt(await rpc("eth_blockNumber"), 16);

  const hex = await rpc("anvil_dumpState");
  if (typeof hex !== "string" || !hex.startsWith("0x")) {
    throw new Error("anvil_dumpState did not return a hex string");
  }
  const gz = Buffer.from(hex.slice(2), "hex");

  // Sanity: must be gzip, and must gunzip to parseable JSON.
  const json = zlib.gunzipSync(gz);
  const parsed = JSON.parse(json.toString("utf8"));
  const best = Number(parsed.best_block_number ?? 0);

  const name = `millow-anvil-state-${count}.json.gz`;
  const out = path.join(CHAIN_DIR, name);
  fs.writeFileSync(out, gz);
  const sha = sha256File(out);
  fs.writeFileSync(`${out}.sha256`, `${sha}  ${name}\n`);

  // Write export metadata sidecar (do NOT mutate the canonical manifest or mapping)
  const meta = {
    exported_at: new Date().toISOString(),
    filename: name,
    // Relative, machine-independent location. An absolute path here would leak
    // the exporting machine's directory layout into a published Release asset.
    relative_location: `${path.basename(CHAIN_DIR)}/${name}`,
    bytes_gz: gz.length,
    bytes_json: json.length,
    compression_pct: gz.length && json.length ? ((gz.length / json.length) * 100).toFixed(1) : null,
    sha256: sha,
    chain_id: net.chainId,
    block_number: liveBlock,
    dumped_best_block: best,
    total_supply: supply,
    canonical_token_count: count,
    contracts: manifest.contracts,
    anvil_version_required: manifest.node.version_required,
    anvil_version_running: clientVersion,
    node_software: manifest.node.software,
  };
  fs.writeFileSync(`${out}.meta.json`, JSON.stringify(meta, null, 2) + "\n");

  console.log("=== state artifact exported ===");
  console.log(`file        : ${out}`);
  console.log(`gzip bytes  : ${gz.length} (${(gz.length / 1048576).toFixed(2)} MB)`);
  console.log(`json bytes  : ${json.length} (${(json.length / 1048576).toFixed(2)} MB)`);
  console.log(`compression : ${((gz.length / json.length) * 100).toFixed(1)}% of raw JSON`);
  console.log(`live block   : ${liveBlock}`);
  console.log(`dumped block : ${best}`);
  console.log(`client       : ${clientVersion || "(unknown)"}`);
  console.log(`accounts    : ${Object.keys(parsed.accounts || {}).length}`);
  console.log(`blocks      : ${(parsed.blocks || []).length}`);
  console.log(`txs         : ${(parsed.transactions || []).length}`);
  console.log(`sha256      : ${sha}`);
  console.log(`sidecar     : ${out}.sha256`);
  console.log(`metadata    : ${out}.meta.json`);
  console.log("");
  console.log("Next:");
  console.log(`  1. record the sha256 in chain-manifest.json state_artifact.sha256`);
  console.log(`  2. attach ${name} to a GitHub Release (do NOT commit it)`);
  console.log(`  3. on Computer 2: verify sha256, gunzip to JSON, place at .chain/state.json`);
}

main().catch((e) => {
  console.error(`chain:export failed: ${e.message}`);
  process.exitCode = 1;
});
