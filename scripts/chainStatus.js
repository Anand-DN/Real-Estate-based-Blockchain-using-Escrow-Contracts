// Reports the state of the persistent MILLOW local chain.
//
//   npm run chain:status
//
// READ-ONLY: RPC calls are limited to eth_chainId / eth_blockNumber /
// eth_getCode / view functions.  Never sends a transaction.
const fs = require("fs");
const path = require("path");
const {
  loadManifest,
  loadTokenMap,
  manifestSha256,
  CHAIN_DIR,
  readPid,
  pidAlive,
  isAnvilProcess,
  isRpcUp,
  rpcUrl,
  rpc,
  sha256File,
} = require("./lib/chain");

async function main() {
  const manifest = loadManifest();
  const url = rpcUrl(manifest);
  const statePath = path.join(CHAIN_DIR, "state.json");

  console.log("=== MILLOW persistent chain status ===");
  console.log(`manifest      : chain-manifest.json (chainId ${manifest.chain_id})`);
  console.log(`rpc           : ${url}`);
  console.log(`expected count: ${manifest.canonical_token_count}`);

  const pid = readPid();
  const managed = pidAlive(pid) && isAnvilProcess(pid);
  console.log(`anvil pid     : ${pid || "(none)"}${managed ? " (running)" : pid ? " (stale)" : ""}`);

  if (!fs.existsSync(statePath)) {
    console.log(`state file    : absent (${statePath})`);
  } else {
    const st = fs.statSync(statePath);
    console.log(
      `state file    : ${st.size} bytes, mtime ${st.mtime.toISOString()}, sha256 ${sha256File(statePath).slice(0, 16)}...`,
    );
  }

  if (!(await isRpcUp(url))) {
    console.log("rpc           : DOWN  (start it with: npm run chain:start)");
    return;
  }
  console.log("rpc           : UP");

  const chainId = parseInt(await rpc("eth_chainId", [], url), 16);
  const block = parseInt(await rpc("eth_blockNumber", [], url), 16);
  console.log(`chainId       : ${chainId}${chainId === manifest.chain_id ? " (correct)" : " *** WRONG ***"}`);
  console.log(`blockNumber   : ${block}`);

  for (const [name, addr] of Object.entries(manifest.contracts)) {
    const code = await rpc("eth_getCode", [addr, "latest"], url);
    const bytes = code === "0x" ? 0 : (code.length - 2) / 2;
    const want = manifest.expected_code_bytes[name];
    const ok = want ? bytes === want : bytes > 0;
    console.log(
      `${name.padEnd(18)}: ${addr} code=${bytes}${want ? `/${want}` : ""} ${ok ? "ok" : "*** MISMATCH ***"}`,
    );
  }

  // totalSupply via a raw eth_call so no signing or artifact deploy is needed.
  const { ethers } = require("ethers");
  const { abi } = require("./lib/chain");
  const iface = new ethers.utils.Interface(abi("PropertyNFT", "PropertyNFT"));
  const data = iface.encodeFunctionData("totalSupply", []);
  const raw = await rpc("eth_call", [{ to: manifest.contracts.property_nft, data }, "latest"], url);
  const supply = raw && raw !== "0x" ? ethers.BigNumber.from(raw).toNumber() : 0;
  console.log(`totalSupply   : ${supply} / ${manifest.canonical_token_count}`);

  const rows = loadTokenMap(manifest);
  const sha = manifestSha256(manifest);
  console.log(`manifest rows : ${rows.length}`);
  console.log(`manifest sha  : ${sha}`);
  console.log(
    `manifest ok   : ${sha === manifest.mapping_manifest.sha256 ? "matches chain-manifest.json" : "*** CHANGED ***"}`,
  );
  if (supply !== manifest.canonical_token_count) {
    console.log("\nRun `npm run chain:verify` for a full read-only audit.");
  }
}

main().catch((e) => {
  console.error(`chain:status failed: ${e.message}`);
  process.exitCode = 1;
});
