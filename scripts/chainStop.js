// Stops the persistent MILLOW local chain.
//
//   npm run chain:stop
//
// Anvil rewrites .chain/state.json every --state-interval seconds, so the state
// on disk is at most that interval behind the chain.  Windows cannot deliver
// SIGTERM to a detached process, so this does not rely on an exit-time dump.
// For a guaranteed-clean artifact use `npm run chain:export` first.
//
// This script never calls a chain RPC that modifies state.
const fs = require("fs");
const path = require("path");
const {
  readPid,
  clearPid,
  pidAlive,
  isAnvilProcess,
  isRpcUp,
  loadManifest,
  rpcUrl,
  CHAIN_DIR,
} = require("./lib/chain");

async function main() {
  const manifest = loadManifest();
  const url = rpcUrl(manifest);
  const statePath = path.join(CHAIN_DIR, "state.json");

  const pid = readPid();
  if (!pid) {
    console.log("No pid file; chain not managed by this repo.");
    console.log(`RPC reachable: ${await isRpcUp(url)}`);
    return;
  }
  if (!pidAlive(pid)) {
    console.log(`Stale pid file (pid ${pid} not running); clearing.`);
    clearPid();
    return;
  }
  if (!isAnvilProcess(pid)) {
    console.log(
      `Refusing to stop pid ${pid}: it is not anvil. Clear ${CHAIN_DIR} manually if needed.`,
    );
    process.exitCode = 1;
    return;
  }

  const before = fs.existsSync(statePath) ? fs.statSync(statePath).mtime : null;
  console.log(`Stopping anvil pid ${pid} ...`);
  try {
    process.kill(pid);
  } catch (e) {
    console.error(`kill failed: ${e.message}`);
    process.exitCode = 1;
    return;
  }

  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    if (!pidAlive(pid)) break;
    await new Promise((r) => setTimeout(r, 400));
  }
  clearPid();
  console.log(`Stopped. RPC reachable: ${await isRpcUp(url)}`);
  if (before) {
    const after = fs.existsSync(statePath) ? fs.statSync(statePath).mtime : null;
    console.log(
      `state.json mtime before stop: ${before.toISOString()}` +
        (after ? `  after: ${after.toISOString()}` : "  (file gone)"),
    );
  } else {
    console.log("no state.json existed (chain had no persisted state yet)");
  }
}

main().catch((e) => {
  console.error(`chain:stop failed: ${e.message}`);
  process.exitCode = 1;
});
