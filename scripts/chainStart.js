// Starts the persistent MILLOW local chain (Anvil) in the background.
//
//   npm run chain:start
//
// The chain state lives in .chain/state.json and is loaded if present, so a
// restart resumes exactly where it stopped.  Nothing here deploys, mints or
// otherwise modifies chain content.
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const {
  loadManifest,
  CHAIN_DIR,
  ensureChainDir,
  anvilArgs,
  assertAnvilVersion,
  readPid,
  writePid,
  clearPid,
  pidAlive,
  isAnvilProcess,
  isRpcUp,
  rpcUrl,
  rpc,
} = require("./lib/chain");

const LOG = path.join(CHAIN_DIR, "anvil.log");
const ERR = path.join(CHAIN_DIR, "anvil.err.log");

function resolveAnvil() {
  if (process.env.MILLOW_ANVIL) return process.env.MILLOW_ANVIL;
  const local = path.join(
    __dirname,
    "..",
    "node_modules",
    ".bin",
    process.platform === "win32" ? "anvil.cmd" : "anvil",
  );
  if (fs.existsSync(local)) return local;
  // Fall back to whatever is on PATH.
  return "anvil";
}

async function main() {
  const manifest = loadManifest();
  const url = rpcUrl(manifest);
  ensureChainDir();

  const existing = readPid();
  if (pidAlive(existing) && isAnvilProcess(existing)) {
    const net = await rpc("eth_chainId", [], url).catch(() => null);
    console.log(`Anvil already running (pid ${existing}, chainId ${net}).`);
    console.log(`RPC: ${url}`);
    return;
  }
  clearPid();

  if (await isRpcUp(url)) {
    throw new Error(
      `Something else is already listening on ${url} and it is not our Anvil ` +
        `(no pid file at ${CHAIN_DIR}). Stop it first, or set MILLOW_RPC_URL.`,
    );
  }

  const bin = resolveAnvil();
  const args = anvilArgs(manifest);
  const statePath = path.join(CHAIN_DIR, "state.json");
  const hadState = fs.existsSync(statePath);

  console.log(`Starting: ${bin} ${args.join(" ")}`);
  try {
    const version = assertAnvilVersion(bin, manifest.node.version_required);
    console.log(`anvil version: ${version} (required ${manifest.node.version_required})`);
  } catch (e) {
    clearPid();
    throw e;
  }
  const out = fs.openSync(LOG, "a");
  const err = fs.openSync(ERR, "a");

  // spawn() reports a missing binary asynchronously, so the error must be
  // captured or the readiness poll below times out with a misleading message.
  const spawnError = await new Promise((resolve) => {
    const child = spawn(bin, args, { detached: true, stdio: ["ignore", out, err] });
    child.on("error", (e) => resolve(e));
    child.on("spawn", () => resolve(null));
    if (child.pid) {
      writePid(child.pid);
      console.log(`pid ${child.pid}  log ${LOG}`);
    }
    child.unref();
  });

  if (spawnError) {
    clearPid();
    throw new Error(
      `Could not start "${bin}": ${spawnError.message}\n` +
        `This toolchain requires Foundry Anvil ${manifest.node.version_required}.\n` +
        "  - install Foundry  (https://getfoundry.sh) so `anvil --version` works, or\n" +
        "  - point MILLOW_ANVIL at an existing anvil binary.",
    );
  }

  // Wait for the RPC to answer.
  const deadline = Date.now() + 30000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      const chainId = await rpc("eth_chainId", [], url);
      const block = await rpc("eth_blockNumber", [], url);
      console.log(`Anvil up. chainId ${parseInt(chainId, 16)}, block ${parseInt(block, 16)}`);
      console.log(`state file: ${hadState ? "present, chain resumed" : "absent, starting a fresh chain"}`);
      console.log(`RPC: ${url}`);
      return;
    } catch {
      /* keep waiting */
    }
  }
  throw new Error(`Anvil did not become ready within 30s. See ${LOG} and ${ERR}.`);
}

main().catch((e) => {
  console.error(`chain:start failed: ${e.message}`);
  process.exitCode = 1;
});
