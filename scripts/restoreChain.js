// Imports a portable Anvil state artifact so another machine continues from an
// already-tokenized MILLOW chain.  This is the Computer 2 entry point.
//
//   npm run chain:import -- --artifact=<path-to-gz>
//   npm run chain:import -- --artifact=<path-to-gz> --force
//
// What it does, in order:
//   1. resolves and gunzips the artifact, checks the SHA-256 sidecar if present
//   2. proves the anvil version is the one this toolchain requires
//   3. places the artifact at .chain/state.json  (the file anvil --state loads)
//   4. refuses to overwrite an existing populated chain unless --force
//   5. starts Anvil (--state alone resumes it; see lib/chain anvilArgs)
//   6. verifies chainId, contract code, token count and the mapping manifest
//
// What it NEVER does:
//   * calls hardhat_reset / anvil_reset / evm_revert (denylisted in lib/chain)
//   * deploys a contract
//   * mints or transfers a token
//   * silently replaces a populated chain
//
// Importing state is not a transaction: the chain is reproduced bit-for-bit
// from the artifact, so the import costs 0 transactions on the target machine.
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const crypto = require("crypto");
const { spawn } = require("child_process");
const {
  loadManifest,
  loadTokenMap,
  anvilArgs,
  CHAIN_DIR,
  ensureChainDir,
  readPid,
  writePid,
  clearPid,
  pidAlive,
  isAnvilProcess,
  isRpcUp,
  rpcUrl,
  rpc,
} = require("./lib/chain");

const STATE_PATH = path.join(CHAIN_DIR, "state.json");

function arg(name) {
  const hit = process.argv.find((a) => a.startsWith(`--${name}=`));
  return hit ? hit.split("=").slice(1).join("=") : null;
}
const hasFlag = (name) => process.argv.includes(`--${name}`);

function decodeArtifact(src) {
  const buf = fs.readFileSync(src);
  let json;
  if (buf[0] === 0x1f && buf[1] === 0x8b) {
    json = zlib.gunzipSync(buf);
  } else {
    // Plain JSON is also accepted (that is what anvil --state wants).
    json = buf;
  }
  let parsed;
  try {
    parsed = JSON.parse(json.toString("utf8"));
  } catch (e) {
    throw new Error(
      `${src} is neither plain JSON nor gzipped JSON (${e.message}).\n` +
        "If it is a hex blob from `anvil_dumpState`, it must be gunzipped and hex-decoded first.",
    );
  }
  // Reject files that parse as JSON but are not an anvil state dump, so a
  // wrong path cannot be installed as a chain.
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error(`${src} is not an anvil state object.`);
  }
  for (const key of ["best_block_number", "accounts", "blocks"]) {
    if (!(key in parsed)) {
      throw new Error(
        `${src} is not an anvil state dump (missing "${key}").\n` +
          "Expected the output of `anvil_dumpState`, gunzipped to JSON.",
      );
    }
  }
  return { json, parsed };
}

async function main() {
  const manifest = loadManifest();
  const url = rpcUrl(manifest);
  const force = hasFlag("force");

  ensureChainDir();

  // ---- 1. refuse to touch a chain that is already running ----
  // These run before the artifact is opened: they are the real blockers, and
  // decoding a multi-MB artifact first would waste time and imply progress.
  const livePid = readPid();
  if (pidAlive(livePid) && isAnvilProcess(livePid)) {
    throw new Error(
      `Anvil is already running (pid ${livePid}). Stop it first: npm run chain:stop`,
    );
  }
  if (await isRpcUp(url)) {
    // A foreign node owns the port. Importing here would leave our state file
    // describing a chain that nothing is serving.
    throw new Error(
      `Something is already serving ${url} but it is not this repo's Anvil ` +
        `(no pid file at ${CHAIN_DIR}). Stop it first, or set MILLOW_RPC_URL.`,
    );
  }
  const existingState = fs.existsSync(STATE_PATH) ? fs.statSync(STATE_PATH) : null;
  if (existingState && !force) {
    throw new Error(
      `Refusing to overwrite an existing chain state:\n` +
        `  ${STATE_PATH} (${existingState.size} bytes, mtime ${existingState.mtime.toISOString()})\n` +
        `This looks like a populated MILLOW chain. Re-run with --force to replace it.`,
    );
  }
  if (existingState && force) {
    console.log(`--force       : will replace existing ${STATE_PATH}`);
  }

  // ---- 2. locate + validate the artifact ----
  let artifact = arg("artifact");
  if (!artifact) {
    const auto = path.join(
      CHAIN_DIR,
      `millow-anvil-state-${manifest.canonical_token_count}.json.gz`,
    );
    if (fs.existsSync(auto)) artifact = auto;
  }
  if (!artifact) {
    throw new Error(
      "No state artifact given.\n" +
        "  Usage: npm run chain:import -- --artifact=<file.json.gz>\n" +
        "Obtain it from the GitHub Release produced by `npm run chain:export`.",
    );
  }
  if (!fs.existsSync(artifact)) throw new Error(`Artifact not found: ${artifact}`);

  const artifactSha = crypto.createHash("sha256").update(fs.readFileSync(artifact)).digest("hex");
  console.log(`artifact     : ${artifact}`);
  console.log(`artifact sha : ${artifactSha}`);

  const sidecar = `${artifact}.sha256`;
  if (fs.existsSync(sidecar)) {
    const expected = fs.readFileSync(sidecar, "utf8").trim().split(/\s+/)[0];
    if (expected !== artifactSha) {
      throw new Error(
        `Artifact SHA-256 mismatch.\n  file:     ${artifactSha}\n  expected: ${expected}`,
      );
    }
    console.log("sha256 check : OK (sidecar matched)");
  } else {
    console.log("sha256 check : no sidecar found, cannot verify integrity");
  }
  if (manifest.state_artifact.sha256 && manifest.state_artifact.sha256 !== artifactSha) {
    console.log(
      `WARNING: chain-manifest.json records sha256 ${manifest.state_artifact.sha256}\n` +
        `         artifact on disk is            ${artifactSha}`,
    );
  }

  const { json, parsed } = decodeArtifact(artifact);
  console.log(`decoded      : ${json.length} bytes of JSON`);
  console.log(
    `state        : best block ${parsed.best_block_number}, ` +
      `${Object.keys(parsed.accounts || {}).length} accounts, ` +
      `${(parsed.blocks || []).length} blocks, ${(parsed.transactions || []).length} txs`,
  );

  // ---- 3. prove the toolchain works BEFORE overwriting the state file ----
  // A wrong/missing anvil must not leave a replaced .chain/state.json behind a
  // failed import, so the version gate runs before any destructive write.
  const bin = path.join(
    __dirname,
    "..",
    "node_modules",
    ".bin",
    process.platform === "win32" ? "anvil.cmd" : "anvil",
  );
  const binPath = process.env.MILLOW_ANVIL || (fs.existsSync(bin) ? bin : "anvil");
  const args = anvilArgs(manifest, { load: true });
  const { assertAnvilVersion } = require("./lib/chain");
  const version = assertAnvilVersion(binPath, manifest.node.version_required);
  console.log(`anvil version: ${version} (required ${manifest.node.version_required})`);

  // ---- 4. install the state file ----
  fs.writeFileSync(STATE_PATH, json);
  console.log(`installed    : ${STATE_PATH} (${json.length} bytes)`);

  // ---- 5. start anvil against it ----
  console.log(`starting     : ${binPath} ${args.join(" ")}`);
  const spawnError = await new Promise((resolve) => {
    const child = spawn(binPath, args, {
      detached: true,
      stdio: ["ignore", fs.openSync(path.join(CHAIN_DIR, "anvil.log"), "a"), fs.openSync(path.join(CHAIN_DIR, "anvil.err.log"), "a")],
    });
    child.on("error", (e) => resolve(e));
    child.on("spawn", () => resolve(null));
    if (child.pid) writePid(child.pid);
    child.unref();
  });
  if (spawnError) {
    clearPid();
    throw new Error(
      `Could not start "${binPath}": ${spawnError.message}\n` +
        `This toolchain requires Foundry Anvil ${manifest.node.version_required} (https://getfoundry.sh), ` +
        "or set MILLOW_ANVIL to the binary.",
    );
  }

  const deadline = Date.now() + 40000;
  let chainId = null;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 500));
    try {
      chainId = parseInt(await rpc("eth_chainId", [], url), 16);
      break;
    } catch {
      /* keep waiting */
    }
  }
  if (chainId === null) {
    clearPid();
    throw new Error("Anvil did not start within 40s. See .chain/anvil.err.log");
  }

  // ---- 6. verify the imported chain ----
  const block = parseInt(await rpc("eth_blockNumber", [], url), 16);
  console.log("\n=== imported chain ===");
  console.log(`chainId      : ${chainId}${chainId === manifest.chain_id ? " (correct)" : " *** WRONG ***"}`);
  console.log(`blockNumber  : ${block}`);
  if (chainId !== manifest.chain_id) {
    throw new Error(`Imported chain has chainId ${chainId}, expected ${manifest.chain_id}`);
  }

  const { ethers } = require("ethers");
  for (const [name, addr] of Object.entries(manifest.contracts)) {
    const code = await rpc("eth_getCode", [addr, "latest"], url);
    const bytes = code === "0x" ? 0 : (code.length - 2) / 2;
    const want = manifest.expected_code_bytes[name];
    const ok = want ? bytes === want : bytes > 0;
    console.log(`${name.padEnd(18)}: ${addr} code=${bytes} ${ok ? "ok" : "*** MISMATCH ***"}`);
    if (!ok) {
      throw new Error(`${name} bytecode mismatch at ${addr} (got ${bytes}, want ${want})`);
    }
  }

  const nftAbi = require("./lib/chain").abi("PropertyNFT", "PropertyNFT");
  const iface = new ethers.utils.Interface(nftAbi);
  const supplyData = iface.encodeFunctionData("totalSupply", []);
  const raw = await rpc("eth_call", [{ to: manifest.contracts.property_nft, data: supplyData }, "latest"], url);
  const supply = raw && raw !== "0x" ? ethers.BigNumber.from(raw).toNumber() : 0;
  console.log(`totalSupply  : ${supply} / ${manifest.canonical_token_count}`);
  if (supply !== manifest.canonical_token_count) {
    throw new Error(
      `Imported chain has totalSupply ${supply}, expected ${manifest.canonical_token_count}. ` +
        "The artifact does not match the canonical MILLOW checkpoint.",
    );
  }

  const rows = loadTokenMap(manifest);
  console.log(`manifest     : ${rows.length} rows ready for full verification`);
  console.log("\nImport complete. Run the full audit:  npm run chain:verify");
}

main().catch((e) => {
  console.error(`\nchain:import failed: ${e.message}`);
  process.exitCode = 1;
});
