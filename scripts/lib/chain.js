// Shared helpers for the persistent MILLOW chain toolchain.
//
// Every helper here is read-only unless the function name says otherwise.
// rpc() enforces a denylist so a destructive chain-wipe can never be issued by
// accident from any script in this toolchain.
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");

const ROOT = path.join(__dirname, "..", "..");
const MANIFEST_PATH = path.join(ROOT, "chain-manifest.json");
const ARTIFACTS = path.join(ROOT, "artifacts", "contracts");
const CHAIN_DIR = path.join(ROOT, ".chain");
const PID_FILE = path.join(CHAIN_DIR, "anvil.pid");

// Methods that wipe or roll back chain state.  restoreChain/verifyChain/
// tokenizeChain must never issue these; the guard makes that structural.
const FORBIDDEN_RPC = new Set([
  "hardhat_reset",
  "anvil_reset",
  "evm_revert",
  "hardhat_setBalance",
  "anvil_setBalance",
  "hardhat_setCode",
  "anvil_setCode",
  "hardhat_setStorageAt",
  "anvil_setStorageAt",
]);

function loadManifest() {
  if (!fs.existsSync(MANIFEST_PATH)) {
    throw new Error(`Missing chain manifest: ${MANIFEST_PATH}`);
  }
  return JSON.parse(fs.readFileSync(MANIFEST_PATH, "utf8"));
}

function rpcUrl(manifest) {
  return process.env.MILLOW_RPC_URL || manifest.rpc_url;
}

// JSON-RPC passthrough with the destructive-method denylist enforced.
async function rpc(method, params = [], url = null) {
  if (FORBIDDEN_RPC.has(method)) {
    throw new Error(
      `Refusing to call forbidden method "${method}". ` +
        "This toolchain never resets or rolls back chain state.",
    );
  }
  const target = url || rpcUrl(loadManifest());
  const res = await fetch(target, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }),
  }).then((r) => r.json());
  if (res.error) {
    const e = new Error(`${method}: ${res.error.message}`);
    e.rpcMethod = method;
    throw e;
  }
  return res.result;
}

// token_id -> mreid, in canonical token order.
// Builds the anvil argument list from the manifest, rewriting the manifest's
// relative ".chain/state.json" to an absolute path so the node does not depend
// on the working directory.
//
// Anvil 1.8.3 makes --dump-state and --load-state MUTUALLY EXCLUSIVE with
// --state ("the argument '--state <PATH>' cannot be used with '--dump-state
// <PATH>'").  --state already does everything the other two were meant to do:
// it loads the file if it exists and dumps it on exit plus every
// --state-interval seconds.  So the correct invocation is --state alone.
// (Verified against anvil 1.8.3: --state resumes a saved chain and persists it
// on a 5s interval; adding either --dump-state or --load-state aborts startup.)
function anvilArgs(manifest, opts = {}) {
  const statePath = path.join(CHAIN_DIR, "state.json");
  const flags = manifest.node.state_flags;
  const i = flags.indexOf("--state-interval");
  const interval = i >= 0 && flags[i + 1] ? flags[i + 1] : "30";
  const args = [
    ...manifest.node.start_flags,
    "--state",
    statePath,
    "--state-interval",
    interval,
  ];
  return args;
}

function loadTokenMap(manifest) {
  const rel = manifest.mapping_manifest.path;
  const p = path.isAbsolute(rel) ? rel : path.join(ROOT, rel);
  if (!fs.existsSync(p)) {
    throw new Error(
      `Missing mapping manifest: ${p}\n` +
        "Regenerate it with: node scripts/buildTokenMap.js",
    );
  }
  const text = fs.readFileSync(p, "utf8");
  const lines = text.split(/\r?\n/);
  if (lines[0].trim() !== "token_id,mreid_id") {
    throw new Error(`Unexpected manifest header: ${lines[0]}`);
  }
  const rows = [];
  for (let i = 1; i < lines.length; i++) {
    const t = lines[i].trim();
    if (!t) continue;
    const c = t.indexOf(",");
    rows.push({
      tokenId: Number(t.slice(0, c)),
      mreid: t.slice(c + 1),
    });
  }
  const expected = manifest.canonical_token_count;
  if (rows.length !== expected) {
    throw new Error(
      `Manifest has ${rows.length} rows but chain-manifest.json expects ${expected}.`,
    );
  }
  for (let i = 0; i < rows.length; i++) {
    if (rows[i].tokenId !== i + 1) {
      throw new Error(`Manifest is not in canonical order at row ${i + 1}`);
    }
  }
  return rows;
}

function manifestSha256(manifest) {
  const rel = manifest.mapping_manifest.path;
  const p = path.isAbsolute(rel) ? rel : path.join(ROOT, rel);
  return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
}

function loadArtifact(sol, name) {
  const p = path.join(ARTIFACTS, `${sol}.sol`, `${name}.json`);
  if (!fs.existsSync(p)) {
    throw new Error(
      `Missing compiled artifact: ${p}\n` +
        "Run: npx hardhat compile",
    );
  }
  const j = JSON.parse(fs.readFileSync(p, "utf8"));
  return { abi: j.abi, bytecode: j.bytecode };
}

function abi(sol, name) {
  return loadArtifact(sol, name).abi;
}

async function getProvider(manifest) {
  const { ethers } = require(path.join(ROOT, "node_modules", "ethers"));
  return new ethers.providers.JsonRpcProvider(rpcUrl(manifest), undefined, {
    staticNetwork: true,
  });
}

// Attaches the three MILLOW contracts read-only (no signer) for verification.
async function attachReadOnly(manifest) {
  const { ethers } = require(path.join(ROOT, "node_modules", "ethers"));
  const provider = await getProvider(manifest);
  const c = manifest.contracts;
  return {
    provider,
    nft: new ethers.Contract(c.property_nft, abi("PropertyNFT", "PropertyNFT"), provider),
    registry: new ethers.Contract(
      c.property_registry,
      abi("PropertyRegistry", "PropertyRegistry"),
      provider,
    ),
    escrow: new ethers.Contract(
      c.millow_escrow,
      abi("MillowEscrow", "MillowEscrow"),
      provider,
    ),
  };
}

function tokenUriFor(mreid) {
  return `http://localhost:3000/metadata/millow/${mreid}.json`;
}

function ensureChainDir() {
  fs.mkdirSync(CHAIN_DIR, { recursive: true });
  return CHAIN_DIR;
}

function readPid() {
  try {
    return Number(fs.readFileSync(PID_FILE, "utf8").trim());
  } catch {
    return null;
  }
}

function writePid(pid) {
  ensureChainDir();
  fs.writeFileSync(PID_FILE, String(pid));
}

function clearPid() {
  try {
    fs.unlinkSync(PID_FILE);
  } catch {
    /* already gone */
  }
}

function pidAlive(pid) {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function isAnvilProcess(pid) {
  try {
    const { execFileSync } = require("child_process");
    const out = execFileSync("powershell.exe", [
      "-NoProfile",
      "-Command",
      `(Get-CimInstance Win32_Process -Filter "ProcessId=${pid}" -ErrorAction SilentlyContinue).Name`,
    ]).toString();
    return /anvil/i.test(out);
  } catch {
    return false;
  }
}

async function isRpcUp(url) {
  try {
    await rpc("eth_chainId", [], url);
    return true;
  } catch {
    return false;
  }
}

function sha256File(p) {
  return crypto.createHash("sha256").update(fs.readFileSync(p)).digest("hex");
}

// Reproducibility depends on the exact foundry release, because state-dump
// layout and default account derivation have changed between versions.  Runs
// `<bin> --version` and compares the first semver it finds against the manifest.
//
// Deliberately strict: a version that differs, or that cannot be parsed, is a
// hard failure.  Set MILLOW_SKIP_VERSION_CHECK=1 to bypass, which is logged
// loudly and is the only escape hatch.
function anvilVersion(bin) {
  const { execFileSync } = require("child_process");
  const opts = { encoding: "utf8", timeout: 15000, windowsHide: true };
  let out;
  if (process.platform === "win32" && /\.(cmd|bat)$/i.test(bin)) {
    // A .cmd shim cannot be exec'd directly on Windows (EINVAL), which is the
    // exact path resolveAnvil() returns on win32, so route it via cmd.exe with
    // an explicit argument vector.  shell:true would concatenate the args
    // unescaped (DEP0190), so build the command line by hand instead.
    out = execFileSync(
      process.env.ComSpec || "cmd.exe",
      ["/d", "/s", "/c", `""${bin}" --version"`],
      { ...opts, windowsVerbatimArguments: true },
    );
  } else {
    out = execFileSync(bin, ["--version"], opts);
  }
  const m = out.match(/(\d+\.\d+\.\d+)/);
  if (!m) throw new Error(`Could not parse anvil version from: ${out.trim()}`);
  return m[1];
}

function assertAnvilVersion(bin, required) {
  if (process.env.MILLOW_SKIP_VERSION_CHECK === "1") {
    console.warn(
      `[version check SKIPPED by MILLOW_SKIP_VERSION_CHECK=1] expected anvil ${required}`,
    );
    return null;
  }
  let found;
  try {
    found = anvilVersion(bin);
  } catch (e) {
    throw new Error(
      `Could not determine the anvil version via "${bin} --version" (${e.message}).\n` +
        `Reproducibility requires anvil ${required}. Install it from https://getfoundry.sh,\n` +
        "or set MILLOW_ANVIL to the right binary, or set MILLOW_SKIP_VERSION_CHECK=1 to override.",
    );
  }
  if (found !== required) {
    throw new Error(
      `anvil version mismatch: found ${found}, this toolchain requires ${required}.\n` +
        "A different foundry release can change state-dump layout and account\n" +
        "derivation, which breaks the reproducibility guarantee. Install anvil " +
        `${required}, set MILLOW_ANVIL, or set MILLOW_SKIP_VERSION_CHECK=1 to override.`,
    );
  }
  return found;
}

// keccak256 of an event signature, for topic-filtered log queries.
function eventTopic(signature) {
  const { ethers } = require(path.join(ROOT, "node_modules", "ethers"));
  return ethers.utils.id(signature);
}

module.exports = {
  ROOT,
  CHAIN_DIR,
  PID_FILE,
  FORBIDDEN_RPC,
  loadManifest,
  rpcUrl,
  rpc,
  anvilArgs,
  loadTokenMap,
  manifestSha256,
  loadArtifact,
  abi,
  getProvider,
  attachReadOnly,
  tokenUriFor,
  ensureChainDir,
  readPid,
  writePid,
  clearPid,
  pidAlive,
  isAnvilProcess,
  isRpcUp,
  sha256File,
  anvilVersion,
  assertAnvilVersion,
  eventTopic,
};
