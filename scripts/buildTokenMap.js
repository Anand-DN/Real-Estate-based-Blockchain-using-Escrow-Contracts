// Builds the canonical MREID -> tokenId mapping manifest that every other
// reproducibility step depends on.
//
// Source of truth: data/processed/chain_index.json, the exported snapshot of
// the original 29,135-property MILLOW chain.  The manifest is written in
// CANONICAL TOKEN ORDER (token_id 1..N), which is what preserves the 1,396
// intentional token-id permutations.  MREID numeric suffix is NOT the token id
// and must never be assumed to be.
//
// Output: data/processed/millow_token_map.csv
//         token_id,mreid_id          (29,135 data rows + header)
//
// This script is a one-time provenance helper.  It only reads the snapshot and
// writes the manifest; it never touches a chain.
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const SNAPSHOT = path.join(ROOT, "data", "processed", "chain_index.json");
const OUT = path.join(ROOT, "data", "processed", "millow_token_map.csv");
const DATASET = path.join(ROOT, "data", "processed", "MREID_property.csv");

function main() {
  if (!fs.existsSync(SNAPSHOT)) {
    throw new Error(
      `Snapshot not found: ${SNAPSHOT}\n` +
        "It is a generated artifact (gitignored). Regenerate it from a chain " +
        "with: npx hardhat run scripts/exportChainIndex.js --network localhost",
    );
  }
  const snap = JSON.parse(fs.readFileSync(SNAPSHOT, "utf8"));
  const entries = Object.entries(snap.properties);
  if (entries.length === 0) throw new Error("Snapshot has no properties.");

  const total = snap.counts.total_properties;
  const byToken = new Array(total + 1);
  for (const [mreid, v] of entries) {
    const id = v.token_id;
    if (byToken[id] !== undefined) {
      throw new Error(`Duplicate token_id ${id}: ${byToken[id]} and ${mreid}`);
    }
    byToken[id] = mreid;
  }
  const missing = [];
  for (let i = 1; i <= total; i++) if (!byToken[i]) missing.push(i);
  if (missing.length) {
    throw new Error(
      `Snapshot does not cover token ids 1..${total}. Missing ${missing.length}: ` +
        missing.slice(0, 10).join(","),
    );
  }

  // Cross-check against the dataset: the manifest must describe exactly the
  // dataset rows, nothing added, nothing dropped.
  const datasetIds = new Set();
  for (const line of fs.readFileSync(DATASET, "utf8").split(/\r?\n/).slice(1)) {
    const t = line.trim();
    if (!t) continue;
    const c = t.indexOf(",");
    const m = c === -1 ? t : t.slice(0, c);
    if (/^MREID_\d+$/.test(m)) datasetIds.add(m);
  }
  const manifestIds = new Set(byToken.slice(1));
  const notInDataset = [...manifestIds].filter((m) => !datasetIds.has(m));
  const notInManifest = [...datasetIds].filter((m) => !manifestIds.has(m));
  if (notInDataset.length || notInManifest.length) {
    throw new Error(
      `Manifest/dataset mismatch: ${notInDataset.length} manifest-only, ` +
        `${notInManifest.length} dataset-only.`,
    );
  }

  const lines = ["token_id,mreid_id"];
  for (let i = 1; i <= total; i++) lines.push(`${i},${byToken[i]}`);
  const csv = lines.join("\n") + "\n";
  fs.writeFileSync(OUT, csv);

  // Report the permutations so the count is auditable.
  let permuted = 0;
  const samples = [];
  for (let i = 1; i <= total; i++) {
    const suffix = Number(byToken[i].replace("MREID_", ""));
    if (suffix !== i) {
      permuted += 1;
      if (samples.length < 8) samples.push(`#${i}=${byToken[i]}`);
    }
  }
  const sha = require("crypto")
    .createHash("sha256")
    .update(csv)
    .digest("hex");

  console.log(`Wrote ${OUT}`);
  console.log(`  rows            : ${total}`);
  console.log(`  dataset rows    : ${datasetIds.size} (exact match)`);
  console.log(`  order           : canonical token_id 1..${total}`);
  console.log(`  permuted ids    : ${permuted}`);
  console.log(`  first 8 perms   : ${samples.join(" ")}`);
  console.log(`  sha256(manifest): ${sha}`);
  console.log(`  size            : ${Buffer.byteLength(csv)} bytes`);
}

main();
