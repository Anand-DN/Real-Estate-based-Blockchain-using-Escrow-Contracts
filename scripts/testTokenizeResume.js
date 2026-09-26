// Offline unit tests for the tokenization resume planner and the export
// guard conditions.  These make NO network calls and touch no chain.
//
//   node scripts/testTokenizeResume.js
const assert = require("assert");
const { planTokenization } = require("./tokenizeChain");
const { loadManifest, loadTokenMap } = require("./lib/chain");

// Real canonical manifest, not a synthetic stand-in: the permutation check is
// only meaningful against the actual data.
const manifest = loadManifest();
const TOTAL = manifest.canonical_token_count;
const rows = loadTokenMap(manifest);

// A lookup that reports tokens 1..n minted correctly, matching the manifest.
const correctPrefix = (n) => async (i) => (i < n ? i + 1 : 0);

let pass = 0;
function t(name, fn) {
  return Promise.resolve()
    .then(fn)
    .then(() => {
      pass += 1;
      console.log(`PASS  ${name}`);
    })
    .catch((e) => {
      console.log(`FAIL  ${name}\n      ${e.message}`);
      process.exitCode = 1;
    });
}

(async () => {
  console.log("=== planTokenization (no chain access) ===");

  await t("fresh chain (supply 0) begins at row 0", async () => {
    const p = await planTokenization({
      total: TOTAL,
      supply0: 0,
      lookup: correctPrefix(0),
    });
    assert.strictEqual(p.resumeAt, 0, "must start at row 0");
    assert.strictEqual(p.todo, TOTAL, "must mint all 29135");
    assert.strictEqual(p.complete, false);
  });

  await t("partial chain (supply 600) resumes at row 600", async () => {
    const p = await planTokenization({
      total: TOTAL,
      supply0: 600,
      lookup: correctPrefix(600),
    });
    assert.strictEqual(p.resumeAt, 600, "must resume at 600");
    assert.strictEqual(p.todo, TOTAL - 600);
    assert.strictEqual(p.complete, false);
  });

  await t("complete chain (supply 29135) does no work", async () => {
    const p = await planTokenization({
      total: TOTAL,
      supply0: TOTAL,
      lookup: correctPrefix(TOTAL),
    });
    assert.strictEqual(p.resumeAt, TOTAL);
    assert.strictEqual(p.todo, 0);
    assert.strictEqual(p.complete, true, "must report complete");
  });

  await t("divergent mapping hard-fails", async () => {
    // token 5 holds the wrong id -> prefix is 4, but supply says 600
    const lookup = async (i) => (i === 4 ? 999 : i < 600 ? i + 1 : 0);
    await assert.rejects(
      () => planTokenization({ total: TOTAL, supply0: 600, lookup }),
      /Inconsistent chain/,
    );
  });

  await t("partial divergence (mid-prefix) hard-fails", async () => {
    const lookup = async (i) => (i === 300 ? 0 : i < 600 ? i + 1 : 0);
    await assert.rejects(
      () => planTokenization({ total: TOTAL, supply0: 600, lookup }),
      /Inconsistent chain/,
    );
  });

  await t("supply greater than canonical hard-fails", async () => {
    await assert.rejects(
      () =>
        planTokenization({ total: TOTAL, supply0: TOTAL + 1, lookup: correctPrefix(0) }),
      /more than the canonical/,
    );
  });

  await t("every resume point is exact across the range", async () => {
    for (const n of [1, 2, 99, 100, 101, 1000, 14567, 29134]) {
      const p = await planTokenization({
        total: TOTAL,
        supply0: n,
        lookup: correctPrefix(n),
      });
      assert.strictEqual(p.resumeAt, n, `resumeAt for ${n}`);
      assert.strictEqual(p.todo, TOTAL - n, `todo for ${n}`);
    }
  });

  await t("canonical mapping is unchanged and 1-based", () => {
    assert.strictEqual(rows.length, TOTAL);
    assert.strictEqual(rows[0].tokenId, 1);
    assert.strictEqual(rows[TOTAL - 1].tokenId, TOTAL);
    for (let i = 0; i < rows.length; i++) {
      assert.strictEqual(rows[i].tokenId, i + 1, `row ${i}`);
    }
  });

  await t("permuted MREIDs are not equal to their token id", () => {
    // The canonical manifest intentionally permutes 1,396 ids.  A planner that
    // assumed mreid number == token id would be wrong; confirm the data is.
    let permuted = 0;
    for (const r of rows) {
      if (Number(r.mreid.replace(/\D/g, "")) !== r.tokenId) permuted += 1;
    }
    assert.strictEqual(permuted, 1396, "expected 1396 permuted ids");
  });

  console.log(`\n${pass} passed, ${process.exitCode ? "FAILURES PRESENT" : "all green"}`);
})();
