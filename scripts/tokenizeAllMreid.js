// Tokenizes every MREID in the processed dataset (29,135 properties) as a
// PropertyNFT on the local Hardhat network, delivering each token to the
// canonical demo SELLER (chain MINTER -> seller) WITHOUT auto-listing it.
//
// Run against a running local node, after deployMillow.js + mintMreid.js:
//     npx hardhat run scripts/tokenizeAllMreid.js --network localhost
//
// Quick smoke run limited to the first N dataset rows (skips the 7 already
// minted by mintMreid.js where present):
//     npx hardhat run scripts/tokenizeAllMreid.js --network localhost --limit 20
//
// Re-running is safe and converges: already-minted properties are skipped,
// tokens that exist but are not yet with the seller are delivered, and a
// partial run simply resumes where it left off.
//
// The token contract mints to msg.sender, so each new property costs two
// writes (mint by MINTER + transfer to SELLER).  Writes are pipelined with
// explicit nonces and a small concurrency pool so the full 29k seed stays a
// handful of minutes instead of an hour.  Each batch runs a mint pool first,
// resolves the fresh token ids, then runs a transfer pool.
//
// Hardhat's automine rejects out-of-order nonces ("Nonce too high — cannot
// queue when automining"), so the parallel first pass can leave holes in the
// MINTER account's nonce sequence.  runPool() therefore follows up with a
// deterministic repair pass that fills the lowest nonce hole with the next
// pending job, one send at a time (mint/transfer jobs are independent, so
// order does not matter).  Every retry re-verifies on-chain state first so a
// transaction that already landed is never duplicated.
const fs = require("fs");
const path = require("path");

const BATCH = Number(process.env.MILLOW_SEED_BATCH) > 0 ? Number(process.env.MILLOW_SEED_BATCH) : 500; // dataset rows scanned per progress step
const WRITE_CONCURRENCY =
  Number(process.env.MILLOW_SEED_CONCURRENCY) > 0
    ? Number(process.env.MILLOW_SEED_CONCURRENCY)
    : 8; // parallel mint/transfer pipeline lanes

const MINTER = "0xBcd4042DE499D14e55001CcbB24a551F3b954096";

// The mreid_id column is the first field of every row.  Splitting on the
// first comma avoids any quoted commas later in the row.
function readMreids(limit) {
  const raw = fs.readFileSync(
    path.join(__dirname, "..", "data", "processed", "MREID_property.csv"),
    "utf8",
  );
  const ids = [];
  const lines = raw.split(/\r?\n/);
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const comma = line.indexOf(",");
    const mreid = comma === -1 ? line : line.slice(0, comma);
    if (!/^MREID_\d+$/.test(mreid)) continue;
    ids.push(mreid);
    if (limit > 0 && ids.length >= limit) break;
  }
  return ids;
}

async function main() {
  const args = process.argv.slice(2);
  let limit = Number(process.env.MILLOW_SEED_LIMIT) || 0;
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith("--limit=")) {
      limit = Number(args[i].split("=")[1]);
      break;
    }
    if (args[i] === "--limit" && args[i + 1] && /^\d+$/.test(args[i + 1])) {
      limit = Number(args[i + 1]);
      break;
    }
  }
  if (!Number.isFinite(limit) || limit < 0) limit = 0;

  const ids = readMreids(limit);
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "src", "config.json"), "utf8"),
  );
  const network = await ethers.provider.getNetwork();
  const chainId = String(network.chainId);
  const chainConfig = config[chainId];
  if (!chainConfig || !chainConfig.propertyNft) {
    throw new Error(
      `No MILLOW contract addresses for chain ${chainId}. Run scripts/deployMillow.js first.`,
    );
  }
  const seller = (config.demoRoles && config.demoRoles.seller) || null;
  if (!seller) {
    throw new Error(
      "config.demoRoles.seller is missing. Run scripts/deployMillow.js first.",
    );
  }

  const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
  const propertyNFT = await PropertyNFT.attach(chainConfig.propertyNft.address);
  const minterSigner = await ethers.getSigner(MINTER);
  const sellerLower = seller.toLowerCase();

  console.log(`Tokenizing ${ids.length} dataset properties (limit=${limit || "all"})`);
  console.log(`MINTER: ${MINTER}`);
  console.log(`SELLER: ${seller}`);
  console.log(`BATCH=${BATCH} WRITE_CONCURRENCY=${WRITE_CONCURRENCY}`);

  // Sanity check: the fixed minter must actually hold MINTER_ROLE (bootstrap
  // happens in deployMillow.js).  Fail loudly before sending any transaction.
  const MINTER_ROLE = await propertyNFT.MINTER_ROLE();
  if (!(await propertyNFT.hasRole(MINTER_ROLE, MINTER))) {
    throw new Error(
      `MINTER_ROLE is not on ${MINTER}. Re-run deployMillow.js against this chain first.`,
    );
  }

  const counters = {
    appearedExisting: 0, // were already tokenized when scanned
    alreadyTokenized: 0, // tokenized AND owned by the seller when scanned
    minted: 0, // minted by this run
    delivered: 0, // owned by the seller after the pass
    transferJobs: 0, // transfer transactions sent
    repaired: 0, // jobs settled by the nonce repair pass
    errors: 0, // scan/verification/repair failures
  };

  const tokenByProperty = async (mreid) =>
    (await propertyNFT.tokenByProperty(mreid)).toNumber();
  const ownerOf = async (tokenId) => (await propertyNFT.ownerOf(tokenId)).toLowerCase();

  // Sends a list of pre-signed job descriptors with explicit nonces on the
  // MINTER account.  Lanes send concurrently; under automine any job that does
  // not land (out-of-order nonce) is retried in a strict repair pass that
  // fills the lowest nonce hole, one send at a time, keeping the account's
  // nonce sequence contiguous.  Mint / transfer jobs are independent, so a
  // different send order is always safe.
  const runPool = async (jobs, label) => {
    if (!jobs.length) return;
    let cursor = 0;
    const baseNonce = await minterSigner.getTransactionCount();
    const encoded = jobs.map((job, index) => ({
      ...job,
      nonce: baseNonce + index,
      data:
        job.kind === "mint"
          ? propertyNFT.interface.encodeFunctionData("mintProperty", [
              job.mreid,
              `http://localhost:3000/metadata/millow/${job.mreid}.json`,
            ])
          : propertyNFT.interface.encodeFunctionData("transferFrom", [
              MINTER,
              seller,
              job.tokenId,
            ]),
    }));
    const sendAndWait = (job) =>
      minterSigner
        .sendTransaction({
          to: chainConfig.propertyNft.address,
          data: job.data,
          nonce: job.nonce,
        })
        .then((tx) => tx.wait());
    const applied = (job) =>
      job.kind === "mint"
        ? tokenByProperty(job.mreid)
            .then((id) => id > 0)
            .catch(() => false)
        : ownerOf(job.tokenId)
            .then((owner) => owner === sellerLower)
            .catch(() => false);

    const failed = [];
    const worker = async () => {
      while (cursor < encoded.length) {
        const job = encoded[cursor++];
        try {
          await sendAndWait(job);
          if (job.kind === "mint") counters.minted += 1;
          else counters.transferJobs += 1;
        } catch {
          failed.push(job);
        }
      }
    };
    await Promise.all(Array.from({ length: WRITE_CONCURRENCY }, worker));

    // Repair pass: settle each failed job at the current lowest nonce hole.
    let repaired = 0;
    let round = 0;
    while (failed.length && round < 60) {
      const job = failed.shift();
      if (await applied(job)) {
        if (job.kind === "mint") counters.minted += 1;
        repaired += 1; // transfers are counted as delivered by the caller
        continue;
      }
      const nextNonce = await minterSigner.getTransactionCount();
      try {
        await sendAndWait({ ...job, nonce: nextNonce });
        if (job.kind === "mint") counters.minted += 1;
        else counters.transferJobs += 1;
        repaired += 1;
      } catch {
        round += 1;
        failed.push(job);
      }
    }
    if (failed.length) {
      throw new Error(
        `[${label}] ${failed.length} job(s) could not be sent after repeated retries`,
      );
    }
    counters.repaired += repaired;
    if (repaired) {
      console.log(`[${label}] repair pass settled ${repaired} delayed job(s)`);
    }
  };

  // One scan-and-write pass over a batch:
  //   1. parallel scan of tokenByProperty + owner (read-only, fast),
  //   2. mint pool for properties without a token,
  //   3. resolve the fresh token ids,
  //   4. transfer pool for every token not yet at the seller.
  const processBatch = async (batch) => {
    const scans = await Promise.all(
      batch.map(async (mreid) => {
        let tokenId = 0;
        try {
          tokenId = await tokenByProperty(mreid);
        } catch {
          counters.errors += 1;
        }
        let ownedBySeller = false;
        if (tokenId) {
          try {
            ownedBySeller = (await ownerOf(tokenId)) === sellerLower;
          } catch {
            counters.errors += 1;
          }
        }
        return { mreid, tokenId, ownedBySeller };
      }),
    );

    const toMint = scans.filter((s) => !s.tokenId);
    const alreadyAtSeller = scans.filter((s) => s.tokenId && s.ownedBySeller);
    const needsTransfer = scans.filter((s) => s.tokenId && !s.ownedBySeller);
    counters.appearedExisting += scans.length - toMint.length;
    counters.alreadyTokenized += alreadyAtSeller.length;

    if (toMint.length) {
      await runPool(toMint.map((s) => ({ kind: "mint", mreid: s.mreid })), "mint");
      // Resolve token ids for the freshly minted properties.
      await Promise.all(
        toMint.map(async (scan) => {
          scan.tokenId = await tokenByProperty(scan.mreid).catch(() => 0);
        }),
      );
    }

    const transferJobs = needsTransfer
      .concat(toMint.filter((s) => s.tokenId))
      .map((s) => ({ kind: "transfer", tokenId: s.tokenId, mreid: s.mreid }));
    if (transferJobs.length) {
      await runPool(transferJobs, "transfer");
      for (const job of transferJobs) {
        const owner = await ownerOf(job.tokenId).catch(() => null);
        if (owner === sellerLower) counters.delivered += 1;
        else counters.errors += 1;
      }
    }

    counters.delivered += alreadyAtSeller.length;
  };

  const startedAt = Date.now();
  let done = 0;
  for (let start = 0; start < ids.length; start += BATCH) {
    const batch = ids.slice(start, start + BATCH);
    await processBatch(batch);
    done += batch.length;
    const pct = ((done / ids.length) * 100).toFixed(1);
    const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
    console.log(
      `[${done}/${ids.length}] ${pct}% in ${elapsed}s — ` +
        `minted ${counters.minted}, was existing ${counters.appearedExisting}, ` +
        `at seller ${counters.delivered}, errors ${counters.errors}`,
    );
  }

  const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
  const first10 = await Promise.all(
    ids.slice(0, 10).map(async (mreid) => {
      const tokenId = await tokenByProperty(mreid).catch(() => 0);
      return `${mreid}#${tokenId}`;
    }),
  );
  let seedMappings = [];
  try {
    const seedIds = JSON.parse(
      fs.readFileSync(path.join(__dirname, "millowSeed.json"), "utf8"),
    );
    seedMappings = await Promise.all(
      seedIds.slice(0, 7).map(async (mreid) => {
        const tokenId = await tokenByProperty(mreid).catch(() => 0);
        return `${mreid}#${tokenId}`;
      }),
    );
  } catch {
    seedMappings = [];
  }

  console.log("--------------------------------------------------------");
  console.log("Tokenization complete in", elapsed + "s");
  const successfullyTokenized = Math.max(0, counters.delivered - counters.alreadyTokenized);
  const remaining = Math.max(0, ids.length - counters.delivered);
  console.log(` successfully tokenized: ${successfullyTokenized}`);
  console.log(` already tokenized: ${counters.alreadyTokenized}`);
  console.log(` failed: ${counters.errors}`);
  console.log(` remaining: ${remaining}`);
  console.log(`Dataset properties handled: ${ids.length}`);
  console.log(` Already tokenized when scanned (any owner): ${counters.appearedExisting}`);
  console.log(` Minted by this run: ${counters.minted}`);
  console.log(` Transfer transactions sent: ${counters.transferJobs}`);
  console.log(` Owned by the seller now: ${counters.delivered}`);
  console.log(` Repair-pass jobs settled: ${counters.repaired}`);
  console.log(` Errors: ${counters.errors}`);
  console.log("First dataset mappings (mreid#token):", first10.join(", "));
  if (seedMappings.length) {
    console.log("Seed subset mappings (mreid#token):", seedMappings.join(", "));
  }
  console.log("No listings were created by this script.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});