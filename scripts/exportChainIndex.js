// Exports an offline chain-index snapshot for every tokenized property to
// data/processed/chain_index.json.  This snapshot is the Phase 2 / Phase 9
// backbone: the frontend and the backend dashboard read catalogue-wide
// tokenization / listing / sale state from this file instead of scanning the
// contract with tens of thousands of RPC calls.
//
// Run against a running local node, after deployMillow.js + tokenizeAllMreid.js:
//     npx hardhat run scripts/exportChainIndex.js --network localhost
//
// On a persistent Anvil chain (see docs/REPRODUCIBILITY.md) the same command
// regenerates the snapshot from an IMPORTED state artifact, which is how a
// second machine populates it without minting anything:
//     npm run chain:index
//
// READ-ONLY BY CONSTRUCTION.  Every chain access below is a view call —
// totalSupply(), propertyOf(), ownerOf(), isOnOffer() and sales().  This script
// sends no transaction, deploys nothing, mints nothing and modifies no chain
// state, so it is safe to run against a populated production-equivalent chain.
//
// The snapshot is a point-in-time read: it must be refreshed after seeding /
// listing / sale activity.  The backend serves the last exported snapshot and
// the snapshot records when it was written so consumers know its age.
// chain_index.json is a GENERATED artifact and is deliberately gitignored; it
// is never committed.
//
// MUST be run through Hardhat, which injects the global `ethers`:
//     npm run chain:index
//     npx hardhat run scripts/exportChainIndex.js --network localhost
// Running it with bare `node` leaves `ethers` undefined, so fail with an
// instruction instead of an opaque ReferenceError further down.
//
// Per property the snapshot stores:
//   tokenized    : has a PropertyNFT token id
//   token_id     : PropertyNFT token id (0 when not tokenized)
//   owner        : current PropertyNFT owner (checksummed, or null)
//   listed       : PropertyRegistry on-offer state (PropertyRegistry.isOnOffer)
//   active_sale  : MillowEscrow sale is not None and not Finalized/Cancelled
//   finalized    : MillowEscrow sale reached Finalized
//   sale_status  : MillowEscrow Status enum name or null
const fs = require("fs");
const path = require("path");

const CONCURRENCY = 16;
const BATCH = 200;

// `ethers` is a global injected by the Hardhat runtime, not a require().  Catch
// the bare-`node` case up front so the failure explains itself.
if (typeof ethers === "undefined") {
  console.error(
    "exportChainIndex.js must be run through Hardhat, which provides `ethers`.\n" +
      "  Correct:   npm run chain:index\n" +
      "  or:        npx hardhat run scripts/exportChainIndex.js --network localhost\n" +
      "  Wrong:     node scripts/exportChainIndex.js\n" +
      "  (a running local node is required on 127.0.0.1:8545)",
  );
  process.exit(1);
}

async function main() {
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "src", "config.json"), "utf8"),
  );
  const provider = ethers.provider;
  const network = await provider.getNetwork();
  const chainId = String(network.chainId);
  const chainConfig = config[chainId];
  if (!chainConfig || !chainConfig.propertyNft) {
    throw new Error(
      `No MILLOW contract addresses for chain ${chainId}. Run scripts/deployMillow.js first.`,
    );
  }

  const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
  const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
  const MillowEscrow = await ethers.getContractFactory("MillowEscrow");

  const propertyNFT = PropertyNFT.attach(chainConfig.propertyNft.address);
  const propertyRegistry = PropertyRegistry.attach(
    chainConfig.propertyRegistry.address,
  );
  const millowEscrow = MillowEscrow.attach(chainConfig.millowEscrow.address);

  const total = await propertyNFT.totalSupply();
  const tokenIds = [];
  for (let id = 1; id <= total; id++) tokenIds.push(id);

  console.log(
    `Exporting chain index on chain ${chainId}: ${total} tokens`,
  );

  const entries = {};
  const counters = {
    tokenized: 0,
    listed: 0,
    active_sale: 0,
    finalized: 0,
    errors: 0,
  };

  const propertyOf = async (id) => {
    try {
      return await propertyNFT.propertyOf(id);
    } catch {
      return null;
    }
  };
  const ownerOf = async (id) => {
    try {
      return await propertyNFT.ownerOf(id);
    } catch {
      return null;
    }
  };
  const registryIsOnOffer = async (id) => {
    try {
      return await propertyRegistry.isOnOffer(id);
    } catch {
      return false;
    }
  };
  const saleStatus = async (id) => {
    try {
      const sale = await millowEscrow.sales(id);
      const status = Number(sale.status);
      const names = [
        "None",
        "Listed",
        "UnderContract",
        "Approved",
        "Finalized",
        "Cancelled",
      ];
      return {
        name: names[status] || "None",
        raw: status,
      };
    } catch {
      return { name: null, raw: -1 };
    }
  };

  let done = 0;
  for (let start = 0; start < tokenIds.length; start += BATCH) {
    const chunk = tokenIds.slice(start, start + BATCH);
    const rows = await Promise.all(
      chunk.map(async (id) => {
        const [mreid, owner, listed, sale] = await Promise.all([
          propertyOf(id),
          ownerOf(id),
          registryIsOnOffer(id),
          saleStatus(id),
        ]);
        return { id, mreid, owner, listed, sale };
      }),
    );
    for (const row of rows) {
      if (!row.mreid) {
        counters.errors += 1;
        continue;
      }
      counters.tokenized += 1;
      const active =
        row.sale.raw >= 1 && row.sale.raw <= 3; // Listed..Approved
      const finalized = row.sale.raw === 4;
      if (row.listed) counters.listed += 1;
      if (active) counters.active_sale += 1;
      if (finalized) counters.finalized += 1;
      entries[row.mreid] = {
        tokenized: true,
        token_id: row.id,
        owner: row.owner || null,
        listed: Boolean(row.listed),
        active_sale: active,
        finalized,
        sale_status: row.sale.name,
      };
    }
    done += chunk.length;
    console.log(`  [${done}/${tokenIds.length}]`);
  }

  const snapshot = {
    chain_id: chainId,
    exported_at: new Date().toISOString(),
    contracts: {
      property_nft: chainConfig.propertyNft.address,
      property_registry: chainConfig.propertyRegistry.address,
      millow_escrow: chainConfig.millowEscrow.address,
    },
    counts: {
      total_properties: total.toNumber(),
      ...counters,
    },
    properties: entries,
  };

  const outPath = path.join(
    __dirname,
    "..",
    "data",
    "processed",
    "chain_index.json",
  );
  fs.writeFileSync(outPath, JSON.stringify(snapshot, null, 2));
  console.log("--------------------------------------------------------");
  console.log(`Chain index exported to ${outPath}`);
  console.log(` tokenized: ${counters.tokenized}`);
  console.log(` listed: ${counters.listed}`);
  console.log(` active sales: ${counters.active_sale}`);
  console.log(` finalized: ${counters.finalized}`);
  console.log(` errors: ${counters.errors}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});