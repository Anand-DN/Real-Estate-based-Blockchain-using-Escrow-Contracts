// Mints the representative MREID seed subset as PropertyNFT tokens on the
// local Hardhat network, delivers each token to the canonical demo SELLER
// (chain MINTER -> seller), and puts the first two tokens on offer in the
// PropertyRegistry.
//
// Run against a running local node (after deployMillow.js):
//     npx hardhat run scripts/mintMreid.js --network localhost
//
// On the deployed chain the MINTER_ROLE lives with the fixed minting account
// (the former demo Registrar); on a brand-new chain the deployer holds it.
// Re-running is safe: already-minted properties are skipped, and listings are
// only created once.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const ids = JSON.parse(
    fs.readFileSync(path.join(__dirname, "millowSeed.json"), "utf8"),
  );
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "src", "config.json"), "utf8"),
  );
  const network = await ethers.provider.getNetwork();
  const chainId = String(network.chainId);
  const chainConfig = config[chainId];
  if (!chainConfig || !chainConfig.propertyNft || !chainConfig.propertyRegistry) {
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
  const sellerSigner = await ethers.getSigner(seller);

  // The deployed chain hands MINTER_ROLE to this fixed minting account.
  const MINTER = "0xBcd4042DE499D14e55001CcbB24a551F3b954096";
  const minterSigner = await ethers.getSigner(MINTER);

  const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
  const propertyNFT = await PropertyNFT.attach(chainConfig.propertyNft.address);

  const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
  const propertyRegistry = await PropertyRegistry.attach(
    chainConfig.propertyRegistry.address,
  );

  const tokenIds = [];
  for (const mreidId of ids) {
    const existing = await propertyNFT.tokenByProperty(mreidId);
    if (!existing.isZero()) {
      console.log(`${mreidId} already minted as token ${existing}`);
      tokenIds.push(existing);
      continue;
    }

    const tokenURI = `http://localhost:3000/metadata/millow/${mreidId}.json`;
    const tx = await propertyNFT.connect(minterSigner).mintProperty(mreidId, tokenURI);
    const receipt = await tx.wait();
    const tokenId = await propertyNFT.tokenByProperty(mreidId);

    console.log(
      `Minted ${mreidId} -> token ${tokenId} (tx ${receipt.transactionHash})`,
    );
    tokenIds.push(tokenId);
  }

  // Chain MINTER -> Seller: deliver every seed token to the canonical demo
  // SELLER so the on-chain seller owns the NFT it lists.
  for (const tokenId of tokenIds) {
    const currentOwner = await propertyNFT.ownerOf(tokenId);
    if (currentOwner.toLowerCase() === seller.toLowerCase()) {
      console.log(`Token ${tokenId} already owned by seller ${seller}`);
      continue;
    }
    const transferTx = await propertyNFT
      .connect(minterSigner)
      .transferFrom(currentOwner, seller, tokenId);
    const transferReceipt = await transferTx.wait();
    console.log(
      `Delivered token ${tokenId} to seller ${seller} (tx ${transferReceipt.transactionHash})`,
    );
  }

  for (let i = 0; i < 2 && i < tokenIds.length; i++) {
    const tokenId = tokenIds[i];
    if (!(await propertyRegistry.isOnOffer(tokenId))) {
      const tx = await propertyRegistry
        .connect(sellerSigner)
        .list(tokenId);
      const receipt = await tx.wait();
      console.log(
        `Listed token ${tokenId} (${ids[i]}) on offer (tx ${receipt.transactionHash})`,
      );
    } else {
      console.log(`Token ${tokenId} (${ids[i]}) is already on offer`);
    }
  }

  console.log("Seed complete.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});