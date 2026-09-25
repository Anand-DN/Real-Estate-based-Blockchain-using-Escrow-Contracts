// Configures a role-based escrow sale on the local Hardhat network through
// listForSaleWithRequirements().  The selected Inspector / Lender gates only
// apply to this listing; plain listForSale() listings keep Payment Model A.
//
// Run against a running local node (after deployMillow.js + mintMreid.js):
//     npx hardhat run scripts/configureRoleSale.js --network localhost
//
// Defaults: token 2 = MREID_0000002, ₹30,000,000 asking price, 10% earnest,
// inspection + lender financing both required.  Override with TOKEN_ID,
// IN_RUPEES, EARNEST_IN_RUPEES, REQUIRE_INSPECTION / REQUIRE_LENDER env vars.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const IN_RUPEES_DEFAULT = 30000000;
const EARNEST_IN_RUPEES_DEFAULT = 3000000;

async function main() {
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "src", "config.json"), "utf8"),
  );
  const network = await ethers.provider.getNetwork();
  const chainId = String(network.chainId);
  const chainConfig = config[chainId];
  const demoRate = config.demoRate && config.demoRate.inrPerEth;

  if (
    !chainConfig ||
    !chainConfig.propertyNft ||
    !chainConfig.propertyRegistry ||
    !chainConfig.millowEscrow
  ) {
    throw new Error(
      `No MILLOW contracts for chain ${chainId}. Run scripts/deployMillow.js first.`,
    );
  }
  if (!demoRate) {
    throw new Error("config.demoRate.inrPerEth is missing. Re-run deployMillow.");
  }

  const seller = (config.demoRoles && config.demoRoles.seller) || null;
  if (!seller) {
    throw new Error(
      "config.demoRoles.seller is missing. Run scripts/deployMillow.js first.",
    );
  }
  const sellerSigner = await ethers.getSigner(seller);

  const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
  const propertyNFT = await PropertyNFT.attach(chainConfig.propertyNft.address);

  const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
  const propertyRegistry = await PropertyRegistry.attach(
    chainConfig.propertyRegistry.address,
  );

  const MillowEscrow = await ethers.getContractFactory("MillowEscrow");
  const escrow = await MillowEscrow.attach(chainConfig.millowEscrow.address);

  const tokenId = process.env.TOKEN_ID ? Number(process.env.TOKEN_ID) : 2;
  const inrPrice = Number(process.env.IN_RUPEES || IN_RUPEES_DEFAULT);
  const inrEarnest = Number(
    process.env.EARNEST_IN_RUPEES || EARNEST_IN_RUPEES_DEFAULT,
  );
  const requireInspection = String(
    process.env.REQUIRE_INSPECTION || "1",
  ) === "1";
  const requireLender = String(process.env.REQUIRE_LENDER || "1") === "1";

  if (!requireInspection && !requireLender) {
    throw new Error("At least one requirement must be enabled for a role sale.");
  }

  const priceWei = ethers.BigNumber.from(inrPrice)
    .mul(ethers.utils.parseEther("1"))
    .div(ethers.BigNumber.from(demoRate));
  const earnestWei = ethers.BigNumber.from(inrEarnest)
    .mul(ethers.utils.parseEther("1"))
    .div(ethers.BigNumber.from(demoRate));

  const mreidId = await propertyNFT.propertyOf(tokenId);
  const ownerOf = await propertyNFT.ownerOf(tokenId);
  if (ownerOf.toLowerCase() !== seller.toLowerCase()) {
    throw new Error(
      `Token ${tokenId} is owned by ${ownerOf}, not the demo seller ${seller}. Run scripts/mintMreid.js to deliver seed tokens to the seller.`,
    );
  }

  console.log(
    `Configuring role sale for token ${tokenId} (${mreidId}) by seller ${seller}`,
  );
  console.log(
    `Requirements: inspection=${requireInspection}, lender=${requireLender}`,
  );
  console.log(`INR asking price: ₹${inrPrice.toLocaleString("en-IN")}`);
  console.log(`Demo rate: ₹${demoRate.toLocaleString("en-IN")} = 1 test ETH`);
  console.log(`ETH price (wei): ${priceWei.toString()}`);
  console.log(`ETH earnest (wei): ${earnestWei.toString()}`);

  if (await escrow.isOnOffer(tokenId)) {
    console.log(`Token ${tokenId} is already in an escrow sale; skipping.`);
    console.log("Done.");
    return;
  }

  const approved = await propertyNFT.getApproved(tokenId);
  if (approved !== escrow.address) {
    const approveTx = await propertyNFT
      .connect(sellerSigner)
      .approve(escrow.address, tokenId);
    await approveTx.wait();
    console.log("Approved escrow as operator for the token.");
  } else {
    console.log("Escrow already approved as operator.");
  }

  const listTx = await escrow
    .connect(sellerSigner)
    .listForSaleWithRequirements(
      tokenId,
      priceWei,
      earnestWei,
      requireInspection,
      requireLender,
    );
  const listReceipt = await listTx.wait();
  console.log(
    `Listed in escrow (tx ${listReceipt.transactionHash}): ${mreidId} with role requirements.`,
  );

  console.log("Done.");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});