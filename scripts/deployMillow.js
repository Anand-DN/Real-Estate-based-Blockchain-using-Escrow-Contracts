// Deploys the MILLOW MREID blockchain layer (PropertyNFT + PropertyRegistry +
// MillowEscrow) and writes the addresses into src/config.json under the active
// chain id.
//
// Run against a local Hardhat node:
//     npx hardhat run scripts/deployMillow.js --network localhost
//
// The existing RealEstate/Escrow demo deployment (scripts/deploy.js) is left
// untouched by this script.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  // Canonical demo roles (Hardhat account order):
  //   account[0] 0xf39F… = BUYER
  //   account[1] 0x7099… = SELLER
  //   account[2] 0x3C44… = INSPECTOR
  //   account[3] 0x90F7… = LENDER
  //   account[10] 0xBcd4… = chain MINTER (scripted provisioning; no app role)
  const [deployer, seller, inspector, lender] = await ethers.getSigners();
  console.log("Deploying MILLOW blockchain layer from", deployer.address);

  const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
  const propertyNFT = await PropertyNFT.deploy();
  await propertyNFT.deployed();

  const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
  const propertyRegistry = await PropertyRegistry.deploy(propertyNFT.address);
  await propertyRegistry.deployed();

  const MillowEscrow = await ethers.getContractFactory("MillowEscrow");
  const millowEscrow = await MillowEscrow.deploy(
    propertyNFT.address,
    propertyRegistry.address,
  );
  await millowEscrow.deployed();

  // Allow the escrow to keep the Phase 7 registry "on offer" flag in sync when
  // a sale settles or is cancelled.
  const LISTING_MANAGER_ROLE = await propertyRegistry.LISTING_MANAGER_ROLE();
  await propertyRegistry.grantRole(LISTING_MANAGER_ROLE, millowEscrow.address);

  // Demo role assignments (only meaningful for properties listed through
  // listForSaleWithRequirements()).  Payment Model A sales created through
  // listForSale() never consult these roles.
  const INSPECTOR_ROLE = await millowEscrow.INSPECTOR_ROLE();
  const LENDER_ROLE = await millowEscrow.LENDER_ROLE();
  await millowEscrow.grantRole(INSPECTOR_ROLE, inspector.address);
  await millowEscrow.grantRole(LENDER_ROLE, lender.address);

  // The simplified MILLOW workflow has no registrar, but the chain still keeps
  // MINTER_ROLE on the fixed minting account (the former Registrar,
  // 0xBcd4…) so scripted provisioning can mint tokens outside the application.
  // The buyer (deployer) stays DEFAULT_ADMIN_ROLE holder and gives up minting;
  // this keeps scripts and the live workflow tests deterministic on a brand-new
  // node (deploying fresh hands MINTER_ROLE to the deployer by default).
  const MINTER_ROLE = await propertyNFT.MINTER_ROLE();
  const MINTER = "0xBcd4042DE499D14e55001CcbB24a551F3b954096";
  if (!(await propertyNFT.hasRole(MINTER_ROLE, MINTER))) {
    await propertyNFT.grantRole(MINTER_ROLE, MINTER);
  }
  if (await propertyNFT.hasRole(MINTER_ROLE, deployer.address)) {
    await propertyNFT.revokeRole(MINTER_ROLE, deployer.address);
  }

  const network = await ethers.provider.getNetwork();
  const chainId = String(network.chainId);

  const configPath = path.join(__dirname, "..", "src", "config.json");
  const config = JSON.parse(fs.readFileSync(configPath, "utf8"));
  config[chainId] = config[chainId] || {};
  config[chainId].propertyNft = { address: propertyNFT.address };
  config[chainId].propertyRegistry = { address: propertyRegistry.address };
  config[chainId].millowEscrow = { address: millowEscrow.address };
  // Demonstration-only INR -> test-ETH conversion used by the frontend and
  // seed scripts to compute test amounts.  The contract takes plain wei and
  // never sees this rate.
  config.demoRate = { inrPerEth: 100000 };
  // Demo account map for the role-based workflow.  The UI derives the active
  // account's role from these canonical demo identities via config.demoRoles,
  // so a demo account always stays mapped to exactly one workspace role.  The
  // simplified workflow has no registrar: the seller owns tokens and lists.
  config.demoRoles = {
    buyer: deployer.address,
    seller: seller.address,
    inspector: inspector.address,
    lender: lender.address,
  };
  fs.writeFileSync(configPath, JSON.stringify(config, null, 4));

  console.log(`Deployed PropertyNFT at: ${propertyNFT.address}`);
  console.log(`Deployed PropertyRegistry at: ${propertyRegistry.address}`);
  console.log(`Deployed MillowEscrow at: ${millowEscrow.address}`);
  console.log(`Granted INSPECTOR_ROLE to ${inspector.address}`);
  console.log(`Granted LENDER_ROLE to ${lender.address}`);
  console.log(`Wrote chain ${chainId} addresses to ${configPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});