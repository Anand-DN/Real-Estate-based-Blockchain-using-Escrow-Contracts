// We require the Hardhat Runtime Environment explicitly here. This is optional
// but useful for running the script in a standalone fashion through `node <script>`.
//
// You can also run a script with `npx hardhat run <script>`. If you do that, Hardhat
// will compile your contracts, add the Hardhat Runtime Environment's members to the
// global scope, and execute the script.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const tokens = (n) => {
  return ethers.utils.parseUnits(n.toString(), 'ether')
}

async function main() {
  // Setup accounts
  const [buyer, seller, inspector, lender] = await ethers.getSigners()

  // Deploy Real Estate
  const RealEstate = await ethers.getContractFactory('RealEstate')
  const realEstate = await RealEstate.deploy()
  await realEstate.deployed()

  console.log(`Deployed Real Estate Contract at: ${realEstate.address}`)
  console.log(`Minting 24 properties...\n`)

  for (let i = 0; i < 24; i++) {
    const transaction = await realEstate.connect(seller).mint(`http://localhost:3000/metadata/${i + 1}.json`)
    await transaction.wait()
  }

  // Deploy Escrow
  const Escrow = await ethers.getContractFactory('Escrow')
  const escrow = await Escrow.deploy(
    realEstate.address,
    inspector.address,
    lender.address
  )
  await escrow.deployed()

  console.log(`Deployed Escrow Contract at: ${escrow.address}`)
  console.log(`Listing 24 properties...\n`)

  for (let i = 0; i < 24; i++) {
    // Approve properties...
    let transaction = await realEstate.connect(seller).approve(escrow.address, i + 1)
    await transaction.wait()
  }

  // Listing properties...
  transaction = await escrow.connect(seller).list(1, tokens(20), tokens(10))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(2, tokens(15), tokens(5))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(3, tokens(10), tokens(5))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(4, tokens(12), tokens(4))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(5, tokens(8), tokens(3))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(6, tokens(18), tokens(6))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(7, tokens(22), tokens(8))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(8, tokens(9), tokens(3))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(9, tokens(14), tokens(5))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(10, tokens(11), tokens(4))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(11, tokens(13), tokens(5))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(12, tokens(26), tokens(10))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(13, tokens(19), tokens(7))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(14, tokens(12), tokens(4))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(15, tokens(10), tokens(4))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(16, tokens(24), tokens(9))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(17, tokens(30), tokens(11))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(18, tokens(28), tokens(10))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(19, tokens(35), tokens(13))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(20, tokens(16), tokens(6))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(21, tokens(21), tokens(8))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(22, tokens(25), tokens(9))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(23, tokens(32), tokens(12))
  await transaction.wait()

  transaction = await escrow.connect(seller).list(24, tokens(18), tokens(7))
  await transaction.wait()

  console.log(`Finished.`)

  // Auto-write addresses to the frontend config
  const configPath = path.join(__dirname, '..', 'src', 'config.json')
  const config = {
    "31337": {
      "realEstate": { "address": realEstate.address },
      "escrow": { "address": escrow.address }
    }
  }
  fs.writeFileSync(configPath, JSON.stringify(config, null, 4))
  console.log(`Wrote frontend config to ${configPath}`)
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
