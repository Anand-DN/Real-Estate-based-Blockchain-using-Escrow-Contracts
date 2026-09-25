// Phase 8 live demo: complete a full escrow sale on the running Hardhat node.
// Targets token 1 (MREID_0000001) already listed by configureSale.js.
// Uses the canonical demo roles from src/config.json (seller = 0x7099…,
// buyer = 0xf39F…).
// Run: npx hardhat run scripts/demoFullSale.js --network localhost
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const ETH = (n) => ethers.BigNumber.from(n).mul(ethers.utils.parseEther("1"));

async function main() {
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "src", "config.json"), "utf8"),
  );
  const chainId = String((await ethers.provider.getNetwork()).chainId);
  const cc = config[chainId];
  if (!cc || !cc.millowEscrow) throw new Error("deployMillow first");

  const roles = config.demoRoles || {};
  const seller = await ethers.getSigner(roles.seller);
  const buyer = await ethers.getSigner(roles.buyer);

  const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
  const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
  const MillowEscrow = await ethers.getContractFactory("MillowEscrow");
  const nft = await PropertyNFT.attach(cc.propertyNft.address);
  const registry = await PropertyRegistry.attach(cc.propertyRegistry.address);
  const escrow = await MillowEscrow.attach(cc.millowEscrow.address);

  const tokenId = 1;
  const priceWei = ETH("300");
  const earnestWei = ETH("30");
  const owedWei = ETH("270");

  const log = (m) => console.log(m);
const saleInfo = () => escrow.sales(tokenId);
  const statusName = async () =>
    ["None", "Listed", "UnderContract", "Approved", "Finalized", "Cancelled"][
      Number((await saleInfo()).status)
    ];

  const sale = await saleInfo();
  if (!sale.seller || sale.seller === ethers.constants.AddressZero) {
    throw new Error("Token 1 is not listed; run configureSale.js first");
  }

  log(`Token ${tokenId} start state: ${await statusName()}`);
  log(`  seller=${sale.seller} buyer=${sale.buyer}`);
  log(`  priceWei=${sale.priceWei} earnestWei=${sale.earnestWei}`);

let current = Number((await saleInfo()).status);

  if (current === 1) {
    const commitTx = await escrow
      .connect(buyer)
      .commitAndDeposit(tokenId, { value: earnestWei });
    await commitTx.wait();
    log(
      `Commit & deposit ${ethers.utils.formatEther(earnestWei)} ETH (tx ${commitTx.hash}) -> ${await statusName()}`,
    );
    current = 2;
  }

  if (current === 2) {
    const { buyerFundedWei, lenderFundedWei } = await saleInfo();
    const funded = buyerFundedWei.add(lenderFundedWei);
    if (funded.lt(priceWei)) {
      const fundTx = await escrow
        .connect(buyer)
        .fundBalance(tokenId, { value: priceWei.sub(funded) });
      await fundTx.wait();
      log(
        `Fund balance ${ethers.utils.formatEther(priceWei.sub(funded))} ETH (tx ${fundTx.hash}) -> funded ${ethers.utils.formatEther(priceWei)} ETH`,
      );
    } else {
      log(`Balance already fully funded (${ethers.utils.formatEther(funded)} ETH).`);
    }

    const s = await saleInfo();
    if (!s.buyerApproved) {
      const approveBuyer = await escrow.connect(buyer).approveSale(tokenId);
      await approveBuyer.wait();
      log(`Buyer approveSale (tx ${approveBuyer.hash})`);
    }
    const s2 = await saleInfo();
    if (!s2.sellerApproved) {
      const approveSeller = await escrow.connect(seller).approveSale(tokenId);
      await approveSeller.wait();
      log(`Seller approveSale (tx ${approveSeller.hash}) -> ${await statusName()}`);
    }
    current = 3;
  }

  if (current === 3) {
    const finalTx = await escrow.connect(buyer).finalizeSale(tokenId);
    await finalTx.wait();
    log(`Finalize sale (tx ${finalTx.hash}) -> ${await statusName()}`);
  } else if (current === 4) {
    log(`Sale already finalized; skipping.`);
  }

  const after = await saleInfo();
  const nftOwner = await nft.ownerOf(tokenId);
  const onOffer = await registry.isOnOffer(tokenId);
  const sellerBalance = await ethers.provider.getBalance(seller.address);
const escrowBalance = await ethers.provider.getBalance(escrow.address);

  log("");
  log("=== VERIFICATION ===");
  log(`status: ${await statusName()} (expected Finalized)`);
  log(`NFT owner of token ${tokenId}: ${nftOwner}`);
  log(`NFT belongs to buyer: ${nftOwner.toLowerCase() === buyer.address.toLowerCase()}`);
  log(`registry.isOnOffer: ${onOffer} (expected false)`);
  log(`seller test-ETH payout recorded: ${ethers.utils.formatEther(after.priceWei)} ETH`);
  log(`escrow balance now: ${ethers.utils.formatEther(escrowBalance)} ETH (expected 0)`);

  const ok =
    Number(after.status) === 4 &&
    nftOwner.toLowerCase() === buyer.address.toLowerCase() &&
    !onOffer &&
    escrowBalance.isZero();
  log(ok ? "RESULT: PASS - full sale flow verified on-chain" : "RESULT: FAIL");

  if (!ok) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
