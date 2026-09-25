// Live demo: cancellation and split refund path on the running Hardhat node
// for a listing created through listForSaleWithRequirements().
//
// Run: npx hardhat run scripts/demoCancelSale.js --network localhost
// Defaults to token 6 (MREID_0015936); override with TOKEN_ID.  Uses the
// canonical demo roles from src/config.json (seller = 0x7099…, buyer = 0xf39F…,
// inspector = 0x3C44…, lender = 0x90F7…).
// Flow: list -> commit -> inspection passed -> request financing -> lender
// approves + disburses -> buyer pays down payment -> buyer cancels -> buyer
// gets the full escrowed share back and the lender gets the disbursed loan
// back, token stays with the seller.
const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

const ETH = (n) => ethers.BigNumber.from(n).mul(ethers.utils.parseEther("1"));

async function main() {
  const signers = await ethers.getSigners();
  const config = JSON.parse(
    fs.readFileSync(path.join(__dirname, "..", "src", "config.json"), "utf8"),
  );
  const chainId = String((await ethers.provider.getNetwork()).chainId);
  const cc = config[chainId];
  if (!cc || !cc.millowEscrow) throw new Error("deployMillow first");

  const roles = config.demoRoles || {};
  const seller = await ethers.getSigner(roles.seller);
  const buyer = await ethers.getSigner(roles.buyer);
  const inspector = await ethers.getSigner(roles.inspector);
  const lender = await ethers.getSigner(roles.lender);
  const stranger = signers[4];

  const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
  const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
  const MillowEscrow = await ethers.getContractFactory("MillowEscrow");

  const nft = await PropertyNFT.attach(cc.propertyNft.address);
  const registry = await PropertyRegistry.attach(cc.propertyRegistry.address);
  const escrow = await MillowEscrow.attach(cc.millowEscrow.address);

  const tokenId = Number(process.env.TOKEN_ID || 6);
  const priceWei = ETH("180");
  const earnestWei = ETH("18");
  const downPaymentPctBps = 2000; // 20%
  const downPaymentWei = priceWei.mul(downPaymentPctBps).div(10000);
  const loanAmountWei = priceWei.sub(downPaymentWei);

  const saleInfo = () => escrow.sales(tokenId);
  const statusName = async () =>
    ["None", "Listed", "UnderContract", "Approved", "Finalized", "Cancelled"][
      Number((await saleInfo()).status)
    ];
  const log = (m) => console.log(m);

  const mreidId = await nft.propertyOf(tokenId);
  log(`Cancellation demo on token ${tokenId} (${mreidId})`);
  log(`price=${ethers.utils.formatEther(priceWei)} ETH, earnest=${ethers.utils.formatEther(earnestWei)} ETH`);
  log(`down payment=${downPaymentPctBps / 100}% (${ethers.utils.formatEther(downPaymentWei)} ETH), loan=${ethers.utils.formatEther(loanAmountWei)} ETH`);

  if (!(await escrow.isOnOffer(tokenId))) {
    const approved = await nft.getApproved(tokenId);
    if (approved !== escrow.address) {
      await (await nft.connect(seller).approve(escrow.address, tokenId)).wait();
    }
    const listTx = await escrow
      .connect(seller)
      .listForSaleWithRequirements(tokenId, priceWei, earnestWei, true, true);
    await listTx.wait();
    log(`Listed token ${tokenId} with inspection + lender requirements (tx ${listTx.hash})`);
  }

  if (Number((await saleInfo()).status) < 2) {
    await (
      await escrow.connect(buyer).commitAndDeposit(tokenId, { value: earnestWei })
    ).wait();
    log(`Commit & deposit ${ethers.utils.formatEther(earnestWei)} ETH -> ${await statusName()}`);
  }

  if (!(await saleInfo()).inspectionPassed) {
    await (await escrow.connect(inspector).setInspectionStatus(tokenId, true)).wait();
    log("Inspector recorded inspection passed");
  }

  const fin = (await saleInfo()).financing;
  if (!fin.requested || fin.rejected) {
    await (
      await escrow.connect(buyer).requestFinancing(tokenId, downPaymentPctBps, 700, 240)
    ).wait();
    log("Buyer requested financing: 20% down @ 7%/yr, 240 months");
  }
  if (!fin.approved) {
    await (await escrow.connect(lender).approveFinancing(tokenId)).wait();
    log("Lender approved financing");
  }
  await (
    await escrow.connect(lender).fundLoan(tokenId, { value: loanAmountWei })
  ).wait();
  log(`Lender disbursed ${ethers.utils.formatEther(loanAmountWei)} ETH loan into escrow`);
  await (
    await escrow.connect(buyer).payDownPayment(tokenId, { value: downPaymentWei.sub(earnestWei) })
  ).wait();
  log(`Buyer paid remaining down payment ${ethers.utils.formatEther(downPaymentWei.sub(earnestWei))} ETH`);

  const before = await saleInfo();
  log(`Escrow holds buyer=${ethers.utils.formatEther(before.buyerFundedWei)} ETH + lender=${ethers.utils.formatEther(before.lenderFundedWei)} ETH`);

  const expectRevert = async (label, fn) => {
    try {
      await fn();
      log(`  UNEXPECTED: ${label} did NOT revert`);
    } catch (e) {
      const reason = e.reason || e.message || String(e);
      log(`  ${label} reverted as expected: "${reason}"`);
    }
  };

  // Only the seller or the buyer may cancel.
  await expectRevert("stranger cancels", () =>
    escrow.connect(stranger).cancelSale(tokenId),
  );

  const buyerBefore = await ethers.provider.getBalance(buyer.address);
  const lenderBefore = await ethers.provider.getBalance(lender.address);
  const escrowContractBefore = await ethers.provider.getBalance(escrow.address);

  await (await escrow.connect(buyer).cancelSale(tokenId)).wait();
  log("Buyer cancelled the sale");

  const escrowContractAfter = await ethers.provider.getBalance(escrow.address);
  const nftOwner = await nft.ownerOf(tokenId);

  const buyerChange = (await ethers.provider.getBalance(buyer.address)).sub(buyerBefore);
  const lenderChange = (await ethers.provider.getBalance(lender.address)).sub(lenderBefore);

  log("");
  log("=== VERIFICATION ===");
  log(`status now: ${await statusName()} (Struct cleared to None after cancel; SaleCancelled was emitted)`);
  log(`NFT still with seller: ${nftOwner.toLowerCase() === seller.address.toLowerCase()}`);
  log(`registry.isOnOffer: ${await registry.isOnOffer(tokenId)} (expected false)`);
  log(`escrow balance ${ethers.utils.formatEther(escrowContractBefore)} -> ${ethers.utils.formatEther(escrowContractAfter)} ETH (expected 0)`);
  log(`buyer balance change net of gas: ${ethers.utils.formatEther(buyerChange)} ETH (refund ~${ethers.utils.formatEther(before.buyerFundedWei)})`);
  log(`lender balance change net of gas: ${ethers.utils.formatEther(lenderChange)} ETH (refund ~${ethers.utils.formatEther(before.lenderFundedWei)})`);

  const ok =
    Number((await saleInfo()).status) === 0 &&
    nftOwner.toLowerCase() === seller.address.toLowerCase() &&
    !(await registry.isOnOffer(tokenId)) &&
    escrowContractAfter.isZero() &&
    buyerChange.gte(before.buyerFundedWei.mul(999).div(1000)) &&
    lenderChange.gte(before.lenderFundedWei.mul(999).div(1000));
  log(ok ? "RESULT: PASS - cancellation + split refunds verified on-chain" : "RESULT: FAIL");
  if (!ok) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});