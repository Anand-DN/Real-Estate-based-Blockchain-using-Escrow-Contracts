// Live demo: financing rejection and re-request path on the running Hardhat
// node for a listing created through listForSaleWithRequirements().
//
// Run: npx hardhat run scripts/demoFinancingRejection.js --network localhost
// Defaults to token 4 (MREID_0009829); override with TOKEN_ID.  Uses the
// canonical demo roles from src/config.json (seller = 0x7099…, buyer = 0xf39F…,
// inspector = 0x3C44…, lender = 0x90F7…).
// Flow: list -> commit -> inspection passed -> request financing -> lender
// rejects -> buyer re-requests with adjusted terms -> lender approves ->
// lender disburses -> buyer pays down payment -> both approve -> finalize.
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
  const inspector = await ethers.getSigner(roles.inspector);
  const lender = await ethers.getSigner(roles.lender);

  const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
  const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
  const MillowEscrow = await ethers.getContractFactory("MillowEscrow");

  const nft = await PropertyNFT.attach(cc.propertyNft.address);
  const registry = await PropertyRegistry.attach(cc.propertyRegistry.address);
  const escrow = await MillowEscrow.attach(cc.millowEscrow.address);

  const tokenId = Number(process.env.TOKEN_ID || 4);
  const priceWei = ETH("240");
  const earnestWei = ETH("24");
  const downPaymentPctBps = 2000; // 20%
  const rate = (bps) => bps;
  const downPaymentWei = priceWei.mul(downPaymentPctBps).div(10000);
  const loanAmountWei = priceWei.sub(downPaymentWei);

  const saleInfo = () => escrow.sales(tokenId);
  const statusName = async () =>
    ["None", "Listed", "UnderContract", "Approved", "Finalized", "Cancelled"][
      Number((await saleInfo()).status)
    ];
  const log = (m) => console.log(m);

  const mreidId = await nft.propertyOf(tokenId);
  log(`Financing rejection demo on token ${tokenId} (${mreidId})`);
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
  if (!fin.requested) {
    await (
      await escrow.connect(buyer).requestFinancing(tokenId, downPaymentPctBps, rate(650), 240)
    ).wait();
    log("Buyer requested financing: 20% down @ 6.5%/yr, 240 months");
  }

  // Rejection path: lender turns down the first request.
  await (await escrow.connect(lender).rejectFinancing(tokenId)).wait();
  log("Lender REJECTED the first financing request");

  const fin2 = (await saleInfo()).financing;
  log(`after rejection: requested=${fin2.requested} approved=${fin2.approved} rejected=${fin2.rejected}`);

  const expectRevert = async (label, fn) => {
    try {
      await fn();
      log(`  UNEXPECTED: ${label} did NOT revert`);
    } catch (e) {
      const reason = e.reason || e.message || String(e);
      log(`  ${label} reverted as expected: "${reason}"`);
    }
  };

  // The buyer may re-request with adjusted terms before any loan is funded.
  await expectRevert("lender funds before re-approval", () =>
    escrow.connect(lender).fundLoan(tokenId, { value: loanAmountWei }),
  );

  await (
    await escrow
      .connect(buyer)
      .requestFinancing(tokenId, downPaymentPctBps, rate(800), 240)
  ).wait();
  log("Buyer re-requested financing: 20% down @ 8%/yr, 240 months");

  await (await escrow.connect(lender).approveFinancing(tokenId)).wait();
  log("Lender approved the re-requested financing");
  await (
    await escrow.connect(lender).fundLoan(tokenId, { value: loanAmountWei })
  ).wait();
  log(`Lender disbursed ${ethers.utils.formatEther(loanAmountWei)} ETH loan into escrow`);
  await (
    await escrow.connect(buyer).payDownPayment(tokenId, { value: downPaymentWei.sub(earnestWei) })
  ).wait();
  log(`Buyer paid remaining down payment ${ethers.utils.formatEther(downPaymentWei.sub(earnestWei))} ETH`);

  await (await escrow.connect(buyer).approveBuyer(tokenId)).wait();
  log("Buyer approved");
  await (await escrow.connect(seller).approveSeller(tokenId)).wait();
  log(`Seller approved (gated) -> ${await statusName()}`);

  await (await escrow.connect(seller).finalizeSale(tokenId)).wait();
  log("Finalize -> " + (await statusName()));

  const after = await saleInfo();
  const nftOwner = await nft.ownerOf(tokenId);
  const onOffer = await registry.isOnOffer(tokenId);
  const escrowBalance = await ethers.provider.getBalance(escrow.address);

  log("");
  log("=== VERIFICATION ===");
  log(`status: ${await statusName()} (expected Finalized)`);
  log(`NFT belongs to buyer: ${nftOwner.toLowerCase() === buyer.address.toLowerCase()}`);
  log(`registry.isOnOffer: ${onOffer} (expected false)`);
  log(`escrow balance now: ${ethers.utils.formatEther(escrowBalance)} ETH (expected 0)`);

  const ok =
    Number(after.status) === 4 &&
    nftOwner.toLowerCase() === buyer.address.toLowerCase() &&
    !onOffer &&
    escrowBalance.isZero();
  log(ok ? "RESULT: PASS - rejection + re-request flow verified on-chain" : "RESULT: FAIL");
  if (!ok) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});