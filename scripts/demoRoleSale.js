// Live demo: full role-based escrow sale on the running Hardhat node for a
// listing created through listForSaleWithRequirements().
//
// Run: npx hardhat run scripts/demoRoleSale.js --network localhost
// Defaults to token 3 (MREID_0005522); override with TOKEN_ID.  Uses the
// canonical demo roles from src/config.json (seller = 0x7099…, buyer = 0xf39F…,
// inspector = 0x3C44…, lender = 0x90F7…).
// The strict extended workflow is exercised end to end:
//   list -> commit -> inspection passed -> request financing -> lender
//   approve -> lender disburses the loan -> buyer pays the down payment ->
//   buyer approves -> seller approves (gated) -> finalize.
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

  const tokenId = Number(process.env.TOKEN_ID || 3);
  const priceWei = ETH("300");
  const earnestWei = ETH("30");
  // 20% down payment, 8% p.a., 240 months.
  const downPaymentPctBps = Number(process.env.DOWN_PAYMENT_PCT_BPS || 2000);
  const interestRateBps = Number(process.env.INTEREST_RATE_BPS || 800);
  const loanTenureMonths = Number(process.env.LOAN_TENURE_MONTHS || 240);
  const downPaymentWei = priceWei.mul(downPaymentPctBps).div(10000);
  const loanAmountWei = priceWei.sub(downPaymentWei);

  const saleInfo = () => escrow.sales(tokenId);
  const statusName = async () =>
    ["None", "Listed", "UnderContract", "Approved", "Finalized", "Cancelled"][
      Number((await saleInfo()).status)
    ];
  const log = (m) => console.log(m);

  const mreidId = await nft.propertyOf(tokenId);
  log(`Role sale demo on token ${tokenId} (${mreidId})`);
  log(`seller=${seller.address} buyer=${buyer.address} inspector=${inspector.address} lender=${lender.address}`);

  log("");
  log("=== TERMS ===");
  log(`price=${ethers.utils.formatEther(priceWei)} ETH, earnest=${ethers.utils.formatEther(earnestWei)} ETH`);
  log(`down payment=${downPaymentPctBps / 100}% (${ethers.utils.formatEther(downPaymentWei)} ETH)`);
  log(`loan=${ethers.utils.formatEther(loanAmountWei)} ETH @ ${interestRateBps / 100}%/yr, ${loanTenureMonths} months`);

  const [iRole, lRole] = await Promise.all([
    escrow.INSPECTOR_ROLE(),
    escrow.LENDER_ROLE(),
  ]);
  const [isInspector, isLender] = await Promise.all([
    escrow.hasRole(iRole, inspector.address),
    escrow.hasRole(lRole, lender.address),
  ]);
  log(`inspector hasRole: ${isInspector}, lender hasRole: ${isLender}`);

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

  const expectRevert = async (label, fn) => {
    try {
      await fn();
      log(`  UNEXPECTED: ${label} did NOT revert`);
    } catch (e) {
      const reason = e.reason || e.message || String(e);
      log(`  ${label} reverted as expected: "${reason}"`);
    }
  };

  if (Number((await saleInfo()).status) < 2) {
    await (
      await escrow.connect(buyer).commitAndDeposit(tokenId, { value: earnestWei })
    ).wait();
    log(`Commit & deposit ${ethers.utils.formatEther(earnestWei)} ETH -> ${await statusName()}`);
  }

  // Lender cannot approve before a financing request exists.
  await expectRevert("lender approve with no request", () =>
    escrow.connect(lender).approveFinancing(tokenId),
  );

  // Buyer cannot request financing before inspection has passed.
  await expectRevert("financing request before inspection", () =>
    escrow.connect(buyer).requestFinancing(tokenId, downPaymentPctBps, interestRateBps, loanTenureMonths),
  );

  // Only the assigned inspector may record an inspection result.
  await expectRevert("stranger inspector action", () =>
    escrow.connect(stranger).setInspectionStatus(tokenId, true),
  );
  await (
    await escrow.connect(inspector).setInspectionStatus(tokenId, true)
  ).wait();
  log("Inspector recorded inspection passed (on-chain, not faked)");

  await expectRevert("financing request with invalid down payment (33%)", () =>
    escrow.connect(buyer).requestFinancing(tokenId, 3300, interestRateBps, loanTenureMonths),
  );

  await (
    await escrow
      .connect(buyer)
      .requestFinancing(tokenId, downPaymentPctBps, interestRateBps, loanTenureMonths)
  ).wait();
  log(`Buyer requested financing: ${downPaymentPctBps / 100}% down (${ethers.utils.formatEther(downPaymentWei)} ETH) + ${ethers.utils.formatEther(loanAmountWei)} ETH loan`);

  await expectRevert("lender funds before approving", () =>
    escrow.connect(lender).fundLoan(tokenId, { value: loanAmountWei }),
  );
  await expectRevert("buyer pays down payment before financing approved", () =>
    escrow.connect(buyer).payDownPayment(tokenId, { value: downPaymentWei.sub(earnestWei) }),
  );

  // Even with the financing request in place, the seller's approval stays
  // gated until the loan is approved AND fully disbursed by the lender.
  await expectRevert("seller approve before loan approved", () =>
    escrow.connect(seller).approveSeller(tokenId),
  );

  await (
    await escrow.connect(lender).approveFinancing(tokenId)
  ).wait();
  log("Lender approved the financing request");

  await expectRevert("seller approve before loan disbursed", () =>
    escrow.connect(seller).approveSeller(tokenId),
  );
  await expectRevert("buyer approve above down payment already possible", () =>
    escrow.connect(buyer).payDownPayment(tokenId, { value: downPaymentWei }),
  );

  // Loan must be fully disbursed for the escrow to settle.
  await (
    await escrow.connect(lender).fundLoan(tokenId, { value: loanAmountWei })
  ).wait();
  log(`Lender disbursed ${ethers.utils.formatEther(loanAmountWei)} ETH loan into escrow`);

  // Earliest deposit already counted; top up to the down payment ceiling.
  await (
    await escrow
      .connect(buyer)
      .payDownPayment(tokenId, { value: downPaymentWei.sub(earnestWei) })
  ).wait();
  log(`Buyer paid remaining down payment ${ethers.utils.formatEther(downPaymentWei.sub(earnestWei))} ETH`);

  const s = await saleInfo();
  log(`Funding: buyer=${ethers.utils.formatEther(s.buyerFundedWei)} ETH + lender=${ethers.utils.formatEther(s.lenderFundedWei)} ETH / ${ethers.utils.formatEther(s.priceWei)} ETH`);

  // Buyer approves without restriction.
  await (await escrow.connect(buyer).approveBuyer(tokenId)).wait();
  log("Buyer approved");

  // Once the loan is fully disbursed, the seller's approval is unblocked.
  await (await escrow.connect(seller).approveSeller(tokenId)).wait();
  log(`Seller approved (gated) -> ${await statusName()}`);

  // Approvals may be withdrawn, but re-approving only works once the gating
  // conditions still hold; re-approving here succeeds because the escrow is
  // funded, inspection passed, the loan disbursed, and the buyer approved.
  await (await escrow.connect(seller).disapproveSale(tokenId)).wait();
  log("Seller withdrew approval -> UnderContract");
  await (await escrow.connect(seller).approveSeller(tokenId)).wait();
  log(`Seller re-approved -> ${await statusName()}`);

  // finalizeSale is permissionless in this prototype: once the escrow
  // conditions are met, any caller may settle the defined payment.  Party
  // restrictions apply to approvals; execution is automatic.
  await (await escrow.connect(stranger).finalizeSale(tokenId)).wait();
  log("Finalize -> " + (await statusName()));

  const after = await saleInfo();
  const nftOwner = await nft.ownerOf(tokenId);
  const onOffer = await registry.isOnOffer(tokenId);
  const escrowBalance = await ethers.provider.getBalance(escrow.address);

  log("");
  log("=== VERIFICATION ===");
  log(`status: ${await statusName()} (expected Finalized)`);
  log(`NFT owner now: ${nftOwner}`);
  log(`NFT belongs to buyer: ${nftOwner.toLowerCase() === buyer.address.toLowerCase()}`);
  log(`registry.isOnOffer: ${onOffer} (expected false)`);
  log(`escrow balance now: ${ethers.utils.formatEther(escrowBalance)} ETH (expected 0)`);
  log(`seller received ${ethers.utils.formatEther(after.priceWei)} ETH`);

  const ok =
    Number(after.status) === 4 &&
    nftOwner.toLowerCase() === buyer.address.toLowerCase() &&
    !onOffer &&
    escrowBalance.isZero() &&
    isInspector &&
    isLender;
  log(ok ? "RESULT: PASS - extended role-based sale verified on-chain" : "RESULT: FAIL");
  if (!ok) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});