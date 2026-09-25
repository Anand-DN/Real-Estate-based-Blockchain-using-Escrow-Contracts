// LIVE walkthrough of the full MILLOW role-gated transaction state machine.
//
// Drives the REAL PropertyNFT / PropertyRegistry / MillowEscrow contracts on
// the running Hardhat node (chain 31337) with the canonical demo signers.  NOT
// a UI test: it proves the on-chain gates that the frontend surfaces:
//   - minting is MINTER_ROLE-only (infrastructure for provisioning demo tokens,
//     NOT an application role — the simplified app has no Registrar at all),
//   - only the NFT owner can list / only with the escrow approved,
//   - the extended (Inspector + Lender) sale settles only after inspection,
//     an approved and fully disbursed loan, full funding and both approvals,
//   - every role-gated action reverts for every other account.
//
// The whole run happens inside an evm_snapshot taken in beforeAll and is rolled
// back in afterAll, so the long-lived demo chain is never polluted.  Uses
// MREID_99990001 / MREID_99990002, which are outside the seeded 29,135-property
// dataset (MREID_0000001..MREID_0029135), so no other live test interferes.
//
// Requires: Hardhat node on 127.0.0.1:8545.  The chain-fixed MINTER_ROLE holder
// below is the deployed PropertyNFT minter (former demo Registrar address).

jest.setTimeout(240000);

const ethers = require("ethers");
const config = require("./config.json");

// eslint-disable-next-line import/no-anonymous-default-export
const nftArtifact = require("../artifacts/contracts/PropertyNFT.sol/PropertyNFT.json");
const registryArtifact = require("../artifacts/contracts/PropertyRegistry.sol/PropertyRegistry.json");
const escrowArtifact = require("../artifacts/contracts/MillowEscrow.sol/MillowEscrow.json");

// The chain-fixed MINTER_ROLE holder (deployed PropertyNFT minter).  This
// account provisions demo tokens for the walkthrough; it is infrastructure,
// not an application role — config.demoRoles no longer contains it.
const MINTER = "0xBcd4042DE499D14e55001CcbB24a551F3b954096";
const SELLER = config.demoRoles.seller;
const BUYER = config.demoRoles.buyer;
const INSPECTOR = config.demoRoles.inspector;
const LENDER = config.demoRoles.lender;
const STRANGER = "0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65";

const PRICE = ethers.utils.parseEther("300");
const EARNEST = ethers.utils.parseEther("30");
const DOWN_PAYMENT_BPS = 2000; // 20% = 60 test ETH
const DOWN_PAYMENT = ethers.utils.parseEther("60");
const LOAN = ethers.utils.parseEther("240"); // 80%

const expectRevert = async (target, reasonPart) => {
  let error;
  try {
    await (typeof target === "function" ? target() : target);
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeTruthy();
  const text = String(
    (error && error.reason) || (error && error.message) || error,
  );
  expect(text).toContain(reasonPart);
};

describe("MILLOW role-gated workflow - live against the local chain", () => {
  let provider;
  let nft;
  let registry;
  let escrow;
  let signers;
  let snapshotId;

  beforeAll(async () => {
    provider = new ethers.providers.JsonRpcProvider("http://127.0.0.1:8545");
    const cfg = config["31337"];
    nft = new ethers.Contract(
      cfg.propertyNft.address,
      nftArtifact.abi,
      provider,
    );
    registry = new ethers.Contract(
      cfg.propertyRegistry.address,
      registryArtifact.abi,
      provider,
    );
    escrow = new ethers.Contract(
      cfg.millowEscrow.address,
      escrowArtifact.abi,
      provider,
    );
    signers = {
      minter: provider.getSigner(MINTER),
      seller: provider.getSigner(SELLER),
      buyer: provider.getSigner(BUYER),
      inspector: provider.getSigner(INSPECTOR),
      lender: provider.getSigner(LENDER),
      stranger: provider.getSigner(STRANGER),
    };
    snapshotId = await provider.send("evm_snapshot", []);
  });

  afterAll(async () => {
    await provider.send("evm_revert", [snapshotId]);
  });

  async function registerToken(mreidId) {
    expect(await nft.tokenByProperty(mreidId)).toEqual(
      ethers.BigNumber.from(0),
    );
    const uri = `http://localhost:3001/metadata/${mreidId}.json`;
    await (
      await nft.connect(signers.minter).mintProperty(mreidId, uri)
    ).wait();
    const tokenId = await nft.tokenByProperty(mreidId);
    expect(tokenId).not.toEqual(ethers.BigNumber.from(0));
    return tokenId;
  }

  async function deliverToSeller(tokenId) {
    await (
      await nft
        .connect(signers.minter)
        .transferFrom(MINTER, SELLER, tokenId)
    ).wait();
    expect(await nft.ownerOf(tokenId)).toEqual(SELLER);
    await (await registry.connect(signers.seller).list(tokenId)).wait();
    expect(await registry.isOnOffer(tokenId)).toBe(true);
    expect(await registry.sellerOf(tokenId)).toEqual(SELLER);
  }

  it("runs the full extended workflow with every role gate enforced on-chain", async () => {
    const MREID = "MREID_99990001";

    // --- Preconditions: the chain MINTER_ROLE is provisioned as configured --
    const MINTER_ROLE = await nft.MINTER_ROLE();
    expect(await nft.hasRole(MINTER_ROLE, MINTER)).toBe(true);
    expect(await nft.hasRole(MINTER_ROLE, BUYER)).toBe(false);

    // --- Minting is MINTER_ROLE-only (token provisioning infrastructure) ----
    await expectRevert(
      nft
        .connect(signers.stranger)
        .mintProperty(MREID, `http://localhost:3001/metadata/${MREID}.json`),
      "missing role",
    );
    const tokenId = await registerToken(MREID);
    expect(await nft.tokenByProperty(MREID)).toEqual(tokenId);
    expect(await nft.ownerOf(tokenId)).toEqual(MINTER);
    expect(await nft.propertyOf(tokenId)).toEqual(MREID);

    // The freshly minted token is still owned by the MINTER: the seller
    // cannot list a token they do not own.
    await expectRevert(
      escrow
        .connect(signers.seller)
        .listForSaleWithRequirements(tokenId, PRICE, EARNEST, true, true),
      "Only the NFT owner can list",
    );

    // --- MINTER delivers the NFT to the seller (registry "on offer") --------
    await deliverToSeller(tokenId);

    // --- Listing requires ownership AND escrow approval --------------------
    await expectRevert(
      escrow
        .connect(signers.seller)
        .listForSaleWithRequirements(tokenId, PRICE, EARNEST, true, true),
      "Escrow not approved",
    );
    await (
      await nft.connect(signers.seller).approve(escrow.address, tokenId)
    ).wait();
    await expectRevert(
      escrow
        .connect(signers.buyer)
        .listForSaleWithRequirements(tokenId, PRICE, EARNEST, true, true),
      "Only the NFT owner can list",
    );
    const listTx = await (
      await escrow
        .connect(signers.seller)
        .listForSaleWithRequirements(tokenId, PRICE, EARNEST, true, true)
    ).wait();
    expect(listTx.events.find((e) => e.event === "SaleListed")).toBeTruthy();
    expect(Number(await escrow.statusOf(tokenId))).toBe(1); // Listed
    expect(await escrow.isOnOffer(tokenId)).toBe(true);
    expect(await escrow.priceOf(tokenId)).toEqual(PRICE);
    expect(await escrow.earnestOf(tokenId)).toEqual(EARNEST);
    await expectRevert(
      escrow
        .connect(signers.seller)
        .listForSaleWithRequirements(tokenId, PRICE, EARNEST, true, true),
      "Already listed",
    );

    // --- Commit: earnest must be deposited, state moves to UnderContract ----
    await expectRevert(
      escrow.connect(signers.buyer).commitAndDeposit(tokenId, {
        value: EARNEST.sub(ethers.utils.parseEther("1")),
      }),
      "Earnest below required",
    );
    // On-chain any account may become the buyer by depositing; the strict
    // commit gating the UI exposes is a frontend rule (see TransactionPanel).
    const commitTx = await (
      await escrow.connect(signers.buyer).commitAndDeposit(tokenId, {
        value: EARNEST,
      })
    ).wait();
    expect(commitTx.events.find((e) => e.event === "EarnestDeposited")).toBeTruthy();
    expect(commitTx.events.find((e) => e.event === "InspectionRequested")).toBeTruthy();
    expect(Number(await escrow.statusOf(tokenId))).toBe(2); // UnderContract
    expect(await escrow.buyerFundedOf(tokenId)).toEqual(EARNEST);
    // A second deposit on an already-committed sale is rejected.
    await expectRevert(
      escrow.connect(signers.stranger).commitAndDeposit(tokenId, {
        value: EARNEST,
      }),
      "Wrong sale state",
    );

    // --- The seller's approval is gated until inspection passes -------------
    await expectRevert(
      escrow.connect(signers.seller).approveSale(tokenId),
      "Seller approval not yet allowed",
    );

    // --- Inspection is INSPECTOR_ROLE-only ---------------------------------
    await expectRevert(
      escrow.connect(signers.seller).setInspectionStatus(tokenId, true),
      "missing role",
    );
    await expectRevert(
      escrow.connect(signers.stranger).setInspectionStatus(tokenId, true),
      "missing role",
    );
    const inspectTx = await (
      await escrow.connect(signers.inspector).setInspectionStatus(tokenId, true)
    ).wait();
    const inspected = inspectTx.events.find(
      (e) => e.event === "InspectionUpdated",
    );
    expect(inspected).toBeTruthy();
    expect(inspected.args.passed).toBe(true);

    // --- Financing request is buyer-only ------------------------------------
    await expectRevert(
      escrow
        .connect(signers.stranger)
        .requestFinancing(tokenId, DOWN_PAYMENT_BPS, 850, 240),
      "Only the buyer",
    );
    const financeTx = await (
      await escrow
        .connect(signers.buyer)
        .requestFinancing(tokenId, DOWN_PAYMENT_BPS, 850, 240)
    ).wait();
    const requested = financeTx.events.find(
      (e) => e.event === "FinancingRequested",
    );
    expect(requested).toBeTruthy();
    expect(requested.args.downPaymentWei).toEqual(DOWN_PAYMENT);

    // --- Financing decisions are LENDER_ROLE-only --------------------------
    await expectRevert(
      escrow.connect(signers.buyer).approveFinancing(tokenId),
      "missing role",
    );
    await expectRevert(
      escrow.connect(signers.stranger).approveFinancing(tokenId),
      "missing role",
    );
    const approveLoanTx = await (
      await escrow.connect(signers.lender).approveFinancing(tokenId)
    ).wait();
    expect(
      approveLoanTx.events.find((e) => e.event === "LoanApproved"),
    ).toBeTruthy();

    // --- Funding: loan only from the lender, down payment only from buyer ---
    await expectRevert(
      escrow.connect(signers.seller).fundLoan(tokenId, {
        value: LOAN,
      }),
      "missing role",
    );
    await (
      await escrow.connect(signers.lender).fundLoan(tokenId, { value: LOAN })
    ).wait();
    expect(await escrow.lenderFundedOf(tokenId)).toEqual(LOAN);
    await (
      await escrow
        .connect(signers.buyer)
        .payDownPayment(tokenId, { value: DOWN_PAYMENT.sub(EARNEST) })
    ).wait();
    expect(await escrow.buyerFundedOf(tokenId)).toEqual(DOWN_PAYMENT);
    expect(await escrow.fundedOf(tokenId)).toEqual(PRICE);
    // The buyer may not exceed the approved down payment.
    await expectRevert(
      escrow.connect(signers.buyer).payDownPayment(tokenId, {
        value: ethers.utils.parseEther("1"),
      }),
      "Above down payment",
    );

    // --- Finalize before the parties approve is rejected -------------------
    await expectRevert(
      escrow.connect(signers.stranger).finalizeSale(tokenId),
      "Wrong sale state",
    );

    // --- Both parties approve; the seller is gated first --------------------
    await expectRevert(
      escrow.connect(signers.seller).approveSale(tokenId),
      "Seller approval not yet allowed",
    );
    const buyerApprove = await (
      await escrow.connect(signers.buyer).approveBuyer(tokenId)
    ).wait();
    expect(
      buyerApprove.events.find((e) => e.event === "BuyerApproved"),
    ).toBeTruthy();
    const sellerApprove = await (
      await escrow.connect(signers.seller).approveSeller(tokenId)
    ).wait();
    expect(
      sellerApprove.events.find((e) => e.event === "SellerApproved"),
    ).toBeTruthy();
    expect(Number(await escrow.statusOf(tokenId))).toBe(3); // Approved

    // --- Finalize settles: NFT to buyer, exact price to seller ---------------
    const sellerBalanceBefore = await provider.getBalance(SELLER);
    const finalizeTx = await (
      await escrow.connect(signers.stranger).finalizeSale(tokenId)
    ).wait();
    const finalized = finalizeTx.events.find(
      (e) => e.event === "SaleFinalized",
    );
    expect(finalized).toBeTruthy();
    expect(finalized.args.buyer).toEqual(BUYER);
    expect(finalized.args.seller).toEqual(SELLER);
    expect(finalized.args.priceWei).toEqual(PRICE);

    // statusOf reports 0 once the sale is unlisted, so settle is verified via
    // the sale being off offer, ownership transfer and the escrow emptying.
    expect(await escrow.isOnOffer(tokenId)).toBe(false);
    expect(await nft.ownerOf(tokenId)).toEqual(BUYER);
    expect(await registry.isOnOffer(tokenId)).toBe(false); // auto-unlisted
    expect(await provider.getBalance(escrow.address)).toEqual(
      ethers.BigNumber.from(0),
    );
    const sellerBalanceAfter = await provider.getBalance(SELLER);
    expect(sellerBalanceAfter.sub(sellerBalanceBefore)).toEqual(PRICE);
  });

  it("refunds the committed earnest when a sale is cancelled", async () => {
    const MREID = "MREID_99990002";
    const tokenId = await registerToken(MREID);
    await deliverToSeller(tokenId);
    await (
      await nft.connect(signers.seller).approve(escrow.address, tokenId)
    ).wait();
    await (
      await escrow.connect(signers.seller).listForSale(tokenId, PRICE, EARNEST)
    ).wait();
    expect(Number(await escrow.statusOf(tokenId))).toBe(1);

    // A stranger can never cancel someone else's sale (once a participant
    // exists, i.e. once the buyer has committed).
    await (
      await escrow.connect(signers.buyer).commitAndDeposit(tokenId, {
        value: EARNEST,
      })
    ).wait();
    expect(await escrow.buyerFundedOf(tokenId)).toEqual(EARNEST);
    await expectRevert(
      escrow.connect(signers.stranger).cancelSale(tokenId),
      "Only the seller or the buyer",
    );

    const buyerBalanceBefore = await provider.getBalance(BUYER);
    await (
      await escrow.connect(signers.buyer).cancelSale(tokenId)
    ).wait();

    // State cleared, token stays with the seller, earnest fully returned.
    expect(Number(await escrow.statusOf(tokenId))).toBe(0);
    expect(await escrow.isOnOffer(tokenId)).toBe(false);
    expect(await nft.ownerOf(tokenId)).toEqual(SELLER);
    expect(await registry.isOnOffer(tokenId)).toBe(false); // auto-unlisted
    expect(await provider.getBalance(escrow.address)).toEqual(
      ethers.BigNumber.from(0),
    );
    // The buyer got the full earnest back (barring the cancel tx gas fee).
    const buyerBalanceAfter = await provider.getBalance(BUYER);
    expect(
      buyerBalanceAfter
        .sub(buyerBalanceBefore)
        .gte(EARNEST.sub(ethers.utils.parseEther("0.1"))),
    ).toBe(true);
  });
});