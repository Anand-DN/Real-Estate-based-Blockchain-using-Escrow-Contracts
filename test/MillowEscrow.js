const { expect } = require("chai");
const { ethers } = require("hardhat");

const ETH = ethers.utils.parseEther;

describe("MillowEscrow", () => {
  let owner, seller, buyer, attacker, inspector, lender;
  let propertyNFT, propertyRegistry, escrow;

  beforeEach(async () => {
    [owner, seller, buyer, attacker, inspector, lender] = await ethers.getSigners();

    const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
    propertyNFT = await PropertyNFT.deploy();

    const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
    propertyRegistry = await PropertyRegistry.deploy(propertyNFT.address);

    const MillowEscrow = await ethers.getContractFactory("MillowEscrow");
    escrow = await MillowEscrow.deploy(
      propertyNFT.address,
      propertyRegistry.address,
    );

    const LISTING_MANAGER_ROLE = await propertyRegistry.LISTING_MANAGER_ROLE();
    await propertyRegistry.grantRole(LISTING_MANAGER_ROLE, escrow.address);

    await propertyNFT.connect(owner).mintProperty("MREID_0000001", "uri://1");
    await propertyNFT.connect(owner).mintProperty("MREID_0000002", "uri://2");
  });

  const listToken1 = async () => {
    await propertyNFT.connect(owner).approve(escrow.address, 1);
    return escrow.connect(owner).listForSale(1, ETH("10"), ETH("1"));
  };

  describe("Deployment", () => {
    it("Returns the NFT address", async () => {
      expect(await escrow.propertyNFT()).to.equal(propertyNFT.address);
    });

    it("Returns the registry address", async () => {
      expect(await escrow.propertyRegistry()).to.equal(propertyRegistry.address);
    });

    it("Grants the deployer the admin role", async () => {
      const DEFAULT_ADMIN_ROLE = ethers.constants.HashZero;
      expect(await escrow.hasRole(DEFAULT_ADMIN_ROLE, owner.address)).to.equal(true);
    });
  });

  describe("Listing", () => {
    it("Lists a token the caller owns", async () => {
      await expect(listToken1()).to.emit(escrow, "SaleListed").withArgs(
        1,
        owner.address,
        ETH("10"),
        ETH("1"),
      );
      expect(await escrow.isOnOffer(1)).to.equal(true);
      expect(await escrow.statusOf(1)).to.equal(1);
      expect((await escrow.sales(1)).seller).to.equal(owner.address);
      expect(await escrow.priceOf(1)).to.equal(ETH("10"));
      expect(await escrow.earnestOf(1)).to.equal(ETH("1"));
    });

    it("Rejects a non-owner listing", async () => {
      await propertyNFT.connect(owner).approve(escrow.address, 1);
      await expect(
        escrow.connect(buyer).listForSale(1, ETH("10"), ETH("1")),
      ).to.be.revertedWith("Only the NFT owner can list");
    });

    it("Rejects an unapproved escrow operator", async () => {
      await expect(
        escrow.connect(owner).listForSale(1, ETH("10"), ETH("1")),
      ).to.be.revertedWith("Escrow not approved");
    });

    it("Rejects a double listing", async () => {
      await listToken1();
      await expect(
        escrow.connect(owner).listForSale(1, ETH("10"), ETH("1")),
      ).to.be.revertedWith("Already listed");
    });

    it("Rejects zero and out-of-range earnest", async () => {
      await propertyNFT.connect(owner).approve(escrow.address, 1);
      await expect(
        escrow.connect(owner).listForSale(1, 0, ETH("1")),
      ).to.be.revertedWith("Price required");
      await expect(
        escrow.connect(owner).listForSale(1, ETH("10"), 0),
      ).to.be.revertedWith("Earnest invalid");
      await expect(
        escrow.connect(owner).listForSale(1, ETH("10"), ETH("10")),
      ).to.be.revertedWith("Earnest invalid");
    });

    it("Cannot list a token the caller does not own", async () => {
      await propertyNFT.connect(owner).approve(escrow.address, 2);
      await propertyNFT.transferFrom(owner.address, seller.address, 2);
      await expect(
        escrow.connect(owner).listForSale(2, ETH("10"), ETH("1")),
      ).to.be.revertedWith("Only the NFT owner can list");
    });
  });

  describe("Close listing", () => {
    it("Lets the seller close a listing with no buyer", async () => {
      await listToken1();
      await expect(escrow.connect(owner).closeListing(1))
        .to.emit(escrow, "SaleCancelled")
        .withArgs(1, owner.address, 0);
      expect(await escrow.isOnOffer(1)).to.equal(false);
      expect(await escrow.activeSales()).to.equal(0);
    });

    it("Rejects a non-seller close", async () => {
      await listToken1();
      await expect(
        escrow.connect(buyer).closeListing(1),
      ).to.be.revertedWith("Only the seller");
    });
  });

  describe("Earnest deposit", () => {
    beforeEach(listToken1);

    it("Commits the first depositor as buyer", async () => {
      await expect(
        escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") }),
      )
        .to.emit(escrow, "EarnestDeposited")
        .withArgs(1, buyer.address, ETH("1"));
      expect((await escrow.sales(1)).buyer).to.equal(buyer.address);
      expect(await escrow.statusOf(1)).to.equal(2);
      expect(await escrow.fundedOf(1)).to.equal(ETH("1"));
    });

    it("Accepts earnest above the minimum against the balance", async () => {
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("3") });
      expect(await escrow.fundedOf(1)).to.equal(ETH("3"));
    });

    it("Rejects earnest below the required amount", async () => {
      await expect(
        escrow.connect(buyer).commitAndDeposit(1, { value: ETH("0.5") }),
      ).to.be.revertedWith("Earnest below required");
    });

    it("Rejects any payment above the full price", async () => {
      await expect(
        escrow.connect(buyer).commitAndDeposit(1, { value: ETH("11") }),
      ).to.be.revertedWith("Overfunded");
    });

    it("Rejects a second buyer once committed", async () => {
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await expect(
        escrow.connect(attacker).commitAndDeposit(1, { value: ETH("1") }),
      ).to.be.revertedWith("Wrong sale state");
    });
  });

  describe("Funding the balance", () => {
    beforeEach(async () => {
      await listToken1();
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
    });

    it("Tops up toward the full price", async () => {
      await expect(
        escrow.connect(buyer).fundBalance(1, { value: ETH("9") }),
      )
        .to.emit(escrow, "BalanceFunded")
        .withArgs(1, buyer.address, ETH("9"));
      expect(await escrow.fundedOf(1)).to.equal(ETH("10"));
    });

    it("Rejects overfunding beyond the price", async () => {
      await expect(
        escrow.connect(buyer).fundBalance(1, { value: ETH("9.5") }),
      ).to.be.revertedWith("Overfunded");
    });

    it("Rejects funding from anyone but the buyer", async () => {
      await escrow.connect(buyer).fundBalance(1, { value: ETH("5") });
      await expect(
        escrow.connect(attacker).fundBalance(1, { value: ETH("4") }),
      ).to.be.revertedWith("Only the buyer");
    });
  });

  describe("Approval", () => {
    beforeEach(async () => {
      await listToken1();
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
    });

    it("Only the buyer or the seller can approve", async () => {
      await expect(
        escrow.connect(attacker).approveSale(1),
      ).to.be.revertedWith("Only the buyer or the seller");
    });

    it("Moves to Approved once both sides approve", async () => {
      await escrow.connect(buyer).approveSale(1);
      expect(await escrow.statusOf(1)).to.equal(2);
      await escrow.connect(owner).approveSale(1);
      expect(await escrow.statusOf(1)).to.equal(3);
      expect((await escrow.sales(1)).buyerApproved).to.equal(true);
      expect((await escrow.sales(1)).sellerApproved).to.equal(true);
    });

    it("Reverts on disapprove to UnderContract", async () => {
      await escrow.connect(buyer).approveSale(1);
      await escrow.connect(owner).approveSale(1);
      await escrow.connect(buyer).disapproveSale(1);
      expect(await escrow.statusOf(1)).to.equal(2);
      expect((await escrow.sales(1)).buyerApproved).to.equal(false);
    });
  });

  describe("Finalize", () => {
    beforeEach(async () => {
      await listToken1();
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(buyer).fundBalance(1, { value: ETH("9") });
      await escrow.connect(buyer).approveSale(1);
      await escrow.connect(owner).approveSale(1);
    });

    it("Transfers the NFT to the buyer and pays the seller", async () => {
      const sellerBefore = await ethers.provider.getBalance(owner.address);
      const tx = await escrow.connect(owner).finalizeSale(1);
      const receipt = await tx.wait();
      const gasCost = receipt.gasUsed.mul(receipt.effectiveGasPrice);
      const sellerAfter = await ethers.provider.getBalance(owner.address);

      expect(await propertyNFT.ownerOf(1)).to.equal(buyer.address);
      expect(await escrow.isOnOffer(1)).to.equal(false);
      expect(await escrow.activeSales()).to.equal(0);
      expect(Number((await escrow.sales(1)).status)).to.equal(4);
      expect(sellerAfter.add(gasCost).sub(sellerBefore)).to.equal(ETH("10"));
    });

    it("Cannot finalize before both parties approve", async () => {
      await escrow.connect(buyer).disapproveSale(1);
      await expect(
        escrow.connect(owner).finalizeSale(1),
      ).to.be.revertedWith("Wrong sale state");
    });
  });

  describe("Finalize without full funding", () => {
    beforeEach(async () => {
      await listToken1();
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(buyer).approveSale(1);
      await escrow.connect(owner).approveSale(1);
    });

    it("Rejects a partially funded sale", async () => {
      await expect(
        escrow.connect(owner).finalizeSale(1),
      ).to.be.revertedWith("Sale is not fully funded");
    });
  });

  describe("Cancellation", () => {
    beforeEach(async () => {
      await listToken1();
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("2") });
      await escrow.connect(buyer).fundBalance(1, { value: ETH("3") });
    });

    it("Refunds every escrowed wei to the buyer", async () => {
      const buyerBefore = await ethers.provider.getBalance(buyer.address);
      const tx = await escrow.connect(buyer).cancelSale(1);
      const receipt = await tx.wait();
      const gasCost = receipt.gasUsed.mul(receipt.effectiveGasPrice);
      const buyerAfter = await ethers.provider.getBalance(buyer.address);

      expect(buyerAfter.add(gasCost).sub(buyerBefore)).to.equal(ETH("5"));
      expect(await escrow.isOnOffer(1)).to.equal(false);
      expect(await escrow.activeSales()).to.equal(0);
      expect(await propertyNFT.ownerOf(1)).to.equal(owner.address);
    });

    it("Allows either party to cancel", async () => {
      await escrow.connect(owner).cancelSale(1);
      expect(await escrow.isOnOffer(1)).to.equal(false);
    });

    it("Rejects cancellation by a stranger", async () => {
      await expect(
        escrow.connect(attacker).cancelSale(1),
      ).to.be.revertedWith("Only the seller or the buyer");
    });
  });

  describe("Registry sync", () => {
    it("Unlists an registry on-offer token when the sale settles", async () => {
      await propertyRegistry.connect(owner).list(1);
      expect(await propertyRegistry.isOnOffer(1)).to.equal(true);

      await propertyNFT.connect(owner).approve(escrow.address, 1);
      await escrow.connect(owner).listForSale(1, ETH("10"), ETH("1"));
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("10") });
      await escrow.connect(buyer).approveSale(1);
      await escrow.connect(owner).approveSale(1);
      await escrow.connect(owner).finalizeSale(1);

      expect(await propertyRegistry.isOnOffer(1)).to.equal(false);
    });
  });

  describe("Pause", () => {
    it("Blocks listings and recoveries while paused", async () => {
      await escrow.connect(owner).pause();
      await propertyNFT.connect(owner).approve(escrow.address, 1);
      await expect(
        escrow.connect(owner).listForSale(1, ETH("10"), ETH("1")),
      ).to.be.revertedWith("Pausable: paused");

      await escrow.connect(owner).unpause();
      await escrow.connect(owner).listForSale(1, ETH("10"), ETH("1"));
      expect(await escrow.isOnOffer(1)).to.equal(true);
    });

    it("Only the admin can pause", async () => {
      await expect(
        escrow.connect(buyer).pause(),
      ).to.be.revertedWith(
        `AccessControl: account ${buyer.address.toLowerCase()} is missing role ${ethers.constants.HashZero.toLowerCase()}`,
      );
    });
  });

  describe("Reentrancy", () => {
    it("Blocks a malicious contract seller from re-entering during payout", async () => {
      const MaliciousRecipient = await ethers.getContractFactory(
        "ReentrantAttacker",
      );
      const attackerContract = await MaliciousRecipient.deploy(escrow.address);

      await propertyNFT.connect(owner).mintProperty("MREID_0000003", "uri://3");
      await propertyNFT.connect(owner).approve(attackerContract.address, 3);
      await propertyNFT
        .connect(owner)
        .transferFrom(owner.address, attackerContract.address, 3);

      await attackerContract.setTokenId(3);
      await attackerContract.approveToken(propertyNFT.address, escrow.address, 3);
      await attackerContract.listTest(escrow.address, ETH("10"), ETH("1"));

      await escrow.connect(buyer).commitAndDeposit(3, { value: ETH("10") });
      await escrow.connect(buyer).approveSale(3);
      await attackerContract.approveSale(escrow.address);

      const attackerBefore = await ethers.provider.getBalance(
        attackerContract.address,
      );
      await attackerContract.finalize(escrow.address);
      const attackerAfter = await ethers.provider.getBalance(
        attackerContract.address,
      );

      // The NFT moved once, and the seller received exactly one payout (10 ETH).
      // The nested re-entering finalizeSale inside the malicious receive() was
      // stopped by the reentrancy guard; it could not steal a second payout.
      expect(await propertyNFT.ownerOf(3)).to.equal(buyer.address);
      expect(attackerAfter.sub(attackerBefore)).to.equal(ETH("10"));
      expect(Number((await escrow.sales(3)).status)).to.equal(4);
    });
  });

  describe("Event-based history reads", () => {
    it("Emits a SaleFinalized with the token and parties", async () => {
      await listToken1();
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(buyer).fundBalance(1, { value: ETH("9") });
      await escrow.connect(buyer).approveSale(1);
      await escrow.connect(owner).approveSale(1);
      await expect(escrow.connect(owner).finalizeSale(1))
        .to.emit(escrow, "SaleFinalized")
        .withArgs(1, buyer.address, owner.address, ETH("10"));
    });
  });
});

describe("MillowEscrow role-based workflow", () => {
  let admin, buyer, stranger, inspector, lender;
  let propertyNFT, propertyRegistry, escrow;
  let INSPECTOR_ROLE, LENDER_ROLE;

  beforeEach(async () => {
    [admin, buyer, stranger, , inspector, lender] = await ethers.getSigners();

    const PropertyNFT = await ethers.getContractFactory("PropertyNFT");
    propertyNFT = await PropertyNFT.deploy();

    const PropertyRegistry = await ethers.getContractFactory("PropertyRegistry");
    propertyRegistry = await PropertyRegistry.deploy(propertyNFT.address);

    const MillowEscrow = await ethers.getContractFactory("MillowEscrow");
    escrow = await MillowEscrow.deploy(
      propertyNFT.address,
      propertyRegistry.address,
    );

    INSPECTOR_ROLE = await escrow.INSPECTOR_ROLE();
    LENDER_ROLE = await escrow.LENDER_ROLE();
    await escrow.grantRole(INSPECTOR_ROLE, inspector.address);
    await escrow.grantRole(LENDER_ROLE, lender.address);

    await propertyNFT.connect(admin).mintProperty("MREID_0000001", "uri://1");
    await propertyNFT.connect(admin).mintProperty("MREID_0000002", "uri://2");
  });

  const approveOwnerToken1 = async () =>
    propertyNFT.connect(admin).approve(escrow.address, 1);

  const listPlain = async () => {
    await approveOwnerToken1();
    return escrow.connect(admin).listForSale(1, ETH("10"), ETH("1"));
  };

  const listExtended = async (inspection = true, lender = true, tokenId = 1) => {
    await propertyNFT.connect(admin).approve(escrow.address, tokenId);
    return escrow
      .connect(admin)
      .listForSaleWithRequirements(
        tokenId,
        ETH("10"),
        ETH("1"),
        inspection,
        lender,
      );
  };

  const toUnderContract = async () => {
    await listPlain();
    await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
    await escrow.connect(buyer).fundBalance(1, { value: ETH("9") });
    await escrow.connect(buyer).approveSale(1);
    await escrow.connect(admin).approveSale(1);
  };

  describe("Listing", () => {
    it("Keeps Payment Model A plain listings free of role requirements", async () => {
      await listPlain();
      const sale = await escrow.sales(1);
      expect(sale.inspectionRequired).to.equal(false);
      expect(sale.lenderRequired).to.equal(false);
      expect(sale.inspectionPassed).to.equal(false);
      expect(sale.financing.approved).to.equal(false);
    });

    it("Refuses listForSaleWithRequirements without at least one requirement", async () => {
      await approveOwnerToken1();
      await expect(
        escrow.connect(admin).listForSaleWithRequirements(1, ETH("10"), ETH("1"), false, false),
      ).to.be.revertedWith("No requirements set");
    });

    it("Records the requested inspection and lender requirements", async () => {
      const tx = await listExtended(true, true);
      await expect(tx)
        .to.emit(escrow, "SaleListed")
        .withArgs(1, admin.address, ETH("10"), ETH("1"));
      const sale = await escrow.sales(1);
      expect(sale.inspectionRequired).to.equal(true);
      expect(sale.lenderRequired).to.equal(true);
      expect(sale.inspectionPassed).to.equal(false);
      expect(sale.financing.approved).to.equal(false);
    });

    it("Applies the same owner and escrow-approval guards to extended listings", async () => {
      await expect(
        escrow.connect(buyer).listForSaleWithRequirements(2, ETH("10"), ETH("1"), true, false),
      ).to.be.revertedWith("Only the NFT owner can list");
      await expect(
        escrow.connect(admin).listForSaleWithRequirements(1, ETH("10"), ETH("1"), true, false),
      ).to.be.revertedWith("Escrow not approved");
    });
  });

  describe("Inspector gate", () => {
    beforeEach(async () => {
      await listExtended(true, false);
    });

    it("Lets only the assigned inspector record an inspection result", async () => {
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await expect(escrow.connect(buyer).setInspectionStatus(1, true)).to.be.revertedWith(
        `AccessControl: account ${buyer.address.toLowerCase()} is missing role ${INSPECTOR_ROLE.toLowerCase()}`,
      );
      await expect(
        escrow.connect(inspector).setInspectionStatus(1, true),
      )
        .to.emit(escrow, "InspectionUpdated")
        .withArgs(1, true);
      expect((await escrow.sales(1)).inspectionPassed).to.equal(true);
    });

    it("Flips a passed inspection back to failed on request", async () => {
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(inspector).setInspectionStatus(1, true);
      await escrow.connect(inspector).setInspectionStatus(1, false);
      expect((await escrow.sales(1)).inspectionPassed).to.equal(false);
    });

    it("Refuses inspection on a sale that did not opt in", async () => {
      await propertyNFT.connect(admin).approve(escrow.address, 2);
      await escrow.connect(admin).listForSale(2, ETH("10"), ETH("1"));
      await escrow.connect(buyer).commitAndDeposit(2, { value: ETH("1") });
      await expect(
        escrow.connect(inspector).setInspectionStatus(2, true),
      ).to.be.revertedWith("Inspection not required");
    });

    it("Refuses inspection before the buyer commits", async () => {
      await expect(
        escrow.connect(inspector).setInspectionStatus(1, true),
      ).to.be.revertedWith("Wrong sale state");
    });

    it("Requests the inspection when the buyer commits", async () => {
      await expect(
        escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") }),
      )
        .to.emit(escrow, "InspectionRequested")
        .withArgs(1);
    });
  });

  describe("Financing request and lender decision", () => {
    beforeEach(async () => {
      await listExtended(false, true);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
    });

    it("Only the buyer can request financing", async () => {
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      await expect(
        escrow.connect(stranger).requestFinancing(1, 2000, 850, 240),
      ).to.be.revertedWith("Only the buyer");
      expect((await escrow.sales(1)).financing.requested).to.equal(true);
    });

    it("Splits the price into the buyer's down payment and the loan", async () => {
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      const fin = (await escrow.sales(1)).financing;
      expect(fin.requested).to.equal(true);
      expect(fin.approved).to.equal(false);
      expect(fin.rejected).to.equal(false);
      expect(fin.downPaymentPctBps).to.equal(2000);
      expect(fin.interestRateBps).to.equal(850);
      expect(fin.loanTenureMonths).to.equal(240);
      expect(fin.downPaymentWei).to.equal(ETH("2"));
      expect(fin.loanAmountWei).to.equal(ETH("8"));
    });

    it("Rejects financing before inspection passes when inspection is required", async () => {
      await propertyNFT.connect(admin).approve(escrow.address, 2);
      await escrow
        .connect(admin)
        .listForSaleWithRequirements(2, ETH("10"), ETH("1"), true, true);
      await escrow.connect(buyer).commitAndDeposit(2, { value: ETH("1") });
      await expect(
        escrow.connect(buyer).requestFinancing(2, 2000, 850, 240),
      ).to.be.revertedWith("Inspection not passed");

      await escrow.connect(inspector).setInspectionStatus(2, true);
      await escrow.connect(buyer).requestFinancing(2, 2000, 850, 240);
      expect((await escrow.sales(2)).financing.requested).to.equal(true);
    });

    it("Rejects out-of-range or misaligned down payments", async () => {
      await expect(
        escrow.connect(buyer).requestFinancing(1, 500, 850, 240),
      ).to.be.revertedWith("Down payment invalid");
      await expect(
        escrow.connect(buyer).requestFinancing(1, 5500, 850, 240),
      ).to.be.revertedWith("Down payment invalid");
      await expect(
        escrow.connect(buyer).requestFinancing(1, 2025, 850, 240),
      ).to.be.revertedWith("Down payment invalid");
      await expect(
        escrow.connect(buyer).requestFinancing(1, 2000, 0, 240),
      ).to.be.revertedWith("Rate invalid");
      await expect(
        escrow.connect(buyer).requestFinancing(1, 2000, 850, 0),
      ).to.be.revertedWith("Tenure invalid");
    });

    it("Lets only the lender role approve and reject a request", async () => {
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      await expect(
        escrow.connect(buyer).approveFinancing(1),
      ).to.be.revertedWith(
        `AccessControl: account ${buyer.address.toLowerCase()} is missing role ${LENDER_ROLE.toLowerCase()}`,
      );
      await expect(
        escrow.connect(stranger).rejectFinancing(1),
      ).to.be.revertedWith(
        `AccessControl: account ${stranger.address.toLowerCase()} is missing role ${LENDER_ROLE.toLowerCase()}`,
      );
    });

    it("Approves a request and again after a rejection re-request", async () => {
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      await expect(escrow.connect(lender).rejectFinancing(1))
        .to.emit(escrow, "LoanRejected")
        .withArgs(1, lender.address);
      let fin = (await escrow.sales(1)).financing;
      expect(fin.approved).to.equal(false);
      expect(fin.rejected).to.equal(true);

      await escrow.connect(buyer).requestFinancing(1, 1500, 900, 120);
      fin = (await escrow.sales(1)).financing;
      expect(fin.rejected).to.equal(false);
      expect(fin.downPaymentPctBps).to.equal(1500);

      await expect(escrow.connect(lender).approveFinancing(1))
        .to.emit(escrow, "LoanApproved")
        .withArgs(1, lender.address, ETH("8.5"));
      fin = (await escrow.sales(1)).financing;
      expect(fin.approved).to.equal(true);
      expect(fin.lender).to.equal(lender.address);
    });
  });

  describe("Lender funding and buyer down payment", () => {
    beforeEach(async () => {
      await listExtended(false, true);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      await escrow.connect(buyer).approveBuyer(1);
      await escrow.connect(lender).approveFinancing(1);
    });

    it("Funds the approved loan into the escrow lender share", async () => {
      await expect(escrow.connect(lender).fundLoan(1, { value: ETH("8") }))
        .to.emit(escrow, "LoanFunded")
        .withArgs(1, lender.address, ETH("8"));
      expect(await escrow.lenderFundedOf(1)).to.equal(ETH("8"));
      expect(await escrow.fundedOf(1)).to.equal(ETH("9"));
    });

    it("Rejects loan funding beyond the approved amount", async () => {
      await escrow.connect(lender).fundLoan(1, { value: ETH("8") });
      await expect(
        escrow.connect(lender).fundLoan(1, { value: ETH("1") }),
      ).to.be.revertedWith("Loan overfunded");
      await expect(
        escrow.connect(lender).fundLoan(1, { value: ETH("9") }),
      ).to.be.revertedWith("Loan overfunded");
    });

    it("Rejects a second lender from routing funds into the escrow", async () => {
      await escrow.grantRole(LENDER_ROLE, stranger.address);
      await expect(
        escrow.connect(stranger).fundLoan(1, { value: ETH("5") }),
      ).to.be.revertedWith("Only the approving lender");
    });

    it("Lets the buyer pay the remainder of the down payment", async () => {
      await expect(
        escrow.connect(buyer).payDownPayment(1, { value: ETH("1") }),
      )
        .to.emit(escrow, "BuyerDownPaymentFunded")
        .withArgs(1, buyer.address, ETH("1"));
      expect(await escrow.buyerFundedOf(1)).to.equal(ETH("2"));
      expect(await escrow.fundedOf(1)).to.equal(ETH("2"));
    });

    it("Caps the buyer share at the approved down payment", async () => {
      await escrow.connect(buyer).payDownPayment(1, { value: ETH("1") });
      await expect(
        escrow.connect(buyer).payDownPayment(1, { value: ETH("1") }),
      ).to.be.revertedWith("Above down payment");
      await expect(
        escrow.connect(buyer).fundBalance(1, { value: ETH("1") }),
      ).to.be.revertedWith("Buyer funding above down payment");
    });
  });

  describe("Extended seller approval gating", () => {
    beforeEach(async () => {
      await listExtended(true, true);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(inspector).setInspectionStatus(1, true);
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
    });

    it("Blocks the seller until financing, funding and buyer approval all pass", async () => {
      await expect(
        escrow.connect(admin).approveSeller(1),
      ).to.be.revertedWith("Seller approval not yet allowed");

      await escrow.connect(lender).approveFinancing(1);
      await expect(
        escrow.connect(admin).approveSeller(1),
      ).to.be.revertedWith("Seller approval not yet allowed");

      await escrow.connect(buyer).approveBuyer(1);
      await expect(
        escrow.connect(admin).approveSeller(1),
      ).to.be.revertedWith("Seller approval not yet allowed");

      await escrow.connect(lender).fundLoan(1, { value: ETH("8") });
      await expect(
        escrow.connect(admin).approveSeller(1),
      ).to.be.revertedWith("Seller approval not yet allowed");

      await escrow.connect(buyer).payDownPayment(1, { value: ETH("1") });
      await expect(escrow.connect(admin).approveSeller(1))
        .to.emit(escrow, "SellerApproved")
        .withArgs(1, admin.address);
      expect(Number((await escrow.sales(1)).status)).to.equal(3);
    });

    it("Blocks the seller until inspection passes", async () => {
      await listExtended(true, true, 2);
      await escrow.connect(buyer).commitAndDeposit(2, { value: ETH("1") });
      await expect(
        escrow.connect(admin).approveSeller(2),
      ).to.be.revertedWith("Seller approval not yet allowed");

      await escrow.connect(inspector).setInspectionStatus(2, true);
      await expect(
        escrow.connect(admin).approveSeller(2),
      ).to.be.revertedWith("Seller approval not yet allowed");
    });

    it("Enforces the same gating through approveSale", async () => {
      await escrow.connect(inspector).setInspectionStatus(1, true);
      await escrow.connect(lender).approveFinancing(1);
      await escrow.connect(buyer).approveBuyer(1);
      await escrow.connect(lender).fundLoan(1, { value: ETH("8") });
      await expect(
        escrow.connect(admin).approveSale(1),
      ).to.be.revertedWith("Seller approval not yet allowed");

      await escrow.connect(buyer).payDownPayment(1, { value: ETH("1") });
      await escrow.connect(admin).approveSale(1);
      expect(Number((await escrow.sales(1)).status)).to.equal(3);
    });

    it("Rejects named approval functions for the wrong party", async () => {
      await escrow.connect(inspector).setInspectionStatus(1, true);
      await escrow.connect(lender).approveFinancing(1);
      await escrow.connect(lender).fundLoan(1, { value: ETH("8") });
      await escrow.connect(buyer).payDownPayment(1, { value: ETH("1") });

      await expect(escrow.connect(admin).approveBuyer(1)).to.be.revertedWith(
        "Only the buyer",
      );
      await expect(escrow.connect(buyer).approveSeller(1)).to.be.revertedWith(
        "Only the seller",
      );
    });

    it("Keeps Payment Model A approvals flexible in either order", async () => {
      await propertyNFT.connect(admin).approve(escrow.address, 2);
      await escrow.connect(admin).listForSale(2, ETH("10"), ETH("1"));
      await escrow.connect(buyer).commitAndDeposit(2, { value: ETH("1") });
      await escrow.connect(buyer).fundBalance(2, { value: ETH("9") });

      await escrow.connect(admin).approveSale(2);
      expect(Number((await escrow.sales(2)).status)).to.equal(2);
      await escrow.connect(buyer).approveSale(2);
      expect(Number((await escrow.sales(2)).status)).to.equal(3);
    });
  });

  describe("Extended finalize gates", () => {
    it("Settles a fully satisfied extended sale", async () => {
      await listExtended(true, true);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(inspector).setInspectionStatus(1, true);
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      await escrow.connect(lender).approveFinancing(1);
      await escrow.connect(lender).fundLoan(1, { value: ETH("8") });
      await escrow.connect(buyer).payDownPayment(1, { value: ETH("1") });
      await escrow.connect(buyer).approveBuyer(1);
      await escrow.connect(admin).approveSeller(1);

      await expect(escrow.connect(admin).finalizeSale(1))
        .to.emit(escrow, "SaleFinalized")
        .withArgs(1, buyer.address, admin.address, ETH("10"));
      expect(await propertyNFT.ownerOf(1)).to.equal(buyer.address);
      expect(await escrow.isOnOffer(1)).to.equal(false);
      expect((await ethers.provider.getBalance(escrow.address)).toString()).to.equal("0");
      expect(Number((await escrow.sales(1)).status)).to.equal(4);
    });

    it("Re-checks inspection independently even after the status reaches Approved", async () => {
      await listExtended(true, false);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(buyer).fundBalance(1, { value: ETH("9") });
      await escrow.connect(buyer).approveBuyer(1);
      await escrow.connect(inspector).setInspectionStatus(1, true);
      await escrow.connect(admin).approveSeller(1);
      expect(Number((await escrow.sales(1)).status)).to.equal(3);

      await escrow.connect(inspector).setInspectionStatus(1, false);
      await expect(
        escrow.connect(admin).finalizeSale(1),
      ).to.be.revertedWith("Inspection not passed");
    });

    it("Blocks finalize once an approval is revoked from the Approved state", async () => {
      await listExtended(false, true);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      await escrow.connect(lender).approveFinancing(1);
      await escrow.connect(lender).fundLoan(1, { value: ETH("8") });
      await escrow.connect(buyer).payDownPayment(1, { value: ETH("1") });
      await escrow.connect(buyer).approveBuyer(1);
      await escrow.connect(admin).approveSeller(1);

      await escrow.connect(buyer).disapproveSale(1);
      await expect(
        escrow.connect(admin).finalizeSale(1),
      ).to.be.revertedWith("Wrong sale state");
    });

    it("Refuses financing funds on a rejected request", async () => {
      await listExtended(false, true);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      await escrow.connect(lender).rejectFinancing(1);
      await expect(
        escrow.connect(lender).fundLoan(1, { value: ETH("8") }),
      ).to.be.revertedWith("Loan not approved");
      await expect(
        escrow.connect(buyer).payDownPayment(1, { value: ETH("1") }),
      ).to.be.revertedWith("Financing not approved");
    });
  });

  describe("Extended cancellation refunds", () => {
    it("Refunds the buyer share and the lender disbursed loan separately", async () => {
      await listExtended(false, true);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(buyer).requestFinancing(1, 2000, 850, 240);
      await escrow.connect(lender).approveFinancing(1);
      await escrow.connect(lender).fundLoan(1, { value: ETH("8") });
      await escrow.connect(buyer).payDownPayment(1, { value: ETH("1") });

      const buyerBefore = await ethers.provider.getBalance(buyer.address);
      const lenderBefore = await ethers.provider.getBalance(lender.address);
      const tx = await escrow.connect(buyer).cancelSale(1);
      const receipt = await tx.wait();
      const gasCost = receipt.gasUsed.mul(receipt.effectiveGasPrice);
      const buyerAfter = await ethers.provider.getBalance(buyer.address);
      const lenderAfter = await ethers.provider.getBalance(lender.address);

      expect(buyerAfter.add(gasCost).sub(buyerBefore)).to.equal(ETH("2"));
      expect(lenderAfter.sub(lenderBefore)).to.equal(ETH("8"));
      expect(await escrow.isOnOffer(1)).to.equal(false);
      expect(await propertyNFT.ownerOf(1)).to.equal(admin.address);
    });

    it("Allows either party to cancel an extended sale", async () => {
      await listExtended(false, true);
      await escrow.connect(buyer).commitAndDeposit(1, { value: ETH("1") });
      await escrow.connect(admin).cancelSale(1);
      expect(await escrow.isOnOffer(1)).to.equal(false);
    });
  });
});