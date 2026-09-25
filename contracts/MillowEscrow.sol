//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

interface IPropertyRegistry {
    function isOnOffer(uint256 tokenId_) external view returns (bool);

    function unlist(uint256 tokenId_) external;
}

contract MillowEscrow is Pausable, ReentrancyGuard, AccessControl {
    enum Status {
        None,
        Listed,
        UnderContract,
        Approved,
        Finalized,
        Cancelled
    }

    bytes32 public constant INSPECTOR_ROLE = keccak256("INSPECTOR_ROLE");
    bytes32 public constant LENDER_ROLE = keccak256("LENDER_ROLE");

    // Down-payment options a buyer may select when requesting financing,
    // expressed in basis points: 1000 = 10%, 1500 = 15%, ..., 5000 = 50%.
    uint16 public constant MIN_DOWN_PAYMENT_BPS = 1000;
    uint16 public constant MAX_DOWN_PAYMENT_BPS = 5000;
    uint16 public constant ALLOWED_DOWN_PAYMENT_STEP_BPS = 500;

    IERC721 public immutable propertyNFT;
    IPropertyRegistry public immutable propertyRegistry;

    struct FinancingTerms {
        address lender;
        bool requested;
        bool approved;
        bool rejected;
        // Down payment as a percentage in basis points (e.g. 2000 = 20%).
        uint16 downPaymentPctBps;
        // Annual interest rate in basis points (e.g. 850 = 8.5% p.a.).
        uint256 interestRateBps;
        // Loan tenure in months (e.g. 240 = 20 years).
        uint16 loanTenureMonths;
        // downPaymentWei + loanAmountWei == priceWei by construction.
        uint256 downPaymentWei;
        uint256 loanAmountWei;
    }

    struct Sale {
        Status status;
        address seller;
        address buyer;
        uint256 priceWei;
        uint256 earnestWei;
        // Escrow is split by contributor: the buyer funds only the down
        // payment / earnest share while the lender funds the approved loan.
        uint256 buyerFundedWei;
        uint256 lenderFundedWei;
        bool sellerApproved;
        bool buyerApproved;
        // Optional role-based requirements.  A listing created through the
        // plain listForSale() keeps every flag false, preserving the Phase 8
        // Buyer/Seller Payment Model A unchanged.  Only listings created via
        // listForSaleWithRequirements() activate the Inspector/Lender flow.
        bool inspectionPassed;
        bool inspectionRequired;
        bool lenderRequired;
        FinancingTerms financing;
    }

    mapping(uint256 => Sale) public sales;
    mapping(uint256 => bool) public isListed;
    uint256 public activeSales;

    event SaleListed(
        uint256 indexed tokenId,
        address indexed seller,
        uint256 priceWei,
        uint256 earnestWei
    );
    event EarnestDeposited(
        uint256 indexed tokenId,
        address indexed buyer,
        uint256 amountWei
    );
    event BalanceFunded(
        uint256 indexed tokenId,
        address indexed buyer,
        uint256 amountWei
    );
    event SaleApproved(uint256 indexed tokenId, address indexed approver);
    event SaleDisapproved(uint256 indexed tokenId, address indexed approver);
    event SaleFinalized(
        uint256 indexed tokenId,
        address indexed buyer,
        address indexed seller,
        uint256 priceWei
    );
    event SaleCancelled(
        uint256 indexed tokenId,
        address indexed seller,
        uint256 refundWei
    );
    event InspectionRequested(uint256 indexed tokenId);
    event InspectionUpdated(uint256 indexed tokenId, bool passed);
    event FinancingRequested(
        uint256 indexed tokenId,
        address indexed buyer,
        uint16 downPaymentPctBps,
        uint256 interestRateBps,
        uint16 loanTenureMonths,
        uint256 downPaymentWei,
        uint256 loanAmountWei
    );
    event LoanApproved(
        uint256 indexed tokenId,
        address indexed lender,
        uint256 loanAmountWei
    );
    event LoanRejected(uint256 indexed tokenId, address indexed lender);
    event BuyerDownPaymentFunded(
        uint256 indexed tokenId,
        address indexed buyer,
        uint256 amountWei
    );
    event LoanFunded(
        uint256 indexed tokenId,
        address indexed lender,
        uint256 amountWei
    );
    event BuyerApproved(uint256 indexed tokenId, address indexed buyer);
    event SellerApproved(uint256 indexed tokenId, address indexed seller);

    constructor(address propertyNFT_, address propertyRegistry_) {
        require(propertyNFT_ != address(0), "NFT address required");
        propertyNFT = IERC721(propertyNFT_);
        propertyRegistry = IPropertyRegistry(propertyRegistry_);
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
    }

    modifier activeSale(uint256 tokenId_) {
        require(isListed[tokenId_], "Not listed");
        _;
    }

    modifier onlySaleParty(uint256 tokenId_, Status expected_) {
        require(sales[tokenId_].status == expected_, "Wrong sale state");
        _;
    }

    modifier saleInProgress(uint256 tokenId_) {
        require(
            sales[tokenId_].status == Status.UnderContract ||
                sales[tokenId_].status == Status.Approved,
            "Wrong sale state"
        );
        _;
    }

    // Seller offers a token for sale.  The NFT stays with the seller; the
    // escrow must already be an approved operator so the transfer can settle at
    // finalize time.  Prices are plain wei amounts determined by the caller
    // (the prototype converts an INR asking price to test ETH off-chain).
    //
    // This is the Phase 8 Payment Model A path: a plain Buyer/Seller sale with
    // no Inspector or Lender requirements, where both parties may approve in
    // either order and the buyer funds the full purchase price.
    // listForSaleWithRequirements() is the only way to opt into the extended
    // role-based workflow.
    function listForSale(
        uint256 tokenId_,
        uint256 priceWei_,
        uint256 earnestWei_
    ) external whenNotPaused nonReentrant returns (uint256) {
        _listSale(tokenId_, priceWei_, earnestWei_, false, false);
        return tokenId_;
    }

    // Seller offers a token with optional Inspector verification and / or
    // Lender financing gating the sale.  Until the selected requirements are
    // satisfied the sale cannot settle: the successful buyer must also pass
    // inspection (when required), obtain an approved loan, and the escrow must
    // be fully funded by the buyer's down payment and the lender's disbursed
    // loan before the seller may approve.
    function listForSaleWithRequirements(
        uint256 tokenId_,
        uint256 priceWei_,
        uint256 earnestWei_,
        bool inspectionRequired_,
        bool lenderRequired_
    ) external whenNotPaused nonReentrant returns (uint256) {
        require(
            inspectionRequired_ || lenderRequired_,
            "No requirements set"
        );
        _listSale(tokenId_, priceWei_, earnestWei_, inspectionRequired_, lenderRequired_);
        return tokenId_;
    }

    function _listSale(
        uint256 tokenId_,
        uint256 priceWei_,
        uint256 earnestWei_,
        bool inspectionRequired_,
        bool lenderRequired_
    ) private {
        require(
            propertyNFT.ownerOf(tokenId_) == msg.sender,
            "Only the NFT owner can list"
        );
        require(!isListed[tokenId_], "Already listed");
        require(
            propertyNFT.getApproved(tokenId_) == address(this),
            "Escrow not approved"
        );
        require(priceWei_ > 0, "Price required");
        require(earnestWei_ > 0 && earnestWei_ < priceWei_, "Earnest invalid");
        require(
            !lenderRequired_ || earnestWei_ > 0,
            "Earnest required for financing"
        );

        sales[tokenId_] = Sale(
            Status.Listed,
            payable(msg.sender),
            address(0),
            priceWei_,
            earnestWei_,
            0,
            0,
            false,
            false,
            false,
            inspectionRequired_,
            lenderRequired_,
            FinancingTerms(
                address(0),
                false,
                false,
                false,
                0,
                0,
                0,
                0,
                0
            )
        );
        isListed[tokenId_] = true;
        activeSales += 1;

        emit SaleListed(tokenId_, msg.sender, priceWei_, earnestWei_);
    }

    // Seller removes a listing that no buyer has committed to.
    function closeListing(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        onlySaleParty(tokenId_, Status.Listed)
    {
        require(sales[tokenId_].seller == msg.sender, "Only the seller");

        _clearSale(tokenId_);
        emit SaleCancelled(tokenId_, msg.sender, 0);
    }

    // Buyer commits by depositing at least the required earnest.  Any excess
    // counts toward the buyer's funded share.  The buyer is committed at this
    // point.  On an extended listing that requires inspection, committing the
    // sale is what requests the inspection.
    function commitAndDeposit(uint256 tokenId_)
        external
        payable
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        onlySaleParty(tokenId_, Status.Listed)
    {
        require(msg.value > 0, "Deposit required");
        require(msg.value >= sales[tokenId_].earnestWei, "Earnest below required");

        sales[tokenId_].buyer = msg.sender;
        sales[tokenId_].status = Status.UnderContract;
        _fundBuyer(tokenId_, msg.value);

        emit EarnestDeposited(tokenId_, msg.sender, msg.value);
        if (sales[tokenId_].inspectionRequired) {
            emit InspectionRequested(tokenId_);
        }
    }

    // Buyer tops up the escrow.  On a Payment Model A sale the buyer may fund
    // the full price.  On an extended sale the buyer's share is capped at the
    // approved down payment once financing is approved.
    function fundBalance(uint256 tokenId_)
        external
        payable
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
    {
        require(msg.value > 0, "Funding amount required");
        require(msg.sender == sales[tokenId_].buyer, "Only the buyer");
        _fundBuyer(tokenId_, msg.value);

        emit BalanceFunded(tokenId_, msg.sender, msg.value);
    }

    // Payment Model A approval.  Both parties may approve in either order and
    // either side may withdraw its approval.  On an extended sale the seller
    // approval is additionally gated, so approveSale() can only move the
    // seller's flag once the extended conditions are satisfied.
    function approveSale(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
    {
        _recordApproval(tokenId_, msg.sender, true);

        emit SaleApproved(tokenId_, msg.sender);
    }

    function disapproveSale(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
    {
        _recordApproval(tokenId_, msg.sender, false);

        emit SaleDisapproved(tokenId_, msg.sender);
    }

    // Explicit buyer approval (extended workflow).  The seller may be the only
    // party whose approval is gated.
    // Explicit buyer approval (extended workflow).  The seller may be the only
    // party whose approval is gated.
    function approveBuyer(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
    {
        require(sales[tokenId_].buyer == msg.sender, "Only the buyer");
        _recordApproval(tokenId_, msg.sender, true);

        emit BuyerApproved(tokenId_, msg.sender);
    }

    // Explicit seller approval (extended workflow).  Emits SellerApproved and
    // enforces the strict extended gating; the approveSale() path enforces the
    // same gating internally so the two cannot diverge.
    function approveSeller(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
    {
        require(sales[tokenId_].seller == msg.sender, "Only the seller");
        _recordApproval(tokenId_, msg.sender, true);

        emit SellerApproved(tokenId_, msg.sender);
    }

    // Inspector records the result of a verification on a sale that opted in
    // via listForSaleWithRequirements().  Only acts as a gate: it never holds
    // or transfers funds.
    function setInspectionStatus(uint256 tokenId_, bool passed_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        onlyRole(INSPECTOR_ROLE)
        saleInProgress(tokenId_)
    {
        require(
            sales[tokenId_].inspectionRequired,
            "Inspection not required"
        );
        sales[tokenId_].inspectionPassed = passed_;
        emit InspectionUpdated(tokenId_, passed_);
    }

    // Buyer elects their down-payment share (10%..50% of the price in 5%
    // steps) plus the financing terms they are asking the lender for.  Only
    // possible on a lender-required sale, only after inspection has passed
    // (when required), and only before the loan has been disbursed.
    function requestFinancing(
        uint256 tokenId_,
        uint16 downPaymentPctBps_,
        uint256 interestRateBps_,
        uint16 loanTenureMonths_
    )
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
    {
        Sale storage sale = sales[tokenId_];
        require(sale.lenderRequired, "Lender not required");
        require(sale.buyer == msg.sender, "Only the buyer");
        require(sale.lenderFundedWei == 0, "Loan already funded");
        require(
            !sale.inspectionRequired || sale.inspectionPassed,
            "Inspection not passed"
        );
        require(
            downPaymentPctBps_ >= MIN_DOWN_PAYMENT_BPS &&
                downPaymentPctBps_ <= MAX_DOWN_PAYMENT_BPS &&
                downPaymentPctBps_ % ALLOWED_DOWN_PAYMENT_STEP_BPS == 0,
            "Down payment invalid"
        );
        require(interestRateBps_ > 0 && interestRateBps_ <= 5000, "Rate invalid");
        require(loanTenureMonths_ > 0, "Tenure invalid");

        uint256 downPaymentWei_ = (sale.priceWei * downPaymentPctBps_) / 10000;
        uint256 loanAmountWei_ = sale.priceWei - downPaymentWei_;

        sale.financing = FinancingTerms(
            address(0),
            true,
            false,
            false,
            downPaymentPctBps_,
            interestRateBps_,
            loanTenureMonths_,
            downPaymentWei_,
            loanAmountWei_
        );

        emit FinancingRequested(
            tokenId_,
            msg.sender,
            downPaymentPctBps_,
            interestRateBps_,
            loanTenureMonths_,
            downPaymentWei_,
            loanAmountWei_
        );
    }

    // Lender accepts the buyer's financing request, binding the loan amount.
    function approveFinancing(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
        onlyRole(LENDER_ROLE)
    {
        Sale storage sale = sales[tokenId_];
        require(sale.lenderRequired, "Lender not required");
        require(sale.financing.requested, "No financing request");
        require(sale.buyer != address(0), "No buyer");
        // Buyer must still be committed and the committed amount must never
        // exceed the requested down payment once a loan is approved.
        require(
            sale.buyerFundedWei <= sale.financing.downPaymentWei,
            "Buyer funded above down payment"
        );

        sale.financing.approved = true;
        sale.financing.rejected = false;
        sale.financing.lender = msg.sender;

        emit LoanApproved(tokenId_, msg.sender, sale.financing.loanAmountWei);
    }

    // Lender rejects the buyer's financing request.  The buyer may adjust the
    // terms and re-request at any point before a loan is disbursed.
    function rejectFinancing(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
        onlyRole(LENDER_ROLE)
    {
        Sale storage sale = sales[tokenId_];
        require(sale.lenderRequired, "Lender not required");
        require(sale.financing.requested, "No financing request");
        require(sale.lenderFundedWei == 0, "Loan already funded");

        sale.financing.approved = false;
        sale.financing.rejected = true;
        sale.financing.lender = address(0);

        emit LoanRejected(tokenId_, msg.sender);
    }

    // Lender disburses part or all of the approved loan into the escrow.
    // Funding may not exceed the approved loan amount, must stay within the
    // sale price, and cannot bypass the existing buyer share.
    function fundLoan(uint256 tokenId_)
        external
        payable
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
        onlyRole(LENDER_ROLE)
    {
        Sale storage sale = sales[tokenId_];
        require(sale.lenderRequired, "Lender not required");
        require(sale.financing.approved, "Loan not approved");
        require(msg.sender == sale.financing.lender, "Only the approving lender");
        require(msg.value > 0, "Loan funding required");
        require(
            sale.lenderFundedWei + msg.value <= sale.financing.loanAmountWei,
            "Loan overfunded"
        );
        require(
            sale.lenderFundedWei + sale.buyerFundedWei + msg.value <= sale.priceWei,
            "Overfunded"
        );

        sale.lenderFundedWei += msg.value;

        emit LoanFunded(tokenId_, msg.sender, msg.value);
    }

    // Buyer completes their committed share up to the approved down payment.
    // The earnest already deposited counts towards the down payment.
    function payDownPayment(uint256 tokenId_)
        external
        payable
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
    {
        Sale storage sale = sales[tokenId_];
        require(sale.buyer == msg.sender, "Only the buyer");
        require(sale.financing.approved, "Financing not approved");
        require(msg.value > 0, "Payment required");
        require(
            sale.buyerFundedWei + msg.value <= sale.financing.downPaymentWei,
            "Above down payment"
        );
        require(
            sale.buyerFundedWei + sale.lenderFundedWei + msg.value <= sale.priceWei,
            "Overfunded"
        );

        sale.buyerFundedWei += msg.value;

        emit BuyerDownPaymentFunded(tokenId_, msg.sender, msg.value);
    }

    // Finalize: the seller must have approved the escrow as an operator, both
    // parties must have approved, and the escrow must be fully funded by the
    // buyer and the lender combined.  The NFT moves to the buyer and the exact
    // price is paid to the seller.  State flips first (checks-effects-
    // interactions).  Every required condition is re-checked here regardless of
    // transaction type: inspection result when required, approved and fully
    // disbursed loan when required, and full funding.
    function finalizeSale(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        onlySaleParty(tokenId_, Status.Approved)
    {
        Sale memory sale = sales[tokenId_];
        require(sale.sellerApproved && sale.buyerApproved, "Not all approved");
        require(
            sale.buyerFundedWei + sale.lenderFundedWei == sale.priceWei,
            "Sale is not fully funded"
        );
        if (sale.inspectionRequired) {
            require(sale.inspectionPassed, "Inspection not passed");
        }
        if (sale.lenderRequired) {
            require(sale.financing.approved, "Lender not approved");
            require(
                sale.lenderFundedWei == sale.financing.loanAmountWei,
                "Loan not fully funded"
            );
        }

        sale.status = Status.Finalized;
        sales[tokenId_] = sale;
        isListed[tokenId_] = false;
        if (activeSales > 0) {
            activeSales -= 1;
        }

        emit SaleFinalized(tokenId_, sale.buyer, sale.seller, sale.priceWei);

        _finishListing(tokenId_, sale.buyer, sale.seller, sale.priceWei);
    }

    // Cancel an open sale before it settles.  The buyer gets back their full
    // escrowed share (earnest and any down payment) and the lender gets back
    // any disbursed loan funds.  The token stays with the seller.  State
    // clears before any payout.
    function cancelSale(uint256 tokenId_)
        external
        whenNotPaused
        nonReentrant
        activeSale(tokenId_)
        saleInProgress(tokenId_)
    {
        Sale memory sale = sales[tokenId_];
        require(
            msg.sender == sale.seller || msg.sender == sale.buyer,
            "Only the seller or the buyer"
        );

        uint256 buyerRefund = sale.buyerFundedWei;
        uint256 lenderRefund = sale.lenderFundedWei;
        address lender = sale.financing.lender;
        _clearSale(tokenId_);
        emit SaleCancelled(tokenId_, sale.seller, buyerRefund + lenderRefund);

        if (buyerRefund > 0) {
            (bool buyerOk, ) = payable(sale.buyer).call{value: buyerRefund}("");
            require(buyerOk, "Refund to buyer failed");
        }
        if (lenderRefund > 0) {
            (bool lenderOk, ) = payable(lender).call{value: lenderRefund}("");
            require(lenderOk, "Refund to lender failed");
        }
    }

    // Internal: apply a buyer contribution.  On an extended sale with an
    // approved loan the buyer's share is capped at the approved down payment;
    // on a Payment Model A sale the buyer may fund the full price.
    function _fundBuyer(uint256 tokenId_, uint256 amount_) private {
        Sale storage sale = sales[tokenId_];
        uint256 buyerCap = sale.priceWei;
        bool cappedByDownPayment = false;
        if (sale.financing.approved) {
            buyerCap = sale.financing.downPaymentWei;
            cappedByDownPayment = true;
        }
        require(
            sale.buyerFundedWei + amount_ <= buyerCap,
            cappedByDownPayment
                ? "Buyer funding above down payment"
                : "Overfunded"
        );
        require(
            sale.buyerFundedWei + sale.lenderFundedWei + amount_ <= sale.priceWei,
            "Overfunded"
        );
        sale.buyerFundedWei += amount_;
    }

    // Internal: record an approval from the buyer or the seller only.  On an
    // extended sale (inspection or lender required) the seller's approval is
    // strictly gated: inspection must have passed, the loan must be approved
    // and fully disbursed, the escrow fully funded, and the buyer already
    // approved.  Payment Model A approvals stay flexible (either order).
    function _requireSellerCanApprove(Sale storage sale_, string memory err_)
        private
        view
    {
        if (!sale_.inspectionRequired && !sale_.lenderRequired) {
            return;
        }
        if (sale_.inspectionRequired) {
            require(sale_.inspectionPassed, err_);
        }
        if (sale_.lenderRequired) {
            require(sale_.financing.approved, err_);
            require(
                sale_.lenderFundedWei == sale_.financing.loanAmountWei,
                err_
            );
        }
        require(sale_.buyerApproved, err_);
        require(
            sale_.buyerFundedWei + sale_.lenderFundedWei == sale_.priceWei,
            err_
        );
    }

    function _recordApproval(uint256 tokenId_, address approver_, bool value_) private {
        Sale storage sale = sales[tokenId_];
        require(
            approver_ == sale.buyer || approver_ == sale.seller,
            "Only the buyer or the seller"
        );
        if (approver_ == sale.seller && value_) {
            _requireSellerCanApprove(sale, "Seller approval not yet allowed");
        }
        if (approver_ == sale.buyer) {
            sale.buyerApproved = value_;
        } else {
            sale.sellerApproved = value_;
        }
        if (sale.buyerApproved && sale.sellerApproved) {
            sale.status = Status.Approved;
        } else {
            sale.status = Status.UnderContract;
        }
    }

    // Internal: clear sale state, keep the registry in sync when available.
    function _clearSale(uint256 tokenId_) private {
        delete sales[tokenId_];
        isListed[tokenId_] = false;
        if (activeSales > 0) {
            activeSales -= 1;
        }
        _unlistFromRegistry(tokenId_);
    }

    // Internal: transfer the NFT and pay the seller once the sale settled.
    // State is already final, and ReentrancyGuard blocks any re-entry.
    function _finishListing(
        uint256 tokenId_,
        address buyer_,
        address seller_,
        uint256 price_
    ) private {
        IERC721(propertyNFT).transferFrom(seller_, buyer_, tokenId_);

        (bool success, ) = payable(seller_).call{value: price_}("");
        require(success, "Transfer to seller failed");

        _unlistFromRegistry(tokenId_);
    }

    // Internal: keep the Phase 7 registry "on offer" flag in sync when the
    // escrow holds the LISTING_MANAGER_ROLE.
    function _unlistFromRegistry(uint256 tokenId_) private {
        if (
            address(propertyRegistry) != address(0) &&
            propertyRegistry.isOnOffer(tokenId_)
        ) {
            propertyRegistry.unlist(tokenId_);
        }
    }

    // Views used by the read-only frontend helpers.

    function isOnOffer(uint256 tokenId_) external view returns (bool) {
        return isListed[tokenId_];
    }

    function statusOf(uint256 tokenId_)
        external
        view
        returns (uint8)
    {
        if (!isListed[tokenId_]) return uint8(Status.None);
        return uint8(sales[tokenId_].status);
    }

    function priceOf(uint256 tokenId_) external view returns (uint256) {
        return sales[tokenId_].priceWei;
    }

    function earnestOf(uint256 tokenId_) external view returns (uint256) {
        return sales[tokenId_].earnestWei;
    }

    function fundedOf(uint256 tokenId_) external view returns (uint256) {
        return
            sales[tokenId_].buyerFundedWei +
            sales[tokenId_].lenderFundedWei;
    }

    function buyerFundedOf(uint256 tokenId_) external view returns (uint256) {
        return sales[tokenId_].buyerFundedWei;
    }

    function lenderFundedOf(uint256 tokenId_) external view returns (uint256) {
        return sales[tokenId_].lenderFundedWei;
    }

    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
}