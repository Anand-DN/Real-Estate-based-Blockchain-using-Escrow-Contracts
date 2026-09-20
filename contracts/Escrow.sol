//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.0;

interface IERC721 {
    function ownerOf(uint256 _id) external view returns (address);

    function transferFrom(
        address _from,
        address _to,
        uint256 _id
    ) external;
}

contract Escrow {
    address public nftAddress;
    mapping(uint256 => address payable) public sellerOf;
    address public inspector;
    address public lender;

    event Listed(uint256 indexed nftID, address indexed seller, uint256 purchasePrice);
    event EarnestDeposited(uint256 indexed nftID, address indexed buyer);
    event InspectionUpdated(uint256 indexed nftID, bool passed);
    event SaleApproved(uint256 indexed nftID, address indexed approver);
    event SaleDisapproved(uint256 indexed nftID, address indexed approver);
    event SaleFinalized(uint256 indexed nftID, address indexed buyer, address indexed seller);
    event SaleCancelled(uint256 indexed nftID, address indexed seller);

    modifier onlySeller(uint256 _nftID) {
        require(msg.sender == sellerOf[_nftID], "Only seller can call this method");
        _;
    }

    modifier onlyInspector() {
        require(msg.sender == inspector, "Only inspector can call this method");
        _;
    }

    modifier onlyParticipant(uint256 _nftID) {
        require(
            msg.sender == buyer[_nftID] ||
                msg.sender == sellerOf[_nftID] ||
                msg.sender == inspector ||
                msg.sender == lender,
            "Only a participant can call this method"
        );
        _;
    }

    mapping(uint256 => bool) public isListed;
    mapping(uint256 => uint256) public purchasePrice;
    mapping(uint256 => uint256) public escrowAmount;
    mapping(uint256 => address) public buyer;
    mapping(uint256 => bool) public inspectionPassed;
    mapping(uint256 => mapping(address => bool)) public approval;

    // Per-property fund accounting.  Funds for every listing are tracked
    // separately so that finalizing or cancelling one sale can never touch
    // the earnest money or lender funds of another listing.
    mapping(uint256 => uint256) public earnestPaid;
    mapping(uint256 => uint256) public lenderFunded;

    constructor(
        address _nftAddress,
        address _inspector,
        address _lender
    ) {
        nftAddress = _nftAddress;
        inspector = _inspector;
        lender = _lender;
    }

    function list(
        uint256 _nftID,
        uint256 _purchasePrice,
        uint256 _escrowAmount
    ) public {
        require(
            IERC721(nftAddress).ownerOf(_nftID) == msg.sender,
            "Only the NFT owner can list"
        );
        require(!isListed[_nftID], "Property already listed");

        // Transfer NFT from seller to this contract
        IERC721(nftAddress).transferFrom(msg.sender, address(this), _nftID);

        isListed[_nftID] = true;
        purchasePrice[_nftID] = _purchasePrice;
        escrowAmount[_nftID] = _escrowAmount;
        buyer[_nftID] = address(0);
        sellerOf[_nftID] = payable(msg.sender);
        earnestPaid[_nftID] = 0;
        lenderFunded[_nftID] = 0;
        approval[_nftID][msg.sender] = false;
        approval[_nftID][lender] = false;

        emit Listed(_nftID, msg.sender, _purchasePrice);
    }

    // Put Under Contract (first depositor commits as the buyer - payable escrow)
    function depositEarnest(uint256 _nftID) public payable {
        require(isListed[_nftID], "Property is not listed");
        require(msg.value > 0, "Earnest deposit required");
        require(
            earnestPaid[_nftID] + msg.value <= escrowAmount[_nftID],
            "Earnest is already fully funded"
        );

        if (buyer[_nftID] == address(0)) {
            buyer[_nftID] = msg.sender;
        } else {
            require(msg.sender == buyer[_nftID], "Only buyer can call this method");
        }

        earnestPaid[_nftID] += msg.value;

        emit EarnestDeposited(_nftID, buyer[_nftID]);
    }

    // Lender funds the balance of the purchase price for a specific sale.
    // The money is attributed to this NFT so it can be refunded on cancel.
    function fundSale(uint256 _nftID) public payable {
        require(isListed[_nftID], "Property is not listed");
        require(msg.sender == lender, "Only lender can fund this sale");
        require(msg.value > 0, "Funding amount required");
        require(
            lenderFunded[_nftID] + msg.value <=
                purchasePrice[_nftID] - escrowAmount[_nftID],
            "Sale is already fully funded"
        );

        lenderFunded[_nftID] += msg.value;

        emit SaleApproved(_nftID, msg.sender);
    }

    // Update Inspection Status (only inspector)
    function updateInspectionStatus(uint256 _nftID, bool _passed)
        public
        onlyInspector
    {
        inspectionPassed[_nftID] = _passed;

        emit InspectionUpdated(_nftID, _passed);
    }

    // Approve Sale
    function approveSale(uint256 _nftID) public {
        approval[_nftID][msg.sender] = true;

        emit SaleApproved(_nftID, msg.sender);
    }

    // Disapprove Sale (revoke your own approval)
    function disapproveSale(uint256 _nftID) public {
        approval[_nftID][msg.sender] = false;

        emit SaleDisapproved(_nftID, msg.sender);
    }

    // Finalize Sale
    // -> Require inspection status (add more items here, like appraisal)
    // -> Require sale to be authorized
    // -> Require funds to be correct amount
    // -> Transfer NFT to buyer
    // -> Transfer Funds to Seller
    function finalizeSale(uint256 _nftID) public {
        require(isListed[_nftID], "Property is not listed");
        require(inspectionPassed[_nftID], "Inspection has not passed");
        require(buyer[_nftID] != address(0), "No buyer committed");
        require(approval[_nftID][buyer[_nftID]], "Buyer has not approved");
        require(approval[_nftID][sellerOf[_nftID]], "Seller has not approved");
        require(approval[_nftID][lender], "Lender has not approved");
        require(
            earnestPaid[_nftID] == escrowAmount[_nftID],
            "Earnest is not fully deposited"
        );
        require(
            lenderFunded[_nftID] == purchasePrice[_nftID] - escrowAmount[_nftID],
            "Lender has not fully funded the sale"
        );

        isListed[_nftID] = false;

        // Pay the seller exactly this property's purchase price, never the
        // contract-wide balance.
        uint256 payout = purchasePrice[_nftID];
        require(address(this).balance >= payout, "Contract is short on funds");
        (bool success, ) = sellerOf[_nftID].call{value: payout}("");
        require(success, "Transfer to seller failed");

        IERC721(nftAddress).transferFrom(address(this), buyer[_nftID], _nftID);

        emit SaleFinalized(_nftID, buyer[_nftID], sellerOf[_nftID]);
    }

    // Cancel Sale (handle earnest deposit)
    // -> if inspection status is not approved, refund the earnest to the buyer,
    //    otherwise the earnest is forfeited to the seller
    // -> the lender always gets their funds back
    // -> return the property to the seller and reopen the listing
    function cancelSale(uint256 _nftID) public onlyParticipant(_nftID) {
        require(isListed[_nftID], "Property is not listed");

        address oldBuyer = buyer[_nftID];
        address payable seller = sellerOf[_nftID];
        uint256 earnest = earnestPaid[_nftID];
        uint256 lent = lenderFunded[_nftID];

        if (inspectionPassed[_nftID]) {
            // Earnest is forfeited to the seller.
            if (earnest > 0) {
                require(
                    address(this).balance >= earnest,
                    "Contract is short on funds"
                );
                seller.transfer(earnest);
            }
        } else if (oldBuyer != address(0) && earnest > 0) {
            // Refund the earnest to the buyer (nothing was ever committed to
            // the zero address).
            require(
                address(this).balance >= earnest,
                "Contract is short on funds"
            );
            payable(oldBuyer).transfer(earnest);
        }

        // The lender always recovers the principal on a cancelled sale.
        if (lent > 0) {
            require(address(this).balance >= lent, "Contract is short on funds");
            payable(lender).transfer(lent);
        }

        isListed[_nftID] = false;
        inspectionPassed[_nftID] = false;
        buyer[_nftID] = address(0);
        earnestPaid[_nftID] = 0;
        lenderFunded[_nftID] = 0;
        approval[_nftID][oldBuyer] = false;
        approval[_nftID][seller] = false;
        approval[_nftID][lender] = false;

        IERC721(nftAddress).transferFrom(address(this), seller, _nftID);

        emit SaleCancelled(_nftID, seller);
    }

    receive() external payable {}

    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
