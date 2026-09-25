//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721.sol";

contract PropertyRegistry is Pausable, ReentrancyGuard, AccessControl {
    bytes32 public constant LISTING_MANAGER_ROLE =
        keccak256("LISTING_MANAGER_ROLE");

    IERC721 public immutable propertyNFT;

    struct Listing {
        bool active;
        address seller;
        uint256 listedAt;
    }

    mapping(uint256 => Listing) public listings;
    uint256 public activeListings;

    event Listed(
        uint256 indexed tokenId,
        address indexed seller,
        uint256 listedAt
    );
    event Unlisted(uint256 indexed tokenId, address indexed actor);

    constructor(address propertyNFT_) {
        require(propertyNFT_ != address(0), "NFT address required");
        propertyNFT = IERC721(propertyNFT_);
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(LISTING_MANAGER_ROLE, msg.sender);
    }

    function list(uint256 tokenId_) external whenNotPaused nonReentrant {
        require(
            propertyNFT.ownerOf(tokenId_) == msg.sender,
            "Only the NFT owner can list"
        );
        require(!listings[tokenId_].active, "Already on offer");

        listings[tokenId_] = Listing({
            active: true,
            seller: msg.sender,
            listedAt: block.timestamp
        });
        activeListings += 1;

        emit Listed(tokenId_, msg.sender, block.timestamp);
    }

    function unlist(uint256 tokenId_) external whenNotPaused nonReentrant {
        require(listings[tokenId_].active, "Not on offer");
        require(
            listings[tokenId_].seller == msg.sender ||
                hasRole(LISTING_MANAGER_ROLE, msg.sender),
            "Only the seller or a listing manager"
        );

        delete listings[tokenId_];
        activeListings -= 1;

        emit Unlisted(tokenId_, msg.sender);
    }

    function isOnOffer(uint256 tokenId_) external view returns (bool) {
        return listings[tokenId_].active;
    }

    function sellerOf(uint256 tokenId_) external view returns (address) {
        return listings[tokenId_].seller;
    }

    function pause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _pause();
    }

    function unpause() external onlyRole(DEFAULT_ADMIN_ROLE) {
        _unpause();
    }
}