//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.17;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract PropertyNFT is ERC721, ERC721URIStorage, AccessControl {
    using Counters for Counters.Counter;

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");

    Counters.Counter private _tokenIds;
    mapping(uint256 => string) private _propertyIds;
    mapping(bytes32 => uint256) private _tokenByProperty;

    event PropertyMinted(
        uint256 indexed tokenId,
        string indexed propertyId,
        address indexed minter
    );

    constructor() ERC721("MILLOW Property", "MILLOW") {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(MINTER_ROLE, msg.sender);
    }

    modifier validPropertyId(string memory propertyId_) {
        require(bytes(propertyId_).length > 0, "Property id is required");
        require(
            _tokenByProperty[keccak256(bytes(propertyId_))] == 0,
            "Property already registered"
        );
        _;
    }

    function mintProperty(string memory propertyId_, string memory tokenURI_)
        external
        onlyRole(MINTER_ROLE)
        validPropertyId(propertyId_)
        returns (uint256)
    {
        _tokenIds.increment();
        uint256 tokenId = _tokenIds.current();

        _propertyIds[tokenId] = propertyId_;
        _tokenByProperty[keccak256(bytes(propertyId_))] = tokenId;

        _safeMint(msg.sender, tokenId);
        _setTokenURI(tokenId, tokenURI_);

        emit PropertyMinted(tokenId, propertyId_, msg.sender);
        return tokenId;
    }

    function setTokenURI(uint256 tokenId_, string memory tokenURI_)
        external
        onlyRole(MINTER_ROLE)
    {
        require(_exists(tokenId_), "Token does not exist");
        _setTokenURI(tokenId_, tokenURI_);
    }

    function tokenByProperty(string memory propertyId_)
        public
        view
        returns (uint256)
    {
        return _tokenByProperty[keccak256(bytes(propertyId_))];
    }

    function propertyOf(uint256 tokenId_) public view returns (string memory) {
        require(_exists(tokenId_), "Token does not exist");
        return _propertyIds[tokenId_];
    }

    function totalSupply() public view returns (uint256) {
        return _tokenIds.current();
    }

    function tokenURI(uint256 tokenId_)
        public
        view
        override(ERC721, ERC721URIStorage)
        returns (string memory)
    {
        return super.tokenURI(tokenId_);
    }

    function _burn(uint256 tokenId_)
        internal
        override(ERC721, ERC721URIStorage)
    {
        super._burn(tokenId_);
    }

    function supportsInterface(bytes4 interfaceId_)
        public
        view
        override(ERC721, AccessControl)
        returns (bool)
    {
        return super.supportsInterface(interfaceId_);
    }
}