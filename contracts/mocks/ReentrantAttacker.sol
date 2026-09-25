//SPDX-License-Identifier: Unlicense
pragma solidity ^0.8.17;

contract ReentrantAttacker {
    address public target;
    uint256 public tokenId;
    bool public malicious = true;

    constructor(address target_) {
        target = target_;
    }

    function setTokenId(uint256 id_) external {
        tokenId = id_;
    }

    function setMalicious(bool malicious_) external {
        malicious = malicious_;
    }

    function approveToken(
        address nft_,
        address operator_,
        uint256 id_
    ) external {
        (bool ok, ) = nft_.call(
            abi.encodeWithSignature(
                "approve(address,uint256)",
                operator_,
                id_
            )
        );
        require(ok, "approve failed");
    }

    function listTest(address escrow_, uint256 price_, uint256 earnest_) external {
        (bool ok, ) = escrow_.call(
            abi.encodeWithSignature(
                "listForSale(uint256,uint256,uint256)",
                tokenId,
                price_,
                earnest_
            )
        );
        require(ok, "list failed");
    }

    function commit(address escrow_) external payable {
        (bool ok, ) = escrow_.call{value: msg.value}(
            abi.encodeWithSignature("commitAndDeposit(uint256)", tokenId)
        );
        require(ok, "commit failed");
    }

    function fund(address escrow_) external payable {
        (bool ok, ) = escrow_.call{value: msg.value}(
            abi.encodeWithSignature("fundBalance(uint256)", tokenId)
        );
        require(ok, "fund failed");
    }

    function approveSale(address escrow_) external {
        (bool ok, ) = escrow_.call(
            abi.encodeWithSignature("approveSale(uint256)", tokenId)
        );
        require(ok, "approve failed");
    }

    function finalize(address escrow_) external {
        (bool ok, ) = escrow_.call(
            abi.encodeWithSignature("finalizeSale(uint256)", tokenId)
        );
        require(ok, "finalize failed");
    }

    // When a payout is pushed to this contract it tries to re-enter the escrow
    // (finalizeSale).  In a safe escrow this nested call must be rejected so
    // the seller can only ever receive a single payout.
    receive() external payable {
        if (malicious && tokenId != 0) {
            (bool ok, ) = address(msg.sender).call(
                abi.encodeWithSignature("finalizeSale(uint256)", tokenId)
            );
            ok;
        }
    }
}