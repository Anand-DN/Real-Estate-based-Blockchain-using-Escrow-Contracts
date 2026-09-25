# MILLOW — Phase 8 Report: Escrow Sale Workflow (Payment Model A)

## 1. Summary

Phase 8 delivers the **pay-in-full escrow sale workflow** between a seller and a buyer
for an MREID-recorded property on the Hardhat local chain (chainId **31337**). A
`MillowEscrow` contract escrows the buyer's test ETH, holds the property NFT, and
disburses the full price to the seller only after both parties approve. The React
frontend gains a wallet-gated **Transaction Panel** and a **Sale History** list, both
fully tested.

## 2. Scope / Design decision

- **Payment Model A (approved):** buyer commits earnest money plus the full asking
  price into escrow before approval; the seller releases the NFT only after the buyer
  is fully funded; on finalize the NFT is transferred to the buyer and the full price
  is paid to the seller.
- INR amounts come from the MREID dataset. A clearly-labelled **demonstration rate**
  (`demoRate.inrPerEth = 100000`, i.e. ₹1,00,000 = 1 test ETH) is used **only** in the
  frontend/seed layer to compute wei. The chain only ever sees wei.
- No real money: test ETH on Hardhat chain 31337; UI carries required disclaimer copy.

## 3. Software

- `contracts/MillowEscrow.sol` — escrow with `AccessControl` (`DEFAULT_ADMIN_ROLE`,
  granted `LISTING_MANAGER_ROLE` on `PropertyRegistry`), OpenZeppelin
  `ReentrancyGuard` + `nonReentrant` on `listForSale`, `cancelSale`,
  `commitAndDeposit`, `approveSale`, `finalizeSale`, `refundDeposit`. Sale state machine:
  `Listed -> UnderContract -> Approved -> Finalized` (plus `Cancelled`). Emits
  `SaleFinalized`, `Funded`, `SaleCancelled` events; performs refunds on cancel.
- `contracts/mocks/ReentrantAttacker.sol` — adversarial contract that attempts
  re-entrancy during `finalizeSale`. Proves the escrow pays out exactly once.

## 4. On-chain test results

- `npx hardhat test` — **95 passing**, 0 failing (includes multi-buyer race,
  under/over-funding, cancellation & refund, seller double-withdrawal, reentrancy).

## 5. Frontend test results

- `npm test -- --watchAll=false` (CI) — **9 suites / 36 tests passing**, 0 failing.
- `npm run build` — **exit code 0**.

### Added/tested components

- `TransactionPanel.js` — wallet-gated panel. Shows seller "Close listing", buyer
  "Commit & deposit earnest", "Fund balance", "Approve sale", "Cancel & refund", and
  finalize/results once `finalizable` (status Approved + both approvals + fully funded).
- `SaleHistory.js` — lists `SaleListed`, `SaleCommitted`, `SaleFunded`, `SaleApproved`,
  `SaleFinalized`, `SaleCancelled` events for a property, newest first.

## 6. Backend regression

- Backend API tests (FastAPI TestClient scripts) — **284 PASS, 0 FAIL**
  (279 baseline + 5 Phase 8 additions).

## 7. Live on-chain walkthrough (chain 31337)

Property **MREID_0000001**, token **1** — asking ₹30,000,000 @ ₹1,00,000/test-ETH ⇒
**priceWei = 300 test ETH**, earnest **30 test ETH** (10%).

Deployed MILLOW contracts (via `scripts/deployMillow.js`):

| Contract | Address |
| --- | --- |
| PropertyNFT | `0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9` |
| PropertyRegistry | `0x5FC8d32690cc91D4c39d9d3abcBD16989F875707` |
| MillowEscrow | `0x0165878A594ca255338adfa4d48449f69242Eb8F` |

Representative transaction hashes:

| Step | Tx hash |
| --- | --- |
| List in escrow | `0x78826a5a84f63e4e850b58a343c532534592d216ead2f4c1637ee29430de3a44` |
| Commit & deposit (30 ETH) | `0x64f03cb105007953316ddfdb3aeb5e9007e0bb2365089894532c8c9231a04d54` |
| Fund balance (270 ETH) | sent during walkthrough (escrow reached 300 ETH) |
| Buyer approve | `0xc7f2e482472f546ebba55c04b0fe9c41c46eaf6891495cadc1dfba97290d980c` |
| Seller approve | `0x234d7081af074242e48c120e644965b4490a4fcf5fd5e1ce6d6271c338bf9968` |
| Finalize sale | `0x1bc69729f42e086bdc0a94e4363acc150b59dea695aecef0696c4a907d4dd408` |

### Verification after finalize — all PASS

- Status: `Finalized`
- NFT token 1 owner is now the **buyer** (`0x7099…79C8`)
- `PropertyRegistry.isOnOffer(token 1)` = **false**
- Seller receives **300 test ETH** payout
- Escrow contract balance = **0 wei** (fully disbursed, reentrancy-guarded)

## 8. Files added/changed in Phase 8

- `contracts/MillowEscrow.sol`, `contracts/mocks/ReentrantAttacker.sol`
- `test/MillowEscrow.js` (incl. reentrancy + full sale lifecycle)
- `src/abis/MillowEscrow.json` (ABI export to frontend)
- `src/components/TransactionPanel.js`, `src/components/TransactionPanel.test.js`
- `src/components/SaleHistory.js`, `src/components/SaleHistory.test.js`
- `src/components/PropertyDetail.js` (mounts TransactionPanel + SaleHistory)
- `src/lib/demoRate.js` (INR→wei demo conversion), `src/lib/blockchain.js`,
  `src/lib/wallet.js`, `src/format.js`
- `src/config.json` (chain 31337 addresses + `demoRate.inrPerEth`)
- `scripts/deployMillow.js`, `scripts/mintMreid.js`, `scripts/configureSale.js`,
  `scripts/demoFullSale.js` (idempotent full-sale driver)
- `src/index.css` (panel/history styling)

## 9. Legal / disclaimer copy (required on UI)

- Prototype on Hardhat test chain using test ETH; no real money.
- INR → test-ETH conversion is a demo-only rate overwritten each deploy.
- Blockchain records are a technical proof-of-concept.
- Legal ownership remains subject to applicable Indian property and registration law.