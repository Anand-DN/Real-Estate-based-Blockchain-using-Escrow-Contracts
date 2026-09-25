# MILLOW - Real Estate NFT DApp

MILLOW is a real-estate DApp where each property is represented by an on-chain
token identified by an `MREID_xxxxxxx` identifier. Ownership, listing and escrow
settlement live in Solidity contracts, the UI is a React app, and valuation /
risk / recommendation insights are served by Python FastAPI services.

## Technology Stack & Tools

- Solidity 0.8.17 (smart contracts and tests)
- [Hardhat](https://hardhat.org/) (development network, compilation, tests, deployment)
- [Ethers.js v5](https://docs.ethers.io/v5/) (contract interaction)
- [React.js 18](https://reactjs.org/) (frontend)
- Python + FastAPI (AI valuation API and AI assistant)
- XGBoost / scikit-learn (valuation, fraud and recommendation models)

## Contracts

| Contract | Purpose |
| --- | --- |
| `PropertyNFT` | ERC-721 ownership token for each MREID property |
| `PropertyRegistry` | Listing status and on/off-offer state per token |
| `MillowEscrow` | Payment-in-ETH escrow with inspection, approval and settlement |

Legacy demo contracts (`RealEstate`, `Escrow`) are also present and are used by
the optional legacy deployment script in step 7b.

## Requirements

- [Node.js](https://nodejs.org/en/) 18 or newer (includes `npm`)
- [Python](https://www.python.org/) 3.9 or newer
- Git
- Optional: [Ollama](https://ollama.com/) running locally as the AI fallback provider

## Setting Up

### 1. Clone/Download the Repository

```bash
git clone https://github.com/Anand-DN/Real-Estate-based-Blockchain-using-Escrow-Contracts.git
cd Real-Estate-based-Blockchain-using-Escrow-Contracts
```

### 2. Install JavaScript Dependencies

```bash
npm install
```

### 3. Install Python Dependencies

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Then install the Python packages used by the AI and valuation services:

```bash
pip install -r ai/requirements.txt
```

### 4. Run Tests

Contract and application tests (Hardhat):

```bash
npx hardhat test
```

Frontend tests:

```bash
npm test
```

### 5. Start the Hardhat Node

Leave this running in **Terminal 1**:

```bash
npx hardhat node
```

### 6. Compile Contracts

In **Terminal 2**:

```bash
npx hardhat compile
```

### 7. Run the Deployment Script

In **Terminal 2** (while the Hardhat node is running):

```bash
npx hardhat run ./scripts/deployMillow.js --network localhost
```

This deploys `PropertyNFT`, `PropertyRegistry` and `MillowEscrow`, grants the
demo roles, and writes the deployed addresses into `src/config.json` under the
active chain id (`31337`).

### 7b. Legacy Demo Deployment (optional)

The original 24-property demo (`RealEstate` + `Escrow`) is still available:

```bash
npx hardhat run ./scripts/deploy.js --network localhost
```

### 8. Seed Property Tokens (optional)

Mint MREID tokens on the local node. A quick four-token run for development:

```bash
# Windows (PowerShell)
$env:MILLOW_SEED_LIMIT='4'; npx hardhat run scripts/tokenizeAllMreid.js --network localhost

# macOS / Linux
MILLOW_SEED_LIMIT=4 npx hardhat run scripts/tokenizeAllMreid.js --network localhost
```

The full dataset (29,135 properties) takes a long time to mint:

```bash
npx hardhat run scripts/tokenizeAllMreid.js --network localhost
```

Optional tuning via environment variables: `MILLOW_SEED_LIMIT` (0 = all),
`MILLOW_SEED_BATCH` (default 500), `MILLOW_SEED_CONCURRENCY` (default 8).

Refresh the frontend/API chain index after seeding:

```bash
npx hardhat run scripts/exportChainIndex.js --network localhost
```

### 9. Start the Valuation API (backend)

```bash
npm run valuation
```

Equivalent: `python backend/app.py` (or `uvicorn backend.app:app --host 0.0.0.0 --port 8001`).
Serves the FastAPI valuation/dashboard API on `http://localhost:8001`.

### 10. Run the AI Service

Copy the example environment file and add a key (Groq primary, Ollama fallback):

```bash
# Windows (PowerShell)
Copy-Item ai/.env.example ai/.env

# macOS / Linux
cp ai/.env.example ai/.env
```

Then start the AI assistant service:

```bash
npm run ai
```

Equivalent: `python ai/app.py`. Serves on `http://localhost:8000`. On first
start it loads or trains its lightweight price/fraud models.

### 11. Start the Metadata Server (optional)

```bash
npm run server
```

Serves property metadata on `http://localhost:3001`.

### 12. Start the Frontend

```bash
npm run start
```

Equivalent: `npm start`. Opens the React app on `http://localhost:3000`.

## Quick Command Summary

| Purpose | Command |
| --- | --- |
| Install JS dependencies | `npm install` |
| Install Python dependencies | `pip install -r ai/requirements.txt` |
| Run Hardhat tests | `npx hardhat test` |
| Run frontend tests | `npm test` |
| Start local chain | `npx hardhat node` |
| Compile contracts | `npx hardhat compile` |
| Deploy MILLOW contracts | `npx hardhat run ./scripts/deployMillow.js --network localhost` |
| Deploy legacy demo | `npx hardhat run ./scripts/deploy.js --network localhost` |
| Seed property tokens | `npx hardhat run scripts/tokenizeAllMreid.js --network localhost` |
| Export chain index | `npx hardhat run scripts/exportChainIndex.js --network localhost` |
| Start valuation API | `npm run valuation` |
| Start AI service | `npm run ai` |
| Start metadata server | `npm run server` |
| Start frontend | `npm run start` |
| Production frontend build | `npm run build` |

## Notes

- Keep `npx hardhat node` running in its own terminal while deploying, seeding
  and using the app against the local network.
- `src/config.json` is written automatically by the deployment scripts; the
  frontend reads contract addresses from it for the active chain id.
- Local demo accounts (Hardhat default mnemonic) map to roles as follows:
  account 0 = Buyer, account 1 = Seller, account 2 = Inspector,
  account 3 = Lender. Import the Hardhat test accounts into MetaMask using
  the default test mnemonic when testing the full workflow.
- `.env` files are intentionally not committed. Provide your own API keys in
  `ai/.env` (template: `ai/.env.example`).
- Reset local chain state by stopping and restarting `npx hardhat node`, then
  re-running the deployment script.
