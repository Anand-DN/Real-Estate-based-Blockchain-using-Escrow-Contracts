# MILLOW — Real Estate NFT DApp

MILLOW is a real-estate DApp where every property is an on-chain token identified
by an `MREID_xxxxxxx` id. Ownership, listing and escrow settlement live in
Solidity contracts, the UI is React, and valuation / risk / recommendation
insights are served by Python FastAPI services.

## Technology Stack & Tools

- Solidity 0.8.17 (writing smart contracts & tests)
- Javascript (React & Testing)
- [Hardhat](https://hardhat.org/) (compilation, tests, deployment scripts)
- [Foundry Anvil](https://book.getfoundry.sh/anvil/) (persistent local chain)
- [Ethers.js v5](https://docs.ethers.io/v5/) (blockchain interaction)
- [React.js 18](https://reactjs.org/) (frontend framework)
- Python + FastAPI, XGBoost / scikit-learn (AI valuation & assistant)

## Contracts

| Contract | Purpose |
| --- | --- |
| `PropertyNFT` | ERC-721 ownership token for each MREID property |
| `PropertyRegistry` | Listing status and on/off-offer state per token |
| `MillowEscrow` | Payment-in-ETH escrow with inspection, approval and settlement |

Legacy demo contracts (`RealEstate`, `Escrow`) are still present and are used
only by the optional legacy deployment in step 8b.

## Requirements For Initial Setup

- [Node.js](https://nodejs.org/en/) 18 or newer (includes `npm`)
- [Python](https://www.python.org/) 3.9 or newer
- Git
- [Foundry](https://getfoundry.sh) — **Anvil 1.8.3 specifically** (see step 4)
- Optional: [Ollama](https://ollama.com/) running locally as the AI fallback

## Setting Up

### 1. Clone/Download the Repository

```bash
git clone https://github.com/Anand-DN/Real-Estate-based-Blockchain-using-Escrow-Contracts.git
cd Real-Estate-based-Blockchain-using-Escrow-Contracts
```

### 2. Install Dependencies

```bash
npm install
```

Python dependencies (valuation API + AI service):

```bash
python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r ai/requirements.txt
```

### 3. Run tests

```bash
npx hardhat test      # contract tests
npm test              # frontend tests
```

### 4. Start the local chain

The catalogue lives on a **persistent** Anvil chain whose state is saved to
`.chain/state.json`, so it survives restarts and crashes.

```bash
anvil --version        # MUST print 1.8.3
npm run chain:start
npm run chain:status
```

Expected status: `chainId 31337`, `totalSupply 29135`, and all three contracts
present. Other chain commands:

```bash
npm run chain:stop        # shut the node down
npm run chain:verify      # read-only audit (-- --full checks all 29,135)
```

> **Do not use `npx hardhat node` for this project.** Hardhat Network keeps
> chain state in memory only, so restarting it wipes the whole catalogue and the
> app falls back to "escrow not deployed". Hardhat is still used for `compile`,
> `test` and `run --network localhost` — just never as the node.
>
> If `anvil` is not on `PATH`, point the tooling at it once per terminal:
> `$env:MILLOW_ANVIL = "C:\foundry\bin\anvil.exe"` (PowerShell).

### 5. Deploy the contracts

In a separate terminal, while the chain is running:

```bash
npx hardhat run ./scripts/deployMillow.js --network localhost
```

Deploys `PropertyNFT`, `PropertyRegistry` and `MillowEscrow`, grants the demo
roles, and writes the addresses into `src/config.json` under chain `31337`.
Confirm the addresses did not move:

```bash
git diff -- src/config.json     # MUST be empty
```

Run this **once**, on an empty chain. Re-running it on a deployed chain shifts
the contracts off the addresses in `src/config.json`.

### 6. Tokenize properties (seed the catalogue)

Skip if the chain already reports `totalSupply 29135` — it is fully seeded.

```bash
npm run chain:tokenize        # all 29,135 in canonical token order, resumable
```

Quick subset for development:

```bash
# Windows (PowerShell)
$env:MILLOW_SEED_LIMIT='4'; npx hardhat run scripts/tokenizeAllMreid.js --network localhost

# macOS / Linux
MILLOW_SEED_LIMIT=4 npx hardhat run scripts/tokenizeAllMreid.js --network localhost
```

Tuning: `MILLOW_SEED_LIMIT` (`0` = all), `MILLOW_SEED_BATCH` (default 500),
`MILLOW_SEED_CONCURRENCY` (default 8), `MILLOW_TOKENIZE_BATCH` (default 100).

Refresh the catalogue index the frontend and dashboard read (read-only, view
calls only):

```bash
npm run chain:index
```

Other seed helpers:

```bash
npx hardhat run scripts/mintMreid.js --network localhost    # small seed set, lists first two tokens
npx hardhat run scripts/verifySeed.js --network localhost   # prints tokenByProperty for MREID_0000001/2
npm run chain:test                                          # offline resume-planner tests
```

> Token ids are **not** the MREID number: 1,396 of the 29,135 tokens are
> intentionally permuted relative to dataset order. The canonical order is
> `data/processed/millow_token_map.csv`; regenerate it with
> `node scripts/buildTokenMap.js`.

### 7. Start the Metadata Server

```bash
npm run server
```

Serves property metadata on `http://localhost:3001`.

### 8. Start the Valuation API

```bash
npm run valuation
```

Equivalent: `python backend/app.py`. Serves `http://localhost:8001`.

### 9. Start the AI Service

```bash
# Windows (PowerShell)
Copy-Item ai/.env.example ai/.env

# macOS / Linux
cp ai/.env.example ai/.env
```

Add your key (Groq primary, Ollama fallback), then:

```bash
npm run ai
```

Equivalent: `python ai/app.py`. Serves `http://localhost:8000` and loads or
trains its price/fraud models on first start.

### 10. Start the Frontend

```bash
npm run start
```

Opens `http://localhost:3000`.

In MetaMask add a network named **Millow Localhost**: RPC
`http://127.0.0.1:8545`, chain ID `31337`, currency ETH. Import the default test
mnemonic `test test test test test test test test test test test junk`.

### 11. Legacy Demo Deployment (optional)

The original 24-property demo (`RealEstate` + `Escrow`):

```bash
npx hardhat run ./scripts/deploy.js --network localhost
```

It overwrites the contract at the canonical `PropertyNFT` address, so never run
it against the persistent catalogue chain.

## Quick Command Summary

| Purpose | Command |
| --- | --- |
| Install JS dependencies | `npm install` |
| Install Python dependencies | `pip install -r ai/requirements.txt` |
| Run Hardhat tests | `npx hardhat test` |
| Run frontend tests | `npm test` |
| Start / stop local chain | `npm run chain:start` / `npm run chain:stop` |
| Chain health | `npm run chain:status` |
| Compile contracts | `npx hardhat compile` |
| Deploy MILLOW contracts | `npx hardhat run ./scripts/deployMillow.js --network localhost` |
| Tokenize all properties | `npm run chain:tokenize` |
| Export chain index | `npm run chain:index` |
| Audit the chain | `npm run chain:verify` |
| Export / import portable chain state | `npm run chain:export` / `npm run chain:import` |
| Start metadata server | `npm run server` |
| Start valuation API | `npm run valuation` |
| Start AI service | `npm run ai` |
| Start frontend | `npm run start` |
| Production frontend build | `npm run build` |

## Notes

- Keep the chain running in its own terminal while deploying, tokenizing and
  using the app.
- `src/config.json` is written by the deployment script; the frontend reads
  contract addresses from it for the active chain id.
- Demo roles (default test accounts): account 0 = Buyer, 1 = Seller,
  2 = Inspector, 3 = Lender.
- `.env` files are not committed. Provide your own API keys in `ai/.env`
  (template: `ai/.env.example`).
- Reset local chain state by stopping the node, deleting `.chain/state.json` and
  re-running steps 4–6.
- Moving the chain to another machine: `npm run chain:export`, then
  `npm run chain:import -- --artifact=<file>.json.gz` on the target — 0
  deployments, 0 mints. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
  for the full Computer 1 → Computer 2 workflow and the determinism guarantees.
