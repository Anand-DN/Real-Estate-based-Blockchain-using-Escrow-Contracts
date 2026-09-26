# MILLOW persistent chain — reproducibility & Computer 1 → Computer 2 hand-off

The MILLOW catalogue (29,135 MREID properties) lives on a **persistent local
Ethereum-compatible chain** so it survives node restarts, process crashes and
computer reboots — and can be moved to another machine **without re-running a
single mint transaction**.

## Why Anvil and not Hardhat

Hardhat Network v2 keeps chain state **in memory only**. There is no state
export or import (`anvil_dumpState` and `hardhat_setAutomine` are both absent;
`hardhat_reset` exists and is destructive). Any Hardhat chain dies with the
process, which is exactly how the previous 29,135-token chain was lost.

| | Anvil 1.8.3 | Ganache 7 | Hardhat 2.22.15 |
|---|---|---|---|
| Single-file state export | `--state` (dumps on exit + interval) / `anvil_dumpState` | none | none |
| State import | `--state` (loads if present) | none | none |
| Artifact carries block env | yes | yes (LevelDB dir) | n/a |
| Portable to another machine | yes (JSON) | fragile | no |
| Default accounts match Hardhat | yes | needs explicit mnemonic | yes |
| 58k-transaction throughput | fast | slow | moderate |

Ganache's only persistence is `--database.dbPath`, a LevelDB **directory** with
no export API — not a distributable artifact.

## Determinism guarantees

Anvil's default mnemonic is the same well-known test phrase Hardhat uses, so
accounts resolve to identical addresses. Deployment from account 0 with a fixed
nonce sequence therefore reproduces the contract addresses byte-for-byte:

| Contract | Address | Code |
|---|---|---|
| `PropertyNFT` | `0x5FbDB2315678afecb367f032d93F642f64180aa3` | 7,818 B |
| `PropertyRegistry` | `0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512` | 3,934 B |
| `MillowEscrow` | `0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0` | 17,716 B |

`src/config.json` already carries exactly these addresses, so the frontend and
backend need **no change** to reconnect.

### Token ids are NOT the MREID number

`PropertyNFT.mintProperty` assigns token ids from an incrementing counter in
**execution order**. The original catalogue contains **1,396 token ids that are
intentionally permuted** relative to dataset order (token 6 = `MREID_0000008`,
token 7 = `MREID_0000006`, token 8 = `MREID_0000007`, …).

Never assume `tokenId == numeric suffix of MREID`. The canonical order lives in:

```
data/processed/millow_token_map.csv     # token_id,mreid_id  (29,135 rows)
sha256: 91fcb8a3346d89a91360ce531b685e2e97683b7f80f2ac2c44dedf83042106f4
```

Regenerate it from a chain snapshot with `node scripts/buildTokenMap.js`.

### Deterministic batching

Token ids follow **nonce order**, not submission order. `tokenizeChain.js`
disables automine (`evm_setAutomine`), submits a batch with consecutive nonces,
then seals it with a single `evm_mine`. Verified: a deliberately shuffled batch
still produced ids in exact nonce sequence. 100 mints/block ≈ 17.5M gas, under
the 30M block limit.

> Anvil has **no** `evm_getAutomine` (that is Hardhat-only). Use `evm_setAutomine`.

### Account count

Anvil defaults to **10 accounts**. The `MINTER_ROLE` holder is account index 10
(`0xBcd4042DE499D14e55001CcbB24a551F3b954096`), so **`--accounts 20` is
mandatory**. Omitting it makes `restoreChain`/tokenizing fail.

## State artifact format

- `anvil_dumpState` returns **hex-encoded gzip**.
- `--state` requires **plain JSON**, not gzip.

So the pipeline is: dump → gunzip → JSON → gzip for transport → gunzip on the
target → write `.chain/state.json` → start Anvil. The target needs the
decompressed size in scratch space.

**`--state` is the only flag the node needs.** In Anvil 1.8.3, `--dump-state`
and `--load-state` are each rejected when combined with `--state`:

```
error: the argument '--state <PATH>' cannot be used with '--dump-state <PATH>'
error: the argument '--state <PATH>' cannot be used with '--load-state <PATH>'
```

`--state` already loads an existing file on startup and dumps it on exit and
every `--state-interval` seconds, so adding either of the other flags buys
nothing and makes the node refuse to boot. Verified on 1.8.3: a node started
with `--state` alone resumed a saved chain (account 0 balance intact) and
persisted to disk on a 5-second interval.

Measured on a 600-token proof chain: ~508 B/token compressed, ~5.4 KB/token raw,
~80 KB fixed. The measured 29,135-token artifact size is recorded in
`chain-manifest.json` under `state_artifact`.

## Computer 1 — build, verify, export

1. **Install dependencies and verify the Anvil version.**
   ```bash
   npm install
   npx hardhat compile                      # artifacts are gitignored
   anvil --version                          # MUST print 1.8.3
   ```
   `chain:start` and `chain:import` run `anvil --version` themselves and refuse
   to proceed on any other release, because a different foundry build can change
   state-dump layout and account derivation. If `anvil` is not on `PATH`, set
   `MILLOW_ANVIL` to the binary. `MILLOW_SKIP_VERSION_CHECK=1` bypasses the check
   and is logged loudly; do not use it for a release.

2. **Start Anvil 1.8.3** (chainId 31337, `--accounts 20`, `.chain/state.json`):
   ```bash
   npm run chain:start
   ```
   Before the first run, free port 8545 — see [Port precondition](#port-precondition).

3. **Deploy MILLOW:**
   ```bash
   npx hardhat run scripts/deployMillow.js --network localhost
   git diff -- src/config.json              # MUST be empty
   ```
   Addresses and bytecode must match `chain-manifest.json` exactly, or stop.

4. **Tokenize the 29,135 canonical properties:**
   ```bash
   npm run chain:tokenize
   ```
   Replays `data/processed/millow_token_map.csv` in canonical order, preserving
   all 1,396 intentional permutations. It resumes safely from `totalSupply` and
   never re-mints.

5. **Run the full read-only verification:**
   ```bash
   npm run chain:verify -- --full
   ```
   Must report `21 passed, 0 failed` — mapping, ownership, roles, wiring,
   `tokenURI` and `propertyOf` for all 29,135 tokens, and exactly 29,135
   `PropertyMinted` events.

6. **Export the state:**
   ```bash
   npm run chain:export
   ```
   Validates the live chain first, then writes
   `.chain/millow-anvil-state-29135.json.gz` plus `.sha256` and `.meta.json`.
   It refuses to dump a foreign or partial chain.

7. **Verify the SHA-256:**
   ```bash
   # POSIX
   sha256sum .chain/millow-anvil-state-29135.json.gz
   # Windows PowerShell
   (Get-FileHash .chain\millow-anvil-state-29135.json.gz -Algorithm SHA256).Hash.ToLower()
   ```
   It must equal:

   ```
   8ab24ca333102284ff1a933882508bc7f9403bb1cf840a5e2dc1b4654cdb4946
   ```

   That value is also recorded in `chain-manifest.json`
   (`state_artifact.sha256`) and matches the verified production run
   (block 592, 58,278 transactions, 23,055,700 bytes).

8. **Publish as a GitHub Release asset** — upload all three files:

   ```
   millow-anvil-state-29135.json.gz
   millow-anvil-state-29135.sha256
   millow-anvil-state-29135.meta.json
   ```

   **Never commit the state artifact to git.** The repository carries only the
   manifest, the mapping CSV, the scripts and this document.

Then stop the node:

```bash
npm run chain:stop
```

### Port precondition

`chain:start`, `chain:import` and `chain:verify` all refuse to run if anything
else is already serving `127.0.0.1:8545`. A Hardhat node, a leftover Anvil, or
the React dev server's backend can all hold the port. **Before the first
production run, stop whatever occupies 8545 and confirm the port is free:**

```bash
# Windows
Get-NetTCPConnection -LocalPort 8545 -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name='node.exe'" |
  Where-Object { $_.CommandLine -match 'hardhat|anvil' } |
  ForEach-Object { "$($_.ProcessId)  $($_.CommandLine)" }
```

Do not kill a process you did not start without checking with the operator. The
tooling deliberately fails closed here rather than adopting an unknown chain.

### After deployment: config must be unchanged

`scripts/deployMillow.js` writes `src/config.json` from the deploy run. On a
correct deterministic deployment it rewrites the *same* addresses, and with
`core.autocrlf=true` git reports no change. Verify that explicitly before
minting, because a silent address change would point the frontend at the wrong
contracts:

```bash
git diff -- src/config.json     # MUST be empty
```

If it is not empty, stop: the deployment did not land on the canonical
addresses recorded in `chain-manifest.json`.

### Anvil version is enforced

`chain:start` and `chain:import` run `anvil --version` and refuse to proceed
unless it reports exactly the version in `chain-manifest.json`
(`node.version_required`). A different foundry release can change state-dump
layout and account derivation, which breaks reproducibility. Set
`MILLOW_SKIP_VERSION_CHECK=1` to bypass; the bypass is logged loudly.

Then record the printed SHA-256 in `chain-manifest.json`
(`state_artifact.sha256`) and publish the `.gz` as a **GitHub Release asset**.

## Computer 2 — import, verify, run

Computer 2 performs **ZERO deployments** and **ZERO mint transactions**. The
chain is reproduced from the Release artifact, not rebuilt.

1. **Clone the repository:**
   ```bash
   git clone <repo> && cd Real-Estate-based-Blockchain-using-Escrow-Contracts
   npm install
   npx hardhat compile
   anvil --version                          # MUST print 1.8.3
   ```

2. **Download the Release state artifact** (all three files) into the repo root:
   ```
   millow-anvil-state-29135.json.gz
   millow-anvil-state-29135.sha256
   millow-anvil-state-29135.meta.json
   ```

3. **Verify the SHA-256** before trusting it:
   ```bash
   # POSIX
   sha256sum millow-anvil-state-29135.json.gz
   # Windows PowerShell
   (Get-FileHash millow-anvil-state-29135.json.gz -Algorithm SHA256).Hash.ToLower()
   ```
   It must equal:

   ```
   8ab24ca333102284ff1a933882508bc7f9403bb1cf840a5e2dc1b4654cdb4946
   ```

   If it does not match, **stop** — do not import. `chain:import` also re-checks
   the sidecar and warns if it disagrees with `chain-manifest.json`.

4. **Start Anvil with the imported state:**
   ```bash
   npm run chain:import -- --artifact=millow-anvil-state-29135.json.gz
   ```
   This decodes the artifact, proves the Anvil version, installs it as
   `.chain/state.json` and starts Anvil — it deploys nothing and mints nothing.
   It refuses to overwrite an existing populated state file unless `--force`.

5. **Run the full read-only verification:**
   ```bash
   npm run chain:verify -- --full
   ```
   Must report `21 passed, 0 failed` at block 592 with `totalSupply` 29,135.

6. **Regenerate `chain_index.json`:**
   ```bash
   npm run chain:index
   ```
   Strictly read-only — `totalSupply`, `propertyOf`, `ownerOf`, `isOnOffer` and
   `sales` view calls only. It sends no transaction.

7. **Start the services:**
   ```bash
   npm run server        # metadata server (serves tokenURI JSON)
   npm run valuation     # backend
   npm start             # frontend
   ```

Importing is **not** a transaction. The 58,278 transactions (8 deploy +
29,135 mint + 29,135 transfer) are replayed from the artifact, so Computer 2
issues **0 deploy transactions and 0 mint transactions** of its own.

`src/config.json` needs no edit — the imported chain already holds code at the
committed addresses, and a post-import
`git diff -- src/config.json` must be empty.

## npm scripts

| Script | Purpose |
|---|---|
| `chain:start` | start Anvil in the background, loading `.chain/state.json` |
| `chain:stop` | stop Anvil |
| `chain:status` | read-only health + supply + manifest check |
| `chain:export` | `anvil_dumpState` → `.chain/millow-anvil-state-29135.json.gz` + `.sha256` + `.meta.json` |
| `chain:import` | verify + install a state artifact, then start and check it |
| `chain:tokenize` | Computer 1: mint 29,135 in canonical order, deliver to seller |
| `chain:verify` | read-only audit against the manifest |
| `chain:index` | regenerate `chain_index.json` (read-only RPC) |

## Safety rules enforced in code

- `scripts/lib/chain.js` **denylists** `hardhat_reset`, `anvil_reset`,
  `evm_revert` and the balance/code/storage setters. No script in this toolchain
  can issue them.
- No script contains a reset, redeploy-from-scratch, or wipe path.
- `restoreChain.js` / `chain:import` refuses to clobber a populated state file
  without `--force`.
- `verifyChain.js` and `chain:index` never send a transaction.
- `tokenizeChain.js` never deploys, never mints a property that already has a
  token, and aborts on any token-id divergence rather than re-minting.
- `chain:export` refuses to dump a chain that is not a valid MILLOW deployment
  with the canonical token count, so a partial or foreign chain cannot be
  published as the release artifact.
- `.chain/` (state, logs, artifacts) is gitignored and must never be committed.

## Persistence notes

- `--state-interval 30` rewrites `.chain/state.json` every 30 s. Windows cannot
  deliver `SIGTERM` to a detached process, so anvil may not dump state on exit;
  the interval bounds loss instead. Use `chain:export` for a clean artifact.
- Contracts, frontend, backend, AI/ML, dataset and escrow workflow are
  untouched by this toolchain.
