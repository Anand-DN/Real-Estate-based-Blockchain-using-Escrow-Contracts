# MILLOW Decision Policy v1.0 — Formal Specification and Frozen Semantics

Status: **FINAL** (frozen for v1.0)

Introduced: implementation already present and locked by `backend/test_decision_api.py` (V1–V7, 399 checks). This document records, formalizes and freezes the semantics that are already implemented and tested. It does not redesign, simplify or reinterpret the implementation.

Reference implementation files:

- `backend/decision_policy.py` — pure, deterministic decision engine (standard-library only).
- `backend/decision_context.py` — FastAPI router exposing `GET /api/properties/{mreid_id}/decision` plus the Option A read-only JSON-RPC helper.
- `backend/test_decision_api.py` — frozen V1–V7 suite.

---

## 1. Identity

- `decision_version = "1.0"` (`decision_policy.DECISION_VERSION`).

Decision states (exact strings):

```
PROCEED
REVIEW_REQUIRED
ENHANCED_REVIEW
HOLD
```

| State | Meaning |
| --- | --- |
| `PROCEED` | No decision rule fired; application workflow may advance subject to normal workflow gating. |
| `REVIEW_REQUIRED` | A review rule fired; operator review required before the application advances. |
| `ENHANCED_REVIEW` | An enhanced-review rule fired; enhanced operator review required. |
| `HOLD` | A prerequisite is missing and/or a high-severity rule fired; the application workflow should not advance. |

HOLD is an application/workflow-level outcome only (see Section 7).

---

## 2. Data prerequisites

Frozen per-requirement decisions:

```
D1  LISTING_DATA_INCOMPLETE     → HOLD
D2  AI_ESTIMATE_UNAVAILABLE     → HOLD
D3  RISK_CONTEXT_UNAVAILABLE    → HOLD
D4  CHAIN_SNAPSHOT_UNAVAILABLE  → HOLD
```

Exact implementation semantics (`decision_policy._prerequisite_reasons`, `decision_policy.evaluate`):

- **D1–D3 prevent escalation-rule evaluation.** Their evidence is consumed by the escalation rules, so when any of D1/D2/D3 fires the rules are not evaluated and the reason list contains only the prerequisite reasons.
- **D4 does NOT prevent escalation-rule evaluation.** A missing (or untimestamped) chain snapshot removes only the on-chain workflow evidence, not the policy evidence. When D4 is the only prerequisite:
  - the escalation rules are still evaluated;
  - rule reasons are appended **after** the D4 reason;
  - the overall decision is still `HOLD` because the prerequisite fired.
- The policy **never** silently returns `PROCEED` for missing required evidence.
- `degraded = bool(prerequisite)` — true when any of D1–D4 fired, false otherwise.

---

## 3. Decision rules

Frozen rule→decision mapping:

```
R1 → HOLD
R2 → ENHANCED_REVIEW
R3 → REVIEW_REQUIRED
R4 → ENHANCED_REVIEW
R5 → ENHANCED_REVIEW
R6 → REVIEW_REQUIRED
R7 → REVIEW_REQUIRED
R8 → PROCEED
```

Stable reason codes (`decision_policy.RULE_DECISION` / `_rule_reason`):

| Rule | Reason code | Rule | Reason code |
| --- | --- | --- | --- |
| R1 | `PRICE_HIGH_SEVERITY` | R5 | `MULTIPLE_ELEVATED_INDICATORS` |
| R2 | `STRUCTURE_HIGH_SEVERITY` | R6 | `SINGLE_ELEVATED_INDICATOR` |
| R3 | `DISCLOSURE_HIGH_SEVERITY` | R7 | `PRICE_OUTSIDE_BAND` |
| R4 | `ELEVATED_RISK_WITH_PRICE_DEVIATION` | R8 | `WITHIN_COMPARISON_BAND_NO_ELEVATED_INDICATORS` |

Each reason carries an implicit severity label for the UI chip:

| Rule | Severity | Rule | Severity |
| --- | --- | --- | --- |
| R1 | `hold` | R5 | `enhanced` |
| R2 | `enhanced` | R6 | `review` |
| R3 | `review` | R7 | `review` |
| R4 | `enhanced` | R8 | `info` |

---

## 4. Rule semantics

Severity cutoffs (frozen):

```
high severity     = severity points >= 3
elevated severity = severity points >= 2
```

Rules:

- **R1** — a high-severity `listing_price_gap` indicator exists (pricing dimension).
- **R2** — a high-severity `bedroom_area` indicator exists (structure dimension).
- **R3** — a high-severity `amenity_disclosure` indicator exists (records/disclosure dimension).
- **R4** fires when there is **at least one elevated indicator** **AND** `abs(difference_pct) > band_pct`.
- **R5** fires when the **number of elevated indicators >= 2**.
- **R6** fires when the **number of elevated indicators == 1**.
- **R7** fires when `abs(difference_pct) > band_pct`.
- **R8 is the default / no-rule-fired case.** It fires only when **no** other rule matched (`decision_policy._matched_rules`: `if not matched: matched.append("R8")`).

Important:

- **R8 is NOT a separate literal band check.** An exact boundary condition does not select R8; R8 is simply the outcome when nothing else matched. Its displayed explanation text is produced by the default path and uses the configured comparison band.
- **R4 can co-fire with R1/R2/R3/R5/R6/R7.** There is no exclusivity between the elevated-and-deviation rule and the other rules.
- **A severity-3 indicator is simultaneously high AND elevated**: it contributes to R1/R2/R3 (high) and to the elevated set counted/evaluated by R4/R5/R6.

The `outside` predicate is defined as `abs(difference_pct) > band_pct`; see Section 6 for boundary semantics.

---

## 5. Precedence

Frozen precedence order:

```
R1 > R2 > R3 > R4 > R5 > R6 > R7 > R8
```

- The **first matched rule in precedence order determines the decision** (`RULE_DECISION[matched[0]]`).
- The `reasons[]` list **preserves the matched-rule order** (precedence order) and reports **all** matched lower-precedence rules in addition to the deciding rule.
- When a prerequisite (D1–D4) is present the decision is `HOLD` regardless of the rules, and prerequisite reasons come first, followed by rule reasons (Section 2).

---

## 6. Threshold provenance

No thresholds are introduced, re-derived or tuned here. All numeric thresholds are consumed from existing production sources:

| Constant | Value | Production source |
| --- | --- | --- |
| `SIGNAL_BAND_PCT` | `5.0` | `backend/properties.py` |
| `difference_pct` | existing production `ai_market_signal.difference_pct` | `backend/properties.ai_market_signal` |
| `Z_MEDIUM` | `2.5` | `backend/risk_analysis.py` |
| `Z_HIGH` | `4.0` | `backend/risk_analysis.py` |
| completeness low / high | `0.50` / `0.25` | `backend/risk_analysis.py` |

- `difference_pct` is **not re-derived** in the policy. The policy consumes the production value, whose derivation uses the existing listed-price denominator.
- Indicator severity comes from `risk_analysis`'s `SEVERITY_LABEL` mapping (produced by the z-score status mapping and the disclosure completeness rule):
  - `none = 0`, `low = 1`, `medium = 2`, `high = 3`.
- Boundary semantics (frozen):
  - `outside band  =  abs(difference_pct) > band_pct`
  - An exactly-on-boundary value (e.g. 5.0 on a 5.0 band) is **inside** the band (`abs(diff) > band` is false at equality).
  - Severity cutoffs use `>=` semantics (points >= 3 is high, points >= 2 is elevated).

`backend/test_decision_api.py` asserts that `pol` (`SIGNAL_BAND_PCT`), the risk thresholds and the severity mapping still equal these production constants, guarding against silent drift.

---

## 7. Application-level authority boundary

The Decision Policy is application/workflow-level transaction decision support only:

- It **cannot prevent a direct wallet transaction** to `MillowEscrow`.
- It **does not replace `MillowEscrow.sol`**; the smart contract remains the only authority for actual on-chain execution.

The exact boundary statement (frozen, emitted verbatim as the start of the disclaimer):

> This panel can prevent the MILLOW application from advancing the workflow. It cannot prevent a direct wallet transaction to MillowEscrow, and an on-chain execution does not by itself certify legality, title or valuation.

Additional frozen application-level rules:

- `human_review_required = (decision != PROCEED)`
- When `decision == HOLD` → `blockchain_actions_allowed = []`.
- For **non-PROCEED** decisions, `blockchain_actions_allowed` is the workflow's allowed actions with the review-gated actions removed:

```
APPROVE_BUYER
APPROVE_SELLER
FINALIZE_SALE
```

- The policy **does not claim to enforce** these restrictions on-chain. Enforcing the actions on-chain is `MillowEscrow.sol`'s responsibility; the policy only gates which actions the MILLOW application presents while a decision is outstanding.

Boundary block also records, per response: `ai_valuation_is_not_legal_valuation`, `risk_score_is_not_fraud_probability`, `ai_decision_is_not_approval`, `nft_is_not_legal_title`, `chain_execution_is_not_lawfulness`, and `conformal_uncertainty_used == false`.

---

## 8. Workflow separation

Decision and workflow are **separate concepts**:

- `decision` — the evidence-based policy result (`decision_policy.evaluate`).
- `workflow` — descriptive escrow stage facts and unmet on-chain conditions (`build_workflow`).

The workflow state **can never change** the AI/evidence decision.

Frozen condition → action mappings (`decision_policy.CONDITION_ACTIONS`):

| Condition | Application action |
| --- | --- |
| `NOT_LISTED` / `NOT_TOKENIZED` | `REGISTRATION_OR_LISTING_REQUIRED` |
| `INSPECTION_REQUIRED_NOT_PASSED` | `OBTAIN_INSPECTION` |
| `LENDER_APPROVAL_MISSING` | `OBTAIN_LENDER_APPROVAL` |
| `FUNDING_INCOMPLETE` | `COMPLETE_FUNDING` |
| `BUYER_APPROVAL_MISSING` | `BUYER_APPROVAL` |
| `SELLER_APPROVAL_MISSING` | `SELLER_APPROVAL` |
| `SALE_ALREADY_FINALIZED` | `SALE_ALREADY_FINALIZED` (terminal) |
| `CHAIN_SNAPSHOT_UNAVAILABLE` | `VERIFY_ON_CHAIN_MANUALLY` |
| `LIVE_CHAIN_READ_FAILED` | `VERIFY_ON_CHAIN_MANUALLY` |

Terminology matches the implemented workflow block: `state` (`unavailable` | `snapshot` | `live`), `stage` (escrow `Status` name: `None`, `Listed`, `UnderContract`, `Approved`, `Finalized`, `Cancelled`), `unmet_conditions[]`, `allowed_actions[]`.

---

## 9. Option A RPC (read-only chain access)

The approved Option A architecture is frozen:

- **Read-only** JSON-RPC helper.
- Only the method `eth_call` is allowed (`READ_ONLY_METHODS = {"eth_call"}`).
- The only contract call is `MillowEscrow.sales(tokenId)` using the fixed selector `0xb5f522f7`.
- Queried **only** when the chain snapshot reports `active_sale == true` (and the property is tokenized with a token id). When no sale is active, the chain is **not queried at all**.
- **No** `eth_sendTransaction`.
- **No** signing and **no** private key / wallet.
- **No** state-changing contract call of any kind (allow-list rejects anything not `eth_call`).

RPC source resolution (frozen, mirrors `scripts/lib/chain.js`):

```
RPC_URL = MILLOW_RPC_URL  (environment override, if set)
        | chain-manifest.json.rpc_url
```

The policy backend never writes to the blockchain. Failures raise `ReadOnlyRpcError` instead of fabricating answers; a failed live read degrades the workflow with `LIVE_CHAIN_READ_FAILED` → `VERIFY_ON_CHAIN_MANUALLY` and never changes the decision.

---

## 10. Forbidden language

The policy-language guard is frozen. Policy-authored strings (reason text, display explanation, disclaimer) must not contain forbidden claims such as:

```
fraud detected
fraud probability
safe transaction
guaranteed safe
guaranteed risky
certified valuation
certified detection
undervalued
overvalued
good deal
bad deal
fair price
model confidence
```

- Passthrough `ai_market_signal.label` is **not** policy-authored prose and is **not** rewritten or checked by the guard.
- `backend/test_decision_api.py` asserts none of the forbidden substrings appear in policy-authored text.

---

## 11. Conformal exclusion

Explicitly frozen:

```
conformal_uncertainty_used = false
```

Decision Policy v1.0 **must not** import or use:

- `conformal.py`
- `artifacts/valuation/v2_4/**`
- Experiment 4 outputs
- `qhat`, interval width, coverage, confidence, calibrated uncertainty

Conformal prediction remains a research-only component and is deliberately excluded from the v1.0 policy evidence. The ModuleDocstring of `decision_policy.py` records this exclusion. `backend/test_decision_api.py` asserts the policy modules are standard-library only (no imports outside the standard library) and never reference conformal artifacts.

---

## 12. Response shape

Implemented API:

```
GET /api/properties/{mreid_id}/decision
```

Response keys (frozen):

| Key | Description |
| --- | --- |
| `mreid_id` | Property identifier. |
| `decision` | `PROCEED` / `REVIEW_REQUIRED` / `ENHANCED_REVIEW` / `HOLD`. |
| `decision_version` | `"1.0"`. |
| `human_review_required` | `decision != PROCEED`. |
| `degraded` | `true` when any data prerequisite (D1–D4) fired. |
| `display_explanation` | Policy-authored display explanation (forbidden-language clean). |
| `disclaimer` | Boundary statement + application-support disclaimer, verbatim. |
| `reasons` | Ordered prerequisite/rule reasons (`code`, `severity`, `evidence`, `text`). |
| `required_actions` | `REVIEW_WITH_OPERATOR` (when review required) + mapped workflow actions, deduplicated, in order. |
| `blockchain_actions_allowed` | Workflow allowed actions filtered by decision (Section 7). |
| `signals` | Policy evidence: `ai_estimated_price`, `ai_market_signal`, `risk`, `market_context` (supporting context only — never a policy input), `chain`. |
| `workflow` | Descriptive escrow/workflow block; never changes the decision. |
| `boundary` | Boundary flags incl. `conformal_uncertainty_used: false`. |

`degraded` is **intentional** and is part of the explicit missing-evidence behaviour required by the policy spec (§11 of the task): a missing-evidence HOLD is always surfaced as degraded rather than reported as a normal rule outcome.

`display_explanation` is included for transparency and is covered by the forbidden-language guard. The disclaimer is `BOUNDARY_STATEMENT` plus the supporting "not legal advice / not fraud detection / not a probability / not a replacement for MillowEscrow.sol" sentence.

---

## 13. Known edge case

Documented exactly, behaviour frozen in this phase:

> If `ai_estimated_price` is available but `difference_pct` is absent, D2 does not fire and the implementation can fall back to the default R8 explanation using the configured comparison band.

Concretely: D2 only requires the `ai_estimated_price` fields to be present. A missing `difference_pct` leaves `outside = false` and no risk rule matched (assuming no elevated indicators), so R8 fires as the default with the band falling back to `POLICY_CONFIG["price_band_pct"]` in the explanation text.

This behavior is kept as implemented for v1.0; it is intentionally not "corrected" in this phase.

---

## 14. Provenance note

The earlier task specification that introduced this feature did **not** itself contain every R1–R8 mapping and co-firing detail (e.g. the rule→decision mapping, the 2/3 severity cutoffs, the "elevated-3-indicator is simultaneously high and elevated" behavior, and R8-as-default).

Those semantics were **already implemented and locked** by the existing Decision Policy implementation (`backend/decision_policy.py`) and the V1–V7 test suite (`backend/test_decision_api.py`, 399 checks) before this document was written.

This specification **formalizes those existing semantics**; it does not define them retroactively, and no implementation change was made to fit this document. This provenance is recorded explicitly and deliberately so it is not hidden.