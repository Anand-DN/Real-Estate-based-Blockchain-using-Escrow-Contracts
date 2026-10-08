"""
MILLOW - Decision Policy API router (Decision Policy v1.0)

Exposes the deterministic, explainable transaction decision policy:

    GET /api/properties/{mreid_id}/decision

The response combines evidence that already exists in the repository
(listed price, the frozen V5 AI research estimate, the existing price
comparison band, the existing listing risk indicators and the exported
chain snapshot) into one of four decision states:

    PROCEED | REVIEW_REQUIRED | ENHANCED_REVIEW | HOLD

Two concepts are kept strictly separate:

  decision  - evidence-based policy result (decision_policy.evaluate)
  workflow  - descriptive escrow stage facts and unmet on-chain
              conditions (build_workflow).  The workflow can never change
              the decision.

Option A (approved design): the only chain access is a minimal READ-ONLY
JSON-RPC helper used when the snapshot reports an active sale for the
property.  It allow-lists eth_call exclusively: it never signs, never
sends a transaction and never touches state.  When no sale is active the
chain is not queried at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

try:  # optional: only needed when a live read is actually required
    import httpx
except Exception:  # pragma: no cover - environment without httpx
    httpx = None

import chain_index
import decision_policy
import market_context
import properties
import risk_analysis

router = APIRouter()

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "chain-manifest.json"
MILLOW_ESCROW_ABI_PATH = ROOT / "src" / "abis" / "MillowEscrow.json"

# keccak256("sales(uint256)")[0:4].  Derived twice with this repository's own
# tooling (ethers.utils.id and the ethereum-cryptography keccak used by
# hardhat) - both agree - and cross-checked against the canonical empty-string
# keccak vector c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470.
SALES_SELECTOR = "0xb5f522f7"

# Nothing outside this set may ever be requested from the node.
READ_ONLY_METHODS = frozenset({"eth_call"})

# MillowEscrow.Status enum names, identical to scripts/exportChainIndex.js
# and src/lib/blockchain.js so snapshot and live stages line up.
STATUS_NAMES = {
    0: "None",
    1: "Listed",
    2: "UnderContract",
    3: "Approved",
    4: "Finalized",
    5: "Cancelled",
}

# Stage-valid contract actions while a sale is in progress
# (MillowEscrow.sol modifiers saleInProgress / onlySaleParty).
IN_PROGRESS_ACTIONS = [
    "FUND_BALANCE",
    "REQUEST_FINANCING",
    "APPROVE_FINANCING",
    "REJECT_FINANCING",
    "FUND_LOAN",
    "PAY_DOWN_PAYMENT",
    "SET_INSPECTION_STATUS",
    "APPROVE_BUYER",
    "APPROVE_SELLER",
    "CANCEL_SALE",
]

_SCALAR_TYPES = frozenset({
    "address", "bool", "bytes32",
    "uint8", "uint16", "uint32", "uint64", "uint128", "uint256",
    "int8", "int16", "int32", "int64", "int128", "int256",
})


class ReadOnlyRpcError(RuntimeError):
    """Raised when the read-only JSON-RPC helper cannot answer safely."""


# ============================================================
# READ-ONLY JSON-RPC HELPER  (Option A)
# ============================================================

def rpc_url():
    """Resolve the local node URL the same way scripts/lib/chain.js does."""
    env_url = os.environ.get("MILLOW_RPC_URL")
    if env_url:
        return env_url
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    return manifest.get("rpc_url")


def escrow_address():
    """MillowEscrow address from the exported snapshot, else the manifest."""
    metadata = chain_index.metadata() or {}
    address = (metadata.get("contracts") or {}).get("millow_escrow")
    if address:
        return address
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    return (manifest.get("contracts") or {}).get("millow_escrow")


def _http_post(url, payload, timeout):
    if httpx is None:
        raise ReadOnlyRpcError("httpx is not available for read-only RPC.")
    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


class ReadOnlyRpcClient:
    """
    Minimal read-only JSON-RPC client.

    Guarantees:
      * only READ_ONLY_METHODS (eth_call) may be requested
      * no signing, no wallet, no private key
      * no state-changing method (eth_sendTransaction, evm_mine, ...)
      * never raises into a fabricated answer - callers get ReadOnlyRpcError
    """

    def __init__(self, url, post=None, timeout=5.0):
        self.url = url
        self.timeout = timeout
        self._post = post if post is not None else (
            lambda u, p: _http_post(u, p, timeout)
        )

    def _request(self, method, params):
        if method not in READ_ONLY_METHODS:
            raise ReadOnlyRpcError(
                f"method '{method}' is not on the read-only allowlist"
            )
        if not self.url:
            raise ReadOnlyRpcError("no RPC url is configured for this node.")
        response = self._post(self.url, {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        })
        if not isinstance(response, dict):
            raise ReadOnlyRpcError("unexpected JSON-RPC response.")
        if response.get("error"):
            raise ReadOnlyRpcError(
                f"JSON-RPC error: {response['error']}"
            )
        return response.get("result")

    def eth_call(self, to, data):
        """Read-only call to a view function.  Returns 0x-prefixed hex."""
        if not isinstance(to, str) or not to.startswith("0x"):
            raise ReadOnlyRpcError("eth_call target must be a 0x address.")
        if not isinstance(data, str) or not data.startswith("0x"):
            raise ReadOnlyRpcError("eth_call data must be 0x-prefixed.")
        result = self._request("eth_call", [
            {"to": to, "data": data},
            "latest",
        ])
        if not isinstance(result, str) or not result.startswith("0x"):
            raise ReadOnlyRpcError("eth_call returned no data.")
        return result


# ============================================================
# ABI ENCODE / DECODE  (static types only, no keccak at runtime)
# ============================================================

def encode_sale_call(token_id):
    """calldata for MillowEscrow.sales(uint256)."""
    value = int(token_id)
    if value < 0:
        raise ReadOnlyRpcError("token id must not be negative.")
    return f"{SALES_SELECTOR}{value:064x}"


def _leaf_types(components):
    leaves = []
    for item in components or []:
        if item.get("type") == "tuple":
            leaves.extend(_leaf_types(item.get("components") or []))
        else:
            leaves.append(item.get("type"))
    return leaves


def _coerce(kind, word):
    if kind == "address":
        return "0x" + word[12:].hex()
    if kind == "bool":
        return int.from_bytes(word, "big") != 0
    if kind == "bytes32":
        return "0x" + word.hex()
    return int.from_bytes(word, "big")


def _decode_components(components, words, index):
    out = {}
    for position, item in enumerate(components or []):
        name = item.get("name") or f"value_{position}"
        if item.get("type") == "tuple":
            nested, index = _decode_components(
                item.get("components") or [], words, index
            )
            out[name] = nested
        else:
            kind = item.get("type")
            if kind not in _SCALAR_TYPES:
                raise ReadOnlyRpcError(
                    f"unsupported ABI type '{kind}' - refusing to guess."
                )
            out[name] = _coerce(kind, words[index])
            index += 1
    return out, index


def decode_outputs(components, data_hex):
    """Decode static (non-dynamic) ABI return data into nested values."""
    raw_hex = data_hex[2:] if data_hex.startswith("0x") else data_hex
    if len(raw_hex) % 64 != 0:
        raise ReadOnlyRpcError("return data length is not ABI aligned.")
    raw = bytes.fromhex(raw_hex)
    leaves = _leaf_types(components)
    if len(raw) != 32 * len(leaves):
        raise ReadOnlyRpcError(
            f"expected {32 * len(leaves)} return bytes for "
            f"{len(leaves)} values, got {len(raw)}."
        )
    words = [raw[i:i + 32] for i in range(0, len(raw), 32)]
    values, used = _decode_components(components, words, 0)
    if used != len(words):
        raise ReadOnlyRpcError("ABI decode did not consume the return data.")
    return values


def load_escrow_abi():
    try:
        with open(MILLOW_ESCROW_ABI_PATH, "r", encoding="utf-8") as fh:
            abi = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadOnlyRpcError(f"MillowEscrow ABI unavailable: {exc}")
    if not isinstance(abi, list):
        raise ReadOnlyRpcError("MillowEscrow ABI has an unexpected shape.")
    return abi


def sales_outputs():
    for entry in load_escrow_abi():
        if entry.get("type") == "function" and entry.get("name") == "sales":
            return entry.get("outputs") or []
    raise ReadOnlyRpcError("sales(uint256) not found in the MillowEscrow ABI.")


def read_live_sale(token_id, client=None):
    """
    Read-only eth_call to MillowEscrow.sales(tokenId).

    Raises ReadOnlyRpcError instead of ever returning guessed data.
    """
    address = escrow_address()
    if not address:
        raise ReadOnlyRpcError("MillowEscrow address is unavailable.")
    rpc = client if client is not None else ReadOnlyRpcClient(rpc_url())
    raw = rpc.eth_call(address, encode_sale_call(token_id))
    return decode_outputs(sales_outputs(), raw)


# ============================================================
# WORKFLOW (pure - descriptive facts only, never a judgement)
# ============================================================

def _wf(state, stage, conditions, actions):
    return {
        "state": state,
        "stage": stage,
        "unmet_conditions": list(conditions),
        "allowed_actions": list(actions),
    }


def _workflow_from_sale(sale):
    status = int(sale.get("status") or 0)
    stage = STATUS_NAMES.get(status, "None")

    if status == 4:
        return _wf("live", stage, ["SALE_ALREADY_FINALIZED"], [])

    if status in (2, 3):
        financing = sale.get("financing") or {}
        conditions = []
        if sale.get("inspectionRequired") and not sale.get("inspectionPassed"):
            conditions.append("INSPECTION_REQUIRED_NOT_PASSED")
        if sale.get("lenderRequired") and not financing.get("approved"):
            conditions.append("LENDER_APPROVAL_MISSING")
        price = int(sale.get("priceWei") or 0)
        funded = (
            int(sale.get("buyerFundedWei") or 0)
            + int(sale.get("lenderFundedWei") or 0)
        )
        if price > 0 and funded != price:
            conditions.append("FUNDING_INCOMPLETE")
        if not sale.get("buyerApproved"):
            conditions.append("BUYER_APPROVAL_MISSING")
        if not sale.get("sellerApproved"):
            conditions.append("SELLER_APPROVAL_MISSING")

        actions = list(IN_PROGRESS_ACTIONS)
        if status == 3:
            actions.append("FINALIZE_SALE")
        return _wf("live", stage, conditions, actions)

    if status == 1:
        return _wf("live", stage, [], ["CLOSE_LISTING", "COMMIT_AND_DEPOSIT"])

    return _wf("live", stage, [], ["LIST_FOR_SALE"])


def build_workflow(
    summary=None,
    snapshot_available=False,
    exported_at=None,
    live_sale=None,
    live_error=None,
):
    """
    Pure mapping from chain facts to the workflow block.

    Live sale data is only ever consulted when the snapshot already says
    the property has an active sale; otherwise no chain read is performed.
    """
    if (
        not snapshot_available
        or not isinstance(summary, dict)
        or not exported_at
    ):
        return _wf(
            "unavailable", None, ["CHAIN_SNAPSHOT_UNAVAILABLE"], []
        )

    stage = summary.get("sale_status")

    if not summary.get("tokenized"):
        return _wf("snapshot", stage, ["NOT_TOKENIZED"], [])

    if summary.get("finalized") or stage == "Finalized":
        return _wf("snapshot", stage, ["SALE_ALREADY_FINALIZED"], [])

    if summary.get("active_sale"):
        if live_error or not isinstance(live_sale, dict):
            return _wf("unavailable", stage, ["LIVE_CHAIN_READ_FAILED"], [])
        return _workflow_from_sale(live_sale)

    if not summary.get("listed"):
        return _wf("snapshot", stage, ["NOT_LISTED"], [])

    return _wf("snapshot", stage, [], ["LIST_FOR_SALE"])


# ============================================================
# ROUTE
# ============================================================

BOUNDARY = {
    "ai_valuation_is_not_legal_valuation": True,
    "risk_score_is_not_fraud_probability": True,
    "ai_decision_is_not_approval": True,
    "nft_is_not_legal_title": True,
    "chain_execution_is_not_lawfulness": True,
    "conformal_uncertainty_used": False,
}


def _market_context_or_none(mreid_id):
    """Supporting context only - never a decision-policy input."""
    try:
        return market_context.property_market_context(mreid_id)
    except HTTPException:
        return None


@router.get("/api/properties/{mreid_id}/decision")
def property_decision(mreid_id: str):
    key = mreid_id.strip()
    if key not in properties._id_to_index:
        raise HTTPException(
            status_code=404,
            detail=f"Property '{mreid_id}' not found.",
        )

    idx = properties._id_to_index[key]
    row = properties.PROPERTIES.iloc[idx]
    ai_price, ai_ppsf = properties._batch_ai_predict(
        properties.PROPERTIES.iloc[[idx]]
    )
    detail = properties._detail(row, ai_price[0], ai_ppsf[0])

    # ---- existing listing risk analysis (D3 when unavailable) ----
    risk_payload = None
    if risk_analysis.risk_available():
        risk_payload = risk_analysis.property_payload(key)

    # ---- chain snapshot ----
    snapshot_available = chain_index.available()
    exported_at = chain_index.exported_at()
    summary = chain_index.as_summary(key) if snapshot_available else None
    chain_signals = None
    if snapshot_available and isinstance(summary, dict) and exported_at:
        chain_signals = {
            **summary,
            "available": True,
            "exported_at": exported_at,
        }

    # ---- workflow: live read ONLY when the snapshot reports a sale ----
    live_sale = None
    live_error = None
    if (
        isinstance(summary, dict)
        and summary.get("active_sale")
        and summary.get("tokenized")
        and summary.get("token_id") is not None
    ):
        try:
            live_sale = read_live_sale(summary.get("token_id"))
        except (ReadOnlyRpcError, OSError, ValueError) as exc:
            live_error = str(exc)

    workflow = build_workflow(
        summary=summary,
        snapshot_available=snapshot_available,
        exported_at=exported_at,
        live_sale=live_sale,
        live_error=live_error,
    )

    # ---- policy inputs (market context is deliberately excluded) ----
    policy_signals = {
        "mreid_id": key,
        "listed_price": detail.get("listed_price"),
        "ai_estimated_price": (
            detail.get("ai_estimation") or {}
        ).get("ai_estimated_price"),
        "ai_market_signal": detail.get("ai_market_signal"),
        "risk": risk_payload,
        "chain": chain_signals,
    }

    result = decision_policy.evaluate(policy_signals, workflow)

    return {
        "mreid_id": key,
        "decision": result["decision"],
        "decision_version": result["decision_version"],
        "human_review_required": result["human_review_required"],
        "degraded": result["degraded"],
        "display_explanation": result["display_explanation"],
        "disclaimer": result["disclaimer"],
        "reasons": result["reasons"],
        "required_actions": result["required_actions"],
        "blockchain_actions_allowed": result["blockchain_actions_allowed"],
        "signals": {
            "ai_estimated_price": policy_signals["ai_estimated_price"],
            "ai_market_signal": policy_signals["ai_market_signal"],
            "risk": risk_payload,
            "market_context": _market_context_or_none(key),
            "chain": chain_signals,
        },
        "workflow": result["workflow"],
        "boundary": dict(BOUNDARY),
    }
