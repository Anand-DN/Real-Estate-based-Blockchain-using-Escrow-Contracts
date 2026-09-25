"""
MILLOW - Chain index snapshot loader (Phase 9)

Serves the offline chain-index snapshot written by
scripts/exportChainIndex.js (data/processed/chain_index.json).  Catalogue
consumers (dashboard, search join, ChatBot, tokenization status) read
catalogue-wide tokenization / listing / sale state from this file instead of
issuing ~29k RPC calls from the frontend.

Rules:
  * The snapshot is a point-in-time read, refreshed by the export script.
    Every payload that uses it carries the exported_at timestamp so consumers
    know how fresh it is.
  * A missing/unreadable snapshot must never crash catalogue endpoints: they
    degrade to "chain state unavailable" instead of fabricating values.
"""

import json
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "data" / "processed" / "chain_index.json"

_INDEX: dict | None = None
_LOAD_TIME: float | None = None
_LOCK = threading.Lock()


def load():
    """Load (once) and return the snapshot dict, or None when unavailable."""
    global _INDEX, _LOAD_TIME
    with _LOCK:
        if _INDEX is not None:
            return _INDEX
        try:
            if not SNAPSHOT_PATH.exists():
                return None
            with open(SNAPSHOT_PATH, "r", encoding="utf-8") as fh:
                _INDEX = json.load(fh)
            _LOAD_TIME = SNAPSHOT_PATH.stat().st_mtime
        except (json.JSONDecodeError, OSError):
            _INDEX = None
            _LOAD_TIME = None
    return _INDEX


def available():
    return load() is not None


def exported_at():
    index = load()
    if index is None:
        return None
    return index.get("exported_at")


def metadata():
    index = load()
    if index is None:
        return None
    return {
        "chain_id": index.get("chain_id"),
        "exported_at": index.get("exported_at"),
        "contracts": index.get("contracts"),
        "counts": index.get("counts"),
    }


def for_property(mreid_id):
    """Chain record for one property, or None when unavailable / not tokenized."""
    index = load()
    if index is None:
        return None
    return index.get("properties", {}).get(str(mreid_id))


def as_summary(mreid_id):
    """
    Minimal catalogue-joinable record for a property.  Returns None when the
    snapshot is unavailable; otherwise a dict with the chain state (tokenized
    false when the property has no token), never raising.
    """
    index = load()
    if index is None:
        return None
    record = index.get("properties", {}).get(str(mreid_id))
    if record is None:
        return {
            "tokenized": False,
            "token_id": None,
            "owner": None,
            "listed": False,
            "active_sale": False,
            "finalized": False,
            "sale_status": None,
        }
    return {
        "tokenized": bool(record.get("tokenized")),
        "token_id": record.get("token_id"),
        "owner": record.get("owner"),
        "listed": bool(record.get("listed")),
        "active_sale": bool(record.get("active_sale")),
        "finalized": bool(record.get("finalized")),
        "sale_status": record.get("sale_status"),
    }


def counts():
    index = load()
    if index is None:
        return None
    return dict(index.get("counts") or {})


def refresh():
    """Drop the cached snapshot so the next call re-reads the file."""
    global _INDEX, _LOAD_TIME
    with _LOCK:
        _INDEX = None
        _LOAD_TIME = None