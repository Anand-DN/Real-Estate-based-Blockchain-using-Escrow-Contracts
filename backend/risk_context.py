"""
MILLOW - Transaction Risk & Anomaly Analysis router (Phase 5)

Exposes two deterministic, explainable anomaly analyses:

  * GET /api/properties/{mreid_id}/risk-analysis
      Property-listing level (MREID dataset). No model, no NHB, and the
      large Horizon transaction summary is not loaded.
  * GET /api/transactions/{transaction_id}/risk-analysis
      Transaction level (Horizon dataset). Uses the precomputed risk
      summary built by scripts/analyze_horizon_transactions.py.

Scores are anomaly scores (0-100). They are NOT probabilities and do NOT
constitute fraud detection or any certified/guaranteed claim.
"""

from fastapi import APIRouter, HTTPException

import risk_analysis

router = APIRouter()


@router.get("/api/properties/{mreid_id}/risk-analysis")
def property_risk_analysis(mreid_id: str):
    if not risk_analysis.risk_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Risk-analysis data (Horizon transaction summaries) is not "
                "available. Build it with "
                "scripts/analyze_horizon_transactions.py."
            ),
        )

    payload = risk_analysis.property_payload(mreid_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Property '{mreid_id}' not found.",
        )
    return payload


@router.get("/api/transactions/{transaction_id}/risk-analysis")
def transaction_risk_analysis(transaction_id: str):
    if not risk_analysis.risk_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Risk-analysis data (Horizon transaction summaries) is not "
                "available. Build it with "
                "scripts/analyze_horizon_transactions.py."
            ),
        )

    payload = risk_analysis.transaction_payload(transaction_id)
    if payload is None:
        raise HTTPException(
            status_code=404,
            detail=f"Transaction '{transaction_id}' not found.",
        )
    return payload