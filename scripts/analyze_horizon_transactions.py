"""
MILLOW - Horizon Transaction Risk Index builder (Phase 5, one-time)

Reads the raw Horizon transaction/property CSVs and writes compact,
deterministic risk-analysis inputs under data/processed/risk/:

  horizon_transaction_summary.pkl
      one row per transaction with precomputed anomaly indicators + score
  horizon_risk_stats.pkl
      comparable-cohort robust statistics (median / MAD) used for scoring
  horizon_city_aggregates.pkl
      city-level descriptive context (used by the property endpoint)

NOT a fraud detector. Scores are statistical anomaly scores derived from
the supplied dataset (which appears procedurally generated) and are not
calibrated probabilities.

Run once from project root:
    python scripts/analyze_horizon_transactions.py
"""

import os
import pickle
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRX = ROOT / "data" / "raw" / "horizon" / "extracted" / "archive" / "Property_Transactions.csv"
PROPS = ROOT / "data" / "raw" / "horizon" / "extracted" / "archive" / "Properties.csv"
OUT_DIR = ROOT / "data" / "processed" / "risk"

# ------------------------------------------------------------- thresholds --
Z_MEDIUM = 2.5
Z_HIGH = 4.0
RAPID_GAP_MEDIUM_DAYS = 180
RAPID_GAP_HIGH_DAYS = 90
JUMP_MEDIUM = 1.5
JUMP_HIGH = 2.5
MIN_COHORT_N = 10
INDICATOR_COUNT = 8
WEIGHT_HIGH = 3


def robust_z(x, med, mad):
    xs = np.asarray(x, dtype=np.float64)
    meds = np.asarray(med, dtype=np.float64)
    mads = np.asarray(mad, dtype=np.float64)
    valid = np.isfinite(meds) & np.isfinite(mads) & (mads > 0)
    z = np.where(valid, 0.6745 * (xs - meds) / np.where(valid, mads, 1.0), np.nan)
    return z


def severity_for_z(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.zeros(len(z), dtype=np.int8)
    a = np.abs(np.where(np.isfinite(z), z, 0.0))
    out[a >= Z_HIGH] = 3
    out[(a >= Z_MEDIUM) & (a < Z_HIGH)] = 2
    return out


def main():
    t0 = time.time()
    print("[1/6] Loading Properties.csv (listed prices)...")
    props = pd.read_csv(
        PROPS, usecols=["Property_ID", "Price_per_sqft"], low_memory=False
    )
    props["Property_ID"] = props["Property_ID"].astype(str).str.strip()
    props = props.drop_duplicates(subset="Property_ID")
    listed = props.set_index("Property_ID")["Price_per_sqft"].astype(float)
    print("      props:", len(props))

    print("[2/6] Loading Property_Transactions.csv...")
    trx = pd.read_csv(
        TRX,
        usecols=[
            "Transaction_ID", "Property_ID", "City_Name", "Transaction_Type",
            "Transaction_Year", "Quarter", "Transaction_Date", "Sale_Price",
            "Price_per_sqft", "Circle_Rate", "Payment_Mode", "Loan_Taken",
            "Negotiation_Percent", "Days_on_Market",
        ],
        low_memory=False,
    )
    print("      transactions:", len(trx))

    print("[3/6] Cleaning / joining...")
    for c in ["Transaction_ID", "Property_ID", "City_Name", "Transaction_Type", "Quarter"]:
        trx[c] = trx[c].astype(str).str.strip()
    trx["qlabel"] = (
        trx["Transaction_Year"].astype(str) + "Q"
        + trx["Quarter"].str.replace("Q", "", regex=False)
    )
    trx["date_int"] = (
        pd.to_datetime(trx["Transaction_Date"], errors="coerce").astype("int64")
        / 10**9 / 86400
    ).astype(int)

    trx["ppsf"] = trx["Price_per_sqft"].astype(float)
    trx["Sale_Price"] = trx["Sale_Price"].astype(float)
    trx["ratio"] = trx["Sale_Price"] / trx["Circle_Rate"].replace(0, np.nan)
    trx["neg"] = pd.to_numeric(trx["Negotiation_Percent"], errors="coerce").astype(
        float
    )
    trx["days"] = trx["Days_on_Market"].astype(float)
    trx["fin"] = (
        (trx["Payment_Mode"] == "Loan") & (trx["Loan_Taken"] != "Yes")
    ).astype(np.int8)

    listed_ppsf = trx["Property_ID"].map(listed).to_numpy(dtype=float)
    trx["listed_ratio"] = np.where(
        np.isfinite(listed_ppsf) & (listed_ppsf > 0),
        trx["ppsf"].to_numpy() / np.where(listed_ppsf > 0, listed_ppsf, 1.0),
        np.nan,
    )

    print("[4/6] Prior-sale history per property (chronological)...")
    trx.sort_values(["Property_ID", "date_int"], inplace=True)
    g = trx.groupby("Property_ID", sort=False)
    trx["prior_count"] = g.cumcount()
    trx["prev_date"] = g["date_int"].shift(1)
    trx["prev_sale"] = g["Sale_Price"].shift(1)
    trx["prev_gap_days"] = (
        (trx["date_int"] - trx["prev_date"]).fillna(-1).astype(int)
    )
    trx["jump_mult"] = np.where(
        trx["prev_sale"].notna() & (trx["prev_sale"] > 0),
        trx["Sale_Price"].astype(float) / trx["prev_sale"],
        np.nan,
    )

    print("[5/6] Comparable cohort statistics (median / MAD)...")
    primary_groups = ["City_Name", "Transaction_Type", "qlabel"]
    fallback_groups = ["City_Name", "qlabel"]
    numeric = ["ppsf", "ratio", "neg", "days"]

    def cohort_stats(df, groups):
        med = df.groupby(groups)[numeric].median()
        mad = (
            df.groupby(groups)[numeric]
            .apply(lambda x: x.apply(lambda c: (c - c.median()).abs().median()))
        )
        mad.columns = [f"mad_{c}" for c in mad.columns]
        n = df.groupby(groups).size().rename("n")
        return med.join(mad).join(n).reset_index()

    stats_primary = cohort_stats(trx, primary_groups)
    stats_fallback = cohort_stats(trx, fallback_groups)

    lr = trx.groupby("City_Name")["listed_ratio"]
    stats_listed = (
        lr.median().rename("med_lr").to_frame().join(
            lr.apply(lambda s: (s - s.median()).abs().median()).rename("mad_lr")
        ).reset_index()
    )

    sp = stats_primary.copy()
    sp.columns = ["p_" + c for c in sp.columns]
    trx = trx.merge(
        sp, how="left",
        left_on=primary_groups,
        right_on=["p_" + c for c in primary_groups],
    )
    sf = stats_fallback.copy()
    sf.columns = ["f_" + c for c in sf.columns]
    trx = trx.merge(
        sf, how="left",
        left_on=fallback_groups,
        right_on=["f_" + c for c in fallback_groups],
    )
    trx = trx.merge(stats_listed, how="left", on="City_Name")

    prim_has = trx["p_n"].notna() & (trx["p_n"] >= MIN_COHORT_N)
    fall_has = trx["f_n"].notna() & (trx["f_n"] >= MIN_COHORT_N)

    mapped = {
        "med_ppsf": ("p_ppsf", "f_ppsf"),
        "mad_ppsf": ("p_mad_ppsf", "f_mad_ppsf"),
        "med_ratio": ("p_ratio", "f_ratio"),
        "mad_ratio": ("p_mad_ratio", "f_mad_ratio"),
        "med_neg": ("p_neg", "f_neg"),
        "mad_neg": ("p_mad_neg", "f_mad_neg"),
        "med_days": ("p_days", "f_days"),
        "mad_days": ("p_mad_days", "f_mad_days"),
    }
    for out_name, (pcol, fcol) in mapped.items():
        trx[out_name] = np.where(
            prim_has, trx[pcol], np.where(fall_has, trx[fcol], np.nan)
        )
    trx["cohort_used"] = np.where(
        prim_has, "primary", np.where(fall_has, "fallback", "n/a")
    )
    trx["cohort_n"] = np.where(
        prim_has, trx["p_n"], np.where(fall_has, trx["f_n"], 0)
    ).astype(int)

    trx["z_ppsf"] = robust_z(trx["ppsf"], trx["med_ppsf"], trx["mad_ppsf"])
    trx["z_ratio"] = robust_z(trx["ratio"], trx["med_ratio"], trx["mad_ratio"])
    trx["z_neg"] = robust_z(trx["neg"], trx["med_neg"], trx["mad_neg"])
    trx["z_days"] = robust_z(trx["days"], trx["med_days"], trx["mad_days"])
    trx["z_listed"] = np.where(
        trx["mad_lr"].notna() & (trx["mad_lr"] > 0) & trx["listed_ratio"].notna(),
        0.6745 * (trx["listed_ratio"] - trx["med_lr"]) / trx["mad_lr"],
        np.nan,
    )

    print("[6/6] Scoring + writing...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s_price = severity_for_z(trx["z_ppsf"].to_numpy())
    s_ratio = severity_for_z(trx["z_ratio"].to_numpy())
    s_neg = severity_for_z(trx["z_neg"].to_numpy())
    s_days = severity_for_z(trx["z_days"].to_numpy())
    s_listed = severity_for_z(trx["z_listed"].to_numpy())
    s_fin = np.where(trx["fin"].to_numpy() == 1, 2, 0).astype(np.int8)
    s_churn = np.where(
        (trx["prev_gap_days"].to_numpy() > 0)
        & (trx["prev_gap_days"].to_numpy() <= RAPID_GAP_MEDIUM_DAYS),
        np.where(trx["prev_gap_days"].to_numpy() <= RAPID_GAP_HIGH_DAYS, 3, 2),
        0,
    ).astype(np.int8)
    s_jump = np.where(
        trx["jump_mult"].notna().to_numpy(),
        np.where(trx["jump_mult"].to_numpy() >= JUMP_HIGH, 3,
                 np.where(trx["jump_mult"].to_numpy() >= JUMP_MEDIUM, 2, 0)),
        0,
    ).astype(np.int8)

    weights = (
        s_price.astype(float) + s_ratio.astype(float) + s_neg.astype(float)
        + s_days.astype(float) + s_fin.astype(float) + s_churn.astype(float)
        + s_jump.astype(float) + s_listed.astype(float)
    )
    score = (
        100.0 * weights / (WEIGHT_HIGH * INDICATOR_COUNT)
    ).round().clip(0, 100).astype(np.int16)

    summary = pd.DataFrame(
        {
            "txn_id": trx["Transaction_ID"].astype("category"),
            "prop_id": trx["Property_ID"].astype("category"),
            "city": trx["City_Name"].astype("category"),
            "txn_type": trx["Transaction_Type"].astype("category"),
            "qlabel": trx["qlabel"].astype("category"),
            "cohort_used": trx["cohort_used"].astype("category"),
            "cohort_n": trx["cohort_n"].to_numpy(dtype=np.int32),
            "ppsf": trx["ppsf"].to_numpy(dtype=np.float32),
            "ratio": trx["ratio"].to_numpy(dtype=np.float32),
            "neg": trx["neg"].to_numpy(dtype=np.float32),
            "days": trx["days"].to_numpy(dtype=np.float32),
            "listed_ratio": trx["listed_ratio"].to_numpy(dtype=np.float32),
            "fin": trx["fin"].to_numpy(dtype=np.int8),
            "prior_count": trx["prior_count"].to_numpy(dtype=np.int16),
            "prev_gap_days": trx["prev_gap_days"].to_numpy(dtype=np.int32),
            "jump_mult": trx["jump_mult"].to_numpy(dtype=np.float32),
            "z_ppsf": trx["z_ppsf"].to_numpy(dtype=np.float32),
            "z_ratio": trx["z_ratio"].to_numpy(dtype=np.float32),
            "z_neg": trx["z_neg"].to_numpy(dtype=np.float32),
            "z_days": trx["z_days"].to_numpy(dtype=np.float32),
            "z_listed": trx["z_listed"].to_numpy(dtype=np.float32),
            "s_price": s_price,
            "s_ratio": s_ratio,
            "s_neg": s_neg,
            "s_days": s_days,
            "s_fin": s_fin,
            "s_churn": s_churn,
            "s_jump": s_jump,
            "s_listed": s_listed,
            "score": score,
        }
    )
    summary = summary.sort_values("txn_id").reset_index(drop=True)

    with open(OUT_DIR / "horizon_risk_stats.pkl", "wb") as f:
        pickle.dump(
            {
                "primary": stats_primary.set_index(primary_groups),
                "fallback": stats_fallback.set_index(fallback_groups),
                "listed": stats_listed.set_index("City_Name"),
            },
            f,
        )

    agg = trx.groupby("City_Name").agg(
        n_txn=("Transaction_ID", "count"),
        n_props=("Property_ID", "nunique"),
        med_ppsf=("ppsf", "median"),
        med_ratio=("ratio", "median"),
        med_neg=("neg", "median"),
        med_days=("days", "median"),
        first_date=("Transaction_Date", "min"),
        last_date=("Transaction_Date", "max"),
    )
    per_city_rapid = trx.groupby("City_Name").apply(
        lambda d: float(
            ((d["prev_gap_days"] > 0)
             & (d["prev_gap_days"] <= RAPID_GAP_MEDIUM_DAYS)).mean()
        )
    ).rename("rapid_share")
    per_city_txnpp = trx.groupby(["City_Name", "Property_ID"]).size()
    per_city_per_prop_median = (
        per_city_txnpp.groupby("City_Name").median()
    ).rename("med_txn_per_prop")
    agg = agg.join(per_city_rapid).join(per_city_per_prop_median)

    summary.to_pickle(OUT_DIR / "horizon_transaction_summary.pkl", compression=None)
    agg.to_pickle(OUT_DIR / "horizon_city_aggregates.pkl", compression=None)

    print("      wrote:", sorted(p.name for p in OUT_DIR.iterdir()))
    print("      rows:", len(summary),
          "score min/median/p95/max:",
          int(summary["score"].min()), int(summary["score"].median()),
          int(summary["score"].quantile(0.95)), int(summary["score"].max()))
    print("      s_fin counts:", summary["s_fin"].value_counts().to_dict())
    print("      cohort used:", summary["cohort_used"].value_counts().to_dict())
    print("      focus cities:", {
        c: int(agg.loc[c, "n_txn"])
        for c in ["Bengaluru", "Chennai", "Kolkata", "Hyderabad", "Mumbai", "New Delhi"]
        if c in agg.index
    })
    print(f"   done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()