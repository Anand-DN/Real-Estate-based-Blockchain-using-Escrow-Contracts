import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/processed/MREID_property.csv"
OUTPUT_DIR = "artifacts/valuation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }


def make_predictions(train, test, level):
    """
    Training-only PPSF baseline.

    level:
        city     -> source_city
        location -> location
    """

    train = train.copy()
    test = test.copy()

    train["ppsf"] = train["price"] / train["area"]

    if level == "city":
        key = "source_city"
    else:
        key = "location"

    stats = (
        train.groupby(key)["ppsf"]
        .agg(["median", "count"])
        .rename(columns={
            "median": "median_ppsf",
            "count": "sample_count"
        })
    )

    global_median = train["ppsf"].median()

    if level == "location":
        city_stats = (
            train.groupby("source_city")["ppsf"]
            .median()
        )

        preds = []

        for _, row in test.iterrows():

            loc = row["location"]
            city = row["source_city"]

            # Require at least 5 training observations
            if loc in stats.index and stats.loc[loc, "sample_count"] >= 5:
                ppsf = stats.loc[loc, "median_ppsf"]

            elif city in city_stats.index:
                ppsf = city_stats.loc[city]

            else:
                ppsf = global_median

            preds.append(ppsf * row["area"])

        return np.array(preds)

    else:
        city_stats = stats["median_ppsf"]

        preds = []

        for _, row in test.iterrows():

            city = row["source_city"]

            if city in city_stats.index:
                ppsf = city_stats.loc[city]
            else:
                ppsf = global_median

            preds.append(ppsf * row["area"])

        return np.array(preds)


def run_random_split(df):

    train, test = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    results = []

    for level in ["city", "location"]:

        pred = make_predictions(train, test, level)
        metrics = evaluate(test["price"].values, pred)

        results.append({
            "split": "random",
            "baseline": level + "_median_ppsf",
            **metrics
        })

    return results


def run_grouped_split(df):

    groups = (
        df["source_city"].astype(str)
        + "__"
        + df["location"].astype(str)
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(df, groups=groups)
    )

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    results = []

    for level in ["city", "location"]:

        pred = make_predictions(train, test, level)
        metrics = evaluate(test["price"].values, pred)

        results.append({
            "split": "location_grouped",
            "baseline": level + "_median_ppsf",
            **metrics
        })

    return results


def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    required = [
        "price",
        "area",
        "location",
        "source_city"
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Remove only technically invalid records
    df = df[
        (df["price"] > 0)
        & (df["area"] > 0)
    ].copy()

    df["location"] = (
        df["location"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["source_city"] = (
        df["source_city"]
        .astype(str)
        .str.strip()
    )

    print("Rows:", len(df))

    results = []

    print("\nRunning random split...")
    results.extend(run_random_split(df))

    print("Running location-grouped split...")
    results.extend(run_grouped_split(df))

    result_df = pd.DataFrame(results)

    print("\n================ RESULTS ================\n")

    for _, row in result_df.iterrows():

        print(
            f"{row['split']:20s} "
            f"{row['baseline']:25s} "
            f"MAE=₹{row['MAE']:,.0f} "
            f"RMSE=₹{row['RMSE']:,.0f} "
            f"R2={row['R2']:.4f} "
            f"MAPE={row['MAPE']:.2f}%"
        )

    output_path = os.path.join(
        OUTPUT_DIR,
        "valuation_baseline_metrics.csv"
    )

    result_df.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()