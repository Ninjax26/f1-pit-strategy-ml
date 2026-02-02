import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


def parse_rounds(spec: str) -> list[int]:
    parts = spec.split(",")
    rounds: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            rounds.extend(list(range(int(a), int(b) + 1)))
        else:
            rounds.append(int(part))
    return sorted(set(rounds))


def build_preprocessor(df: pd.DataFrame):
    categorical = [c for c in ["Compound", "Driver", "Team", "TrackStatus", "EventName"] if c in df.columns]
    numeric = [
        c
        for c in [
            "LapNumber",
            "Stint",
            "TyreLife",
            "AirTemp",
            "TrackTemp",
            "Humidity",
            "WindSpeed",
            "WindDirection",
            "RoundNumber",
        ]
        if c in df.columns
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical),
            ("num", numeric_transformer, numeric),
        ]
    )
    return preprocessor, categorical, numeric


def train_and_eval(
    name: str,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    target_is_delta: bool,
    test_race_median: pd.Series | None,
    y_test_seconds: pd.Series | None,
):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if target_is_delta and test_race_median is not None and y_test_seconds is not None:
        preds_seconds = preds + test_race_median.to_numpy()
        mae = mean_absolute_error(y_test_seconds, preds_seconds)
        rmse = np.sqrt(mean_squared_error(y_test_seconds, preds_seconds))
    else:
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

    return {"mae": mae, "rmse": rmse}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline and tree models")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--features", type=str, default="data/features")
    parser.add_argument("--out-dir", type=str, default="data/models")
    parser.add_argument("--metrics-dir", type=str, default="data/metrics")
    parser.add_argument("--train-rounds", type=str, default="1-16")
    parser.add_argument("--test-rounds", type=str, default="17-24")
    parser.add_argument("--include-pit-laps", action="store_true")
    parser.add_argument("--include-safety-cars", action="store_true")
    args = parser.parse_args()

    features_path = Path(args.features) / f"features_{args.season}.parquet"
    df = pd.read_parquet(features_path)

    train_rounds = parse_rounds(args.train_rounds)
    test_rounds = parse_rounds(args.test_rounds)

    if not args.include_pit_laps and "IsPitLap" in df.columns:
        df = df[~df["IsPitLap"]]
    if not args.include_safety_cars and "IsSafetyCar" in df.columns:
        df = df[~df["IsSafetyCar"]]

    train_df = df[df["RoundNumber"].isin(train_rounds)].copy()
    test_df = df[df["RoundNumber"].isin(test_rounds)].copy()

    target = "LapTimeDelta" if "LapTimeDelta" in df.columns else "LapTimeSeconds"
    y_train = train_df[target]
    y_test = test_df[target]

    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    if "LapTimeSeconds" in X_train.columns:
        X_train = X_train.drop(columns=["LapTimeSeconds"])
        X_test = X_test.drop(columns=["LapTimeSeconds"])

    test_race_median = test_df["RaceMedianLap"] if "RaceMedianLap" in test_df.columns else None
    y_test_seconds = test_df["LapTimeSeconds"] if "LapTimeSeconds" in test_df.columns else None

    preprocessor, _, _ = build_preprocessor(df)

    ridge = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )

    hgb = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", HistGradientBoostingRegressor(max_depth=8, learning_rate=0.1, random_state=42)),
        ]
    )

    metrics = {
        "ridge": train_and_eval(
            "ridge",
            ridge,
            X_train,
            y_train,
            X_test,
            y_test,
            target_is_delta=(target == "LapTimeDelta"),
            test_race_median=test_race_median,
            y_test_seconds=y_test_seconds,
        ),
        "hgb": train_and_eval(
            "hgb",
            hgb,
            X_train,
            y_train,
            X_test,
            y_test,
            target_is_delta=(target == "LapTimeDelta"),
            test_race_median=test_race_median,
            y_test_seconds=y_test_seconds,
        ),
        "train_rounds": train_rounds,
        "test_rounds": test_rounds,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ridge, out_dir / "ridge_model.joblib")
    joblib.dump(hgb, out_dir / "hgb_model.joblib")

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
