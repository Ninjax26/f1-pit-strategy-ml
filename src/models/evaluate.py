import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
    numeric = [c for c in ["LapNumber", "Stint", "TyreLife", "AirTemp", "TrackTemp", "Humidity", "WindSpeed", "WindDirection", "RoundNumber"] if c in df.columns]
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical),
            ("num", numeric_transformer, numeric),
        ]
    )


def make_model(model_name: str, preprocessor: ColumnTransformer):
    if model_name == "ridge":
        estimator = Ridge(alpha=1.0, random_state=42)
    elif model_name == "hgb":
        estimator = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.1, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def metrics_for_split(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def group_metrics(df_pred: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for key, grp in df_pred.groupby(group_col):
        rows.append({
            group_col: key,
            "mae": mean_absolute_error(grp["LapTimeSeconds"], grp["pred"]),
            "rmse": np.sqrt(mean_squared_error(grp["LapTimeSeconds"], grp["pred"])),
            "n": len(grp),
        })
    return pd.DataFrame(rows).sort_values("mae")


def rolling_splits(rounds: list[int], train_min: int, test_window: int, step: int):
    max_round = max(rounds)
    splits = []
    for train_end in range(train_min, max_round - test_window + 1, step):
        train_rounds = list(range(1, train_end + 1))
        test_rounds = list(range(train_end + 1, train_end + 1 + test_window))
        splits.append((train_rounds, test_rounds))
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models and generate diagnostics")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--features", type=str, default="data/features")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--metrics-dir", type=str, default="data/metrics")
    parser.add_argument("--train-rounds", type=str, default="1-16")
    parser.add_argument("--test-rounds", type=str, default="17-24")
    parser.add_argument("--include-pit-laps", action="store_true")
    parser.add_argument("--include-safety-cars", action="store_true")
    parser.add_argument("--rolling", action="store_true")
    parser.add_argument("--rolling-train-min", type=int, default=8)
    parser.add_argument("--rolling-test-window", type=int, default=2)
    parser.add_argument("--rolling-step", type=int, default=2)
    parser.add_argument("--models", type=str, default="ridge,hgb")
    args = parser.parse_args()

    features_path = Path(args.features) / f"features_{args.season}.parquet"
    df = pd.read_parquet(features_path)
    if not args.include_pit_laps and "IsPitLap" in df.columns:
        df = df[~df["IsPitLap"]]
    if not args.include_safety_cars and "IsSafetyCar" in df.columns:
        df = df[~df["IsSafetyCar"]]

    target = "LapTimeDelta" if "LapTimeDelta" in df.columns else "LapTimeSeconds"
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.rolling:
        rounds = sorted(df["RoundNumber"].dropna().unique().astype(int))
        splits = rolling_splits(rounds, args.rolling_train_min, args.rolling_test_window, args.rolling_step)
        for model_name in model_names:
            rolling_rows = []
            for train_rounds, test_rounds in splits:
                train_df = df[df["RoundNumber"].isin(train_rounds)].copy()
                test_df = df[df["RoundNumber"].isin(test_rounds)].copy()
                preprocessor = build_preprocessor(df)
                model = make_model(model_name, preprocessor)
                X_train = train_df.drop(columns=[target])
                X_test = test_df.drop(columns=[target])
                if "LapTimeSeconds" in X_train.columns:
                    X_train = X_train.drop(columns=["LapTimeSeconds"])
                    X_test = X_test.drop(columns=["LapTimeSeconds"])
                model.fit(X_train, train_df[target])
                preds = model.predict(X_test)
                if target == "LapTimeDelta" and "RaceMedianLap" in test_df.columns and "LapTimeSeconds" in test_df.columns:
                    preds_seconds = preds + test_df["RaceMedianLap"].to_numpy()
                    metrics = metrics_for_split(test_df["LapTimeSeconds"], preds_seconds)
                else:
                    metrics = metrics_for_split(test_df[target], preds)
                rolling_rows.append({"model": model_name, "train_rounds": train_rounds, "test_rounds": test_rounds, **metrics})
            with open(metrics_dir / f"rolling_metrics_{model_name}.json", "w") as f:
                json.dump(rolling_rows, f, indent=2)
            print(f"Wrote {metrics_dir / f'rolling_metrics_{model_name}.json'}")

    train_rounds = parse_rounds(args.train_rounds)
    test_rounds = parse_rounds(args.test_rounds)
    train_df = df[df["RoundNumber"].isin(train_rounds)].copy()
    test_df = df[df["RoundNumber"].isin(test_rounds)].copy()

    for model_name in model_names:
        model_path = Path(args.models_dir) / f"{model_name}_model.joblib"
        model = joblib.load(model_path)
        X_test = test_df.drop(columns=[target])
        if "LapTimeSeconds" in X_test.columns:
            X_test = X_test.drop(columns=["LapTimeSeconds"])
        preds = model.predict(X_test)
        df_pred = test_df.copy()
        if target == "LapTimeDelta" and "RaceMedianLap" in df_pred.columns and "LapTimeSeconds" in df_pred.columns:
            df_pred["pred"] = preds + df_pred["RaceMedianLap"].to_numpy()
            df_pred["residual"] = df_pred["pred"] - df_pred["LapTimeSeconds"]
            overall = metrics_for_split(df_pred["LapTimeSeconds"], df_pred["pred"])
        else:
            df_pred["pred"] = preds
            df_pred["residual"] = df_pred["pred"] - df_pred[target]
            overall = metrics_for_split(df_pred[target], df_pred["pred"])
        group_results = {
            "compound": group_metrics(df_pred, "Compound") if "Compound" in df_pred.columns else None,
            "stint": group_metrics(df_pred, "Stint") if "Stint" in df_pred.columns else None,
            "round": group_metrics(df_pred, "RoundNumber"),
        }
        df_pred.to_parquet(metrics_dir / f"predictions_{model_name}.parquet", index=False)
        with open(metrics_dir / f"metrics_{model_name}.json", "w") as f:
            json.dump({"overall": overall, "train_rounds": train_rounds, "test_rounds": test_rounds}, f, indent=2)
        if group_results["compound"] is not None:
            group_results["compound"].to_csv(metrics_dir / f"mae_by_compound_{model_name}.csv", index=False)
        if group_results["stint"] is not None:
            group_results["stint"].to_csv(metrics_dir / f"mae_by_stint_{model_name}.csv", index=False)
        group_results["round"].to_csv(metrics_dir / f"mae_by_round_{model_name}.csv", index=False)
        print(json.dumps({"model": model_name, "overall": overall}, indent=2))


if __name__ == "__main__":
    main()
