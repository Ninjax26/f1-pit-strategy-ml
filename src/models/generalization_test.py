"""
Multi-Season Generalization Testing for F1 Pit Strategy ML
===========================================================

Runs three clearly separated experiments:

  Baseline  — 2024 Rounds 1-16 train → 2024 Rounds 17-24 test (within-season)
  Exp 1     — 2022+2023+2024 train   → 2025 test (cross-season, same era)
  Exp 2     — 2021 train             → 2022 test (regulation-shift stress test)

Outputs
-------
  data/metrics/generalization/generalization_results.csv
  data/metrics/generalization/generalization_results.json
  Prints a Markdown summary table to stdout.

Usage
-----
  python src/models/generalization_test.py
  python src/models/generalization_test.py --features-dir data/features --models hgb,ridge
"""
import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ─── Constants ────────────────────────────────────────────────────────────────

CATEGORICAL_FEATURES = ["Compound", "Driver", "Team", "TrackStatus", "EventName"]
NUMERIC_FEATURES = [
    "LapNumber", "Stint", "TyreLife",
    "AirTemp", "TrackTemp", "Humidity",
    "WindSpeed", "WindDirection", "RoundNumber",
]

# Columns never used as model inputs (bookkeeping / target columns)
DROP_FROM_X = {"LapTimeSeconds", "LapTimeDelta", "RaceMedianLap", "PreRaceBaseline", "BaselineSourceSeason", "Season",
               "SessionName", "IsPitLap", "IsSafetyCar"}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_season(features_dir: Path, season: int) -> pd.DataFrame | None:
    """Load a single season's feature parquet. Returns None if missing."""
    path = features_dir / f"features_{season}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    # Ensure Season column is present (older parquets may predate the column)
    if "Season" not in df.columns:
        df["Season"] = season
    return df


def load_seasons(features_dir: Path, seasons: list[int]) -> tuple[pd.DataFrame, list[int]]:
    """
    Load and concatenate feature parquets for the given season years.
    Returns (combined_df, loaded_seasons).  Missing seasons are skipped with a warning.
    """
    frames = []
    loaded = []
    for season in seasons:
        df = load_season(features_dir, season)
        if df is None:
            print(f"  WARNING: features_{season}.parquet not found in {features_dir} — skipping.")
            continue
        print(f"  Loaded {season}: {len(df):,} laps")
        frames.append(df)
        loaded.append(season)

    if not frames:
        raise RuntimeError(
            f"None of the required seasons {seasons} were found in {features_dir}. "
            "Run `build_features.py --season YEAR` for each required year first."
        )

    return pd.concat(frames, ignore_index=True), loaded


# ─── Splitting ────────────────────────────────────────────────────────────────

def season_split(
    df: pd.DataFrame,
    train_seasons: list[int],
    test_seasons: list[int],
    *,
    train_rounds: list[int] | None = None,
    test_rounds: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reusable split function used by all three experiments.

    - For within-season splits, pass a single season in both lists and restrict
      further via train_rounds / test_rounds.
    - For cross-season splits, pass different season lists; rounds are ignored.
    """
    if train_rounds is not None:
        train_df = df[
            df["Season"].isin(train_seasons) & df["RoundNumber"].isin(train_rounds)
        ].copy()
    else:
        train_df = df[df["Season"].isin(train_seasons)].copy()

    if test_rounds is not None:
        test_df = df[
            df["Season"].isin(test_seasons) & df["RoundNumber"].isin(test_rounds)
        ].copy()
    else:
        test_df = df[df["Season"].isin(test_seasons)].copy()

    return train_df, test_df


# ─── Unseen-Track Audit ───────────────────────────────────────────────────────

def audit_unseen_tracks(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[set, int]:
    """
    Report tracks present in test but absent from train.
    Rows are KEPT (OneHotEncoder handle_unknown='ignore' zeros them out).
    Returns (unseen_track_names, n_affected_rows).
    """
    if "EventName" not in train_df.columns or "EventName" not in test_df.columns:
        return set(), 0

    train_tracks = set(train_df["EventName"].dropna().unique())
    test_tracks = set(test_df["EventName"].dropna().unique())
    unseen = test_tracks - train_tracks
    n_rows = int(test_df["EventName"].isin(unseen).sum())
    return unseen, n_rows


# ─── Model Building ───────────────────────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build sklearn ColumnTransformer fitted on df column availability."""
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, cat_cols),
        ("num", num_pipe, num_cols),
    ])


def make_pipeline(model_name: str, preprocessor: ColumnTransformer) -> Pipeline:
    if model_name == "ridge":
        estimator = Ridge(alpha=1.0, random_state=42)
    elif model_name == "hgb":
        estimator = HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.1, random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


# ─── X / y Preparation ────────────────────────────────────────────────────────

def prepare_xy(
    df: pd.DataFrame,
    exclude_pit_laps: bool = True,
    exclude_safety_cars: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None, pd.Series | None]:
    """
    Returns (X, y, race_median_series, y_seconds_series).
    y is LapTimeDelta if available, else LapTimeSeconds.
    race_median_series and y_seconds_series are used to convert delta
    predictions back to absolute seconds for fair MAE comparison.
    """
    mask = pd.Series(True, index=df.index)
    if exclude_pit_laps and "IsPitLap" in df.columns:
        mask &= ~df["IsPitLap"]
    if exclude_safety_cars and "IsSafetyCar" in df.columns:
        mask &= ~df["IsSafetyCar"]
    df = df[mask].copy()
    if "PreRaceBaseline" not in df.columns:
        raise ValueError("PreRaceBaseline is required for leakage-free generalization testing")
    df = df[df["PreRaceBaseline"].notna()].copy()

    target = "LapTimeDelta" if "LapTimeDelta" in df.columns else "LapTimeSeconds"
    y = df[target]
    race_median = df["PreRaceBaseline"]
    y_seconds = df["LapTimeSeconds"] if "LapTimeSeconds" in df.columns else None

    drop_cols = DROP_FROM_X | {target}
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return X, y, race_median, y_seconds


def eval_predictions(
    preds: np.ndarray,
    y: pd.Series,
    target_is_delta: bool,
    race_median: pd.Series | None,
    y_seconds: pd.Series | None,
) -> dict:
    if target_is_delta and race_median is not None and y_seconds is not None:
        preds_abs = preds + race_median.to_numpy()
        mae = float(mean_absolute_error(y_seconds, preds_abs))
        rmse = float(np.sqrt(mean_squared_error(y_seconds, preds_abs)))
    else:
        mae = float(mean_absolute_error(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
    return {"mae": mae, "rmse": rmse}


# ─── Single Experiment Runner ─────────────────────────────────────────────────

def run_experiment(
    label: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_names: list[str],
    output_dir: Path,
) -> list[dict]:
    """
    Trains and evaluates each model for one experiment.
    Returns a list of result dicts (one per model).
    """
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Train: {len(train_df):,} laps  |  Test: {len(test_df):,} laps")

    X_train, y_train, _, _ = prepare_xy(train_df)
    X_test, y_test, test_race_median, y_test_seconds = prepare_xy(test_df)
    target_is_delta = "LapTimeDelta" in test_df.columns

    unseen_tracks, _ = audit_unseen_tracks(train_df, test_df)
    n_unseen_rows = int(X_test["EventName"].isin(unseen_tracks).sum()) if "EventName" in X_test.columns else 0
    if unseen_tracks:
        print(f"  Unseen tracks in test ({n_unseen_rows} evaluated rows): {sorted(unseen_tracks)}")
    else:
        print("  No unseen tracks in test set.")

    preprocessor = build_preprocessor(X_train)

    rows = []
    for model_name in model_names:
        pipe = make_pipeline(model_name, preprocessor)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = eval_predictions(preds, y_test, target_is_delta, test_race_median, y_test_seconds)

        # Persist the experiment model (never overwrites app models)
        model_out = output_dir / f"{label.replace(' ', '_').replace('→', 'to')}_{model_name}.joblib"
        import joblib
        joblib.dump(pipe, model_out)

        row = {
            "experiment": label,
            "model": model_name,
            "mae": round(metrics["mae"], 4),
            "rmse": round(metrics["rmse"], 4),
            "n_test": int(X_test.shape[0]),
            "unseen_track_rows": n_unseen_rows,
        }
        print(f"  [{model_name.upper():6s}]  MAE={metrics['mae']:.3f}s  RMSE={metrics['rmse']:.3f}s")
        rows.append(row)

    return rows


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-season generalization test")
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--metrics-dir", default="data/metrics")
    parser.add_argument("--models", default="hgb,ridge")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_dir = Path(args.metrics_dir) / "generalization"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    all_rows: list[dict] = []

    # ── Baseline: within-season 2024 ──────────────────────────────────────────
    print("\n[Baseline] Loading 2024 ...")
    df_2024 = load_season(features_dir, 2024)
    if df_2024 is None:
        raise RuntimeError("features_2024.parquet is required for the baseline. Run build_features.py --season 2024 first.")

    train_df_base, test_df_base = season_split(
        df_2024, [2024], [2024],
        train_rounds=list(range(1, 17)),
        test_rounds=list(range(17, 25)),
    )
    rows = run_experiment(
        "Baseline 2024 R1-16 → R17-24",
        train_df_base, test_df_base,
        model_names, output_dir,
    )
    for r in rows:
        r["train_set"] = "2024 Rounds 1–16"
        r["test_set"] = "2024 Rounds 17–24"
    all_rows.extend(rows)

    # ── Experiment 1: 2022+2023+2024 → 2025 ──────────────────────────────────
    print("\n[Experiment 1] Loading 2022, 2023, 2024, 2025 ...")
    df_train_e1, loaded_train_e1 = load_seasons(features_dir, [2022, 2023, 2024])
    df_test_e1, loaded_test_e1 = load_seasons(features_dir, [2025])

    if loaded_train_e1 and loaded_test_e1:
        train_df_e1, test_df_e1 = season_split(df_train_e1, loaded_train_e1, loaded_test_e1)
        # Fix: combine both season sets into one df for season_split to work correctly
        combined_e1 = pd.concat([df_train_e1, df_test_e1], ignore_index=True)
        train_df_e1, test_df_e1 = season_split(combined_e1, loaded_train_e1, loaded_test_e1)
        train_label = "+".join(str(s) for s in sorted(loaded_train_e1))
        test_label = "+".join(str(s) for s in sorted(loaded_test_e1))
        rows = run_experiment(
            f"Exp1 {train_label} → {test_label}",
            train_df_e1, test_df_e1,
            model_names, output_dir,
        )
        for r in rows:
            r["train_set"] = f"{train_label} (full seasons)"
            r["test_set"] = f"{test_label} ({test_df_e1['RoundNumber'].nunique()} rounds)"
        all_rows.extend(rows)
    else:
        print("  SKIPPED Experiment 1: insufficient data (need 2022–2024 + 2025).")

    # ── Experiment 2: 2021 → 2022 (regulation-shift stress test) ─────────────
    print("\n[Experiment 2] Loading 2021, 2022 ...")
    df_train_e2, loaded_train_e2 = load_seasons(features_dir, [2021])
    df_test_e2, loaded_test_e2 = load_seasons(features_dir, [2022])

    if loaded_train_e2 and loaded_test_e2 and "PreRaceBaseline" in df_train_e2.columns:
        combined_e2 = pd.concat([df_train_e2, df_test_e2], ignore_index=True)
        train_df_e2, test_df_e2 = season_split(combined_e2, loaded_train_e2, loaded_test_e2)
        rows = run_experiment(
            "Exp2 2021 → 2022 (reg-shift)",
            train_df_e2, test_df_e2,
            model_names, output_dir,
        )
        for r in rows:
            r["train_set"] = "2021 (full season)"
            r["test_set"] = "2022 (full season)"
        all_rows.extend(rows)
    else:
        print("  SKIPPED Experiment 2: 2021 has no earlier-season baseline and cannot be evaluated leakage-free.")

    # ── Output ────────────────────────────────────────────────────────────────
    if not all_rows:
        print("\nNo experiments completed. Exiting.")
        return

    results_df = pd.DataFrame(all_rows)
    # Reorder columns for readability
    col_order = ["experiment", "train_set", "test_set", "model", "mae", "rmse", "n_test", "unseen_track_rows"]
    results_df = results_df[[c for c in col_order if c in results_df.columns]]

    csv_path = output_dir / "generalization_results.csv"
    json_path = output_dir / "generalization_results.json"
    results_df.to_csv(csv_path, index=False)
    results_df.to_json(json_path, orient="records", indent=2)
    print(f"\nSaved results to:\n  {csv_path}\n  {json_path}")

    # Pretty-print table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(results_df.to_string(index=False, float_format="%.3f"))
    print("="*80)

    # Interpretation hint
    if len(all_rows) >= 4:
        base_hgb = next((r for r in all_rows if "Baseline" in r["experiment"] and r["model"] == "hgb"), None)
        e1_hgb = next((r for r in all_rows if "Exp1" in r["experiment"] and r["model"] == "hgb"), None)
        if base_hgb and e1_hgb:
            gap_e1 = e1_hgb["mae"] - base_hgb["mae"]
            print(textwrap.dedent(f"""
            INTERPRETATION (HGB):
              Baseline MAE:          {base_hgb['mae']:.3f}s
              Cross-season MAE:      {e1_hgb['mae']:.3f}s  (Δ +{gap_e1:.3f}s vs baseline)
            """))


if __name__ == "__main__":
    main()
