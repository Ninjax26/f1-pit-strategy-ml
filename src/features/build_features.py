import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_parquets(raw_root: Path) -> pd.DataFrame:
    paths = list(raw_root.glob("round_*_*/laps.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {raw_root}")
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean feature dataset")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/features")
    parser.add_argument("--exclude-safety-cars", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_dir) / str(args.season)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = load_all_parquets(raw_root)

    # Basic filters
    if "LapTime" in df.columns:
        df = df[df["LapTime"].notna()].copy()
        if np.issubdtype(df["LapTime"].dtype, np.timedelta64):
            df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
        else:
            df["LapTimeSeconds"] = pd.to_numeric(df["LapTime"], errors="coerce")
    else:
        raise ValueError("LapTime column not found")

    # Drop deleted laps if column exists
    if "Deleted" in df.columns:
        df = df[df["Deleted"] == False]

    # Add pit-lap and safety-car flags
    is_pit_lap = pd.Series(False, index=df.index)
    for col in ["PitInTime", "PitOutTime"]:
        if col in df.columns:
            is_pit_lap = is_pit_lap | df[col].notna()
    df["IsPitLap"] = is_pit_lap

    if "TrackStatus" in df.columns:
        status_str = df["TrackStatus"].astype(str)
        df["IsSafetyCar"] = status_str.str.contains("4|5", regex=True)
    else:
        df["IsSafetyCar"] = False

    # Race-level normalization target (median race pace on clean laps)
    base_mask = (~df["IsPitLap"]) & (~df["IsSafetyCar"])
    base_df = df[base_mask].copy()
    if "RoundNumber" in base_df.columns and not base_df.empty:
        race_median = base_df.groupby("RoundNumber")["LapTimeSeconds"].median()
        df["RaceMedianLap"] = df["RoundNumber"].map(race_median)
    else:
        df["RaceMedianLap"] = np.nan

    overall_median = base_df["LapTimeSeconds"].median() if not base_df.empty else df["LapTimeSeconds"].median()
    df["RaceMedianLap"] = df["RaceMedianLap"].fillna(overall_median)
    df["LapTimeDelta"] = df["LapTimeSeconds"] - df["RaceMedianLap"]

    # Optionally remove safety car / VSC laps
    if args.exclude_safety_cars:
        df = df[~df["IsSafetyCar"]]

    # Keep a clean, focused feature set
    keep_cols = [
        "LapTimeSeconds",
        "LapNumber",
        "Stint",
        "Compound",
        "TyreLife",
        "TrackStatus",
        "Driver",
        "Team",
        "RoundNumber",
        "EventName",
        "AirTemp",
        "TrackTemp",
        "Humidity",
        "WindSpeed",
        "WindDirection",
        "SessionName",
        "IsPitLap",
        "IsSafetyCar",
        "RaceMedianLap",
        "LapTimeDelta",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Basic cleanup
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].astype(str).str.upper()
    if "TrackStatus" in df.columns:
        df["TrackStatus"] = df["TrackStatus"].astype(str)

    df = df.dropna(subset=["LapTimeSeconds"])

    out_path = out_root / f"features_{args.season}.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
