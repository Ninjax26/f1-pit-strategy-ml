import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_parquets(raw_root: Path, season: int) -> pd.DataFrame:
    """Load all lap parquets for a single season directory."""
    paths = list(raw_root.glob("round_*_*/laps.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {raw_root}")
    frames = [pd.read_parquet(p) for p in sorted(paths)]
    df = pd.concat(frames, ignore_index=True)
    df["Season"] = season
    return df


def build_feature_df(df: pd.DataFrame, exclude_safety_cars: bool = False) -> pd.DataFrame:
    """Apply feature engineering to a raw laps DataFrame (may span multiple seasons)."""
    # Basic filters
    if "LapTime" in df.columns:
        df = df[df["LapTime"].notna()].copy()
        if np.issubdtype(df["LapTime"].dtype, np.timedelta64):
            df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
        else:
            df["LapTimeSeconds"] = pd.to_numeric(df["LapTime"], errors="coerce")
    else:
        raise ValueError("LapTime column not found")

    # Drop deleted laps
    if "Deleted" in df.columns:
        df = df[df["Deleted"] == False]

    # Pit-lap and safety-car flags
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

    # Race-level normalization target: group by (Season, RoundNumber) to avoid
    # cross-season contamination when multiple seasons are loaded together.
    base_mask = (~df["IsPitLap"]) & (~df["IsSafetyCar"])
    base_df = df[base_mask].copy()

    group_keys = [k for k in ["Season", "RoundNumber"] if k in base_df.columns]
    if group_keys and not base_df.empty:
        race_median = base_df.groupby(group_keys)["LapTimeSeconds"].median()
        df = df.join(race_median.rename("RaceMedianLap"), on=group_keys)
    else:
        df["RaceMedianLap"] = np.nan

    overall_median = base_df["LapTimeSeconds"].median() if not base_df.empty else df["LapTimeSeconds"].median()
    df["RaceMedianLap"] = df["RaceMedianLap"].fillna(overall_median)
    df["LapTimeDelta"] = df["LapTimeSeconds"] - df["RaceMedianLap"]

    if exclude_safety_cars:
        df = df[~df["IsSafetyCar"]]

    # Select columns for modeling
    keep_cols = [
        "Season",
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
        "PreRaceBaseline",
        "BaselineSourceSeason",
        "LapTimeDelta",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].astype(str).str.upper()
    if "TrackStatus" in df.columns:
        df["TrackStatus"] = df["TrackStatus"].astype(str)

    df = df.dropna(subset=["LapTimeSeconds"])
    return df


def apply_pre_race_baseline(df: pd.DataFrame, history_df: pd.DataFrame, target_season: int) -> pd.DataFrame:
    """Create an absolute-time baseline using only seasons before the target season."""
    result = df.copy()
    history = history_df.copy()
    if "Season" not in history.columns or "EventName" not in history.columns:
        raise ValueError("Historical features must contain Season and EventName")

    history = history[history["Season"] < target_season]
    if "IsPitLap" in history.columns:
        history = history[~history["IsPitLap"]]
    if "IsSafetyCar" in history.columns:
        history = history[~history["IsSafetyCar"]]
    history = history.dropna(subset=["EventName", "LapTimeSeconds"])

    race_baselines = (
        history.groupby(["Season", "EventName"], as_index=False)["LapTimeSeconds"]
        .median()
        .sort_values("Season")
        .drop_duplicates("EventName", keep="last")
    )
    baseline_map = race_baselines.set_index("EventName")["LapTimeSeconds"]
    source_map = race_baselines.set_index("EventName")["Season"]
    result["PreRaceBaseline"] = result["EventName"].map(baseline_map)
    result["BaselineSourceSeason"] = result["EventName"].map(source_map).astype("Int64")
    result["LapTimeDelta"] = result["LapTimeSeconds"] - result["PreRaceBaseline"]
    return result


def load_prior_feature_history(features_dir: Path, target_season: int) -> pd.DataFrame:
    frames = []
    for path in sorted(features_dir.glob("features_*.parquet")):
        suffix = path.stem.removeprefix("features_")
        if not suffix.isdigit() or int(suffix) >= target_season:
            continue
        frame = pd.read_parquet(path)
        if "Season" not in frame.columns:
            frame["Season"] = int(suffix)
        frames.append(frame)
    if not frames:
        raise RuntimeError(
            f"No prior-season feature files found in {features_dir}. "
            "A leakage-free pre-race baseline requires at least one earlier season."
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean feature dataset (single or multi-season)")
    # Single season (original interface — unchanged)
    parser.add_argument("--season", type=int, default=2024,
                        help="Single season year (ignored if --seasons is given)")
    # Multi-season (new)
    parser.add_argument("--seasons", type=str, default=None,
                        help="Comma-separated years, e.g. '2022,2023,2024'. Overrides --season.")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/features")
    parser.add_argument("--exclude-safety-cars", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve which seasons to load
    if args.seasons:
        season_list = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    else:
        season_list = [args.season]

    frames = []
    for season in season_list:
        raw_root = raw_dir / str(season)
        if not raw_root.exists():
            print(f"WARNING: Raw data directory not found for season {season}: {raw_root}. Skipping.")
            continue
        print(f"Loading season {season} from {raw_root} ...")
        frame = load_all_parquets(raw_root, season)
        print(f"  → {len(frame):,} raw laps loaded")
        frames.append(frame)

    if not frames:
        raise RuntimeError("No seasons could be loaded. Check --raw-dir and --seasons arguments.")

    df = pd.concat(frames, ignore_index=True)
    print(f"Total raw laps across all seasons: {len(df):,}")

    df = build_feature_df(df, exclude_safety_cars=args.exclude_safety_cars)
    if len(season_list) == 1:
        history_df = load_prior_feature_history(out_root, season_list[0])
        df = apply_pre_race_baseline(df, history_df, season_list[0])
        supported = int(df["PreRaceBaseline"].notna().sum())
        print(f"Pre-race baseline available for {supported:,}/{len(df):,} laps")
    print(f"Total laps after feature engineering: {len(df):,}")

    # Output path: single season → features_YYYY.parquet (original name),
    # multi-season → features_YYYY_YYYY.parquet
    if len(season_list) == 1:
        out_name = f"features_{season_list[0]}.parquet"
    else:
        out_name = f"features_{min(season_list)}_{max(season_list)}.parquet"

    out_path = out_root / out_name
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  ({len(df):,} rows, {df['Season'].nunique()} season(s))")


if __name__ == "__main__":
    main()
