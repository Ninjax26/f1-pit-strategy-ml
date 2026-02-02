import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def parse_strategy(spec: str) -> list[tuple[str, int]]:
    # Example: "SOFT:18,MEDIUM:22,HARD:20"
    stints = []
    for part in spec.split(","):
        compound, length = part.split(":")
        stints.append((compound.strip().upper(), int(length)))
    return stints


def available_compounds(race_df: pd.DataFrame, include_wet: bool) -> list[str]:
    compounds = sorted(set(race_df.get("Compound", pd.Series([])).astype(str).str.upper().dropna()))
    dry = [c for c in compounds if c in {"SOFT", "MEDIUM", "HARD"}]
    wet = [c for c in compounds if c not in {"SOFT", "MEDIUM", "HARD"}]
    if include_wet:
        return dry + wet if dry else wet
    return dry if dry else compounds


def generate_strategies(
    total_laps: int,
    compounds: list[str],
    max_stops: int,
    min_stint: int,
    max_stint: int,
    step: int,
    require_two_compounds: bool,
) -> dict[str, list[tuple[str, int]]]:
    strategies: dict[str, list[tuple[str, int]]] = {}

    if max_stint <= 0:
        max_stint = total_laps

    # One-stop (2 stints)
    if max_stops >= 1:
        for c1 in compounds:
            for c2 in compounds:
                if require_two_compounds and c1 == c2:
                    continue
                for s1 in range(min_stint, total_laps - min_stint + 1, step):
                    s2 = total_laps - s1
                    if s2 < min_stint or s2 > max_stint:
                        continue
                    name = f"1stop_{c1[0]}-{c2[0]}_{s1}-{s2}"
                    strategies[name] = [(c1, s1), (c2, s2)]

    # Two-stop (3 stints)
    if max_stops >= 2:
        for c1 in compounds:
            for c2 in compounds:
                for c3 in compounds:
                    if require_two_compounds and len({c1, c2, c3}) < 2:
                        continue
                    for s1 in range(min_stint, total_laps - 2 * min_stint + 1, step):
                        for s2 in range(min_stint, total_laps - s1 - min_stint + 1, step):
                            s3 = total_laps - s1 - s2
                            if s3 < min_stint or s3 > max_stint:
                                continue
                            name = f"2stop_{c1[0]}-{c2[0]}-{c3[0]}_{s1}-{s2}-{s3}"
                            strategies[name] = [(c1, s1), (c2, s2), (c3, s3)]

    return strategies


def build_laps(base_laps: pd.DataFrame, strategy: list[tuple[str, int]]) -> pd.DataFrame:
    laps = base_laps.copy().reset_index(drop=True)
    lap_idx = 0
    stint_idx = 1
    for compound, length in strategy:
        for i in range(length):
            if lap_idx >= len(laps):
                break
            laps.loc[lap_idx, "Compound"] = compound
            if "TyreLife" in laps.columns:
                laps.loc[lap_idx, "TyreLife"] = i + 1
            if "Stint" in laps.columns:
                laps.loc[lap_idx, "Stint"] = stint_idx
            lap_idx += 1
        stint_idx += 1
    return laps


def _to_seconds(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.timedelta64):
        return series.dt.total_seconds()
    return pd.to_numeric(series, errors="coerce")


def _robust_filter(losses: list[float]) -> list[float]:
    losses = [x for x in losses if pd.notna(x) and x > 0]
    if len(losses) < 5:
        return losses

    q05, q95 = np.quantile(losses, [0.05, 0.95])
    low = max(q05, 5.0)
    high = min(q95, 60.0)
    filtered = [x for x in losses if low <= x <= high]
    return filtered if len(filtered) >= 3 else losses


def estimate_pit_loss_from_raw(raw_df: pd.DataFrame) -> float | None:
    if "LapTime" not in raw_df.columns:
        return None

    df = raw_df.copy()
    df["LapTimeSeconds"] = _to_seconds(df["LapTime"])

    clean = df.copy()
    if "Deleted" in clean.columns:
        clean = clean[clean["Deleted"] == False]
    if "PitInTime" in clean.columns:
        clean = clean[clean["PitInTime"].isna()]
    if "PitOutTime" in clean.columns:
        clean = clean[clean["PitOutTime"].isna()]
    if "TrackStatus" in clean.columns:
        status_str = clean["TrackStatus"].astype(str)
        clean = clean[~status_str.str.contains("4|5", regex=True)]

    if clean.empty:
        return None

    losses = []
    if "Driver" in df.columns and "LapNumber" in df.columns:
        for driver, ddf in df.groupby("Driver"):
            ddf = ddf.sort_values("LapNumber")
            base = clean[clean["Driver"] == driver]["LapTimeSeconds"].median()
            if np.isnan(base):
                continue
            pit_in = ddf[ddf["PitInTime"].notna()] if "PitInTime" in ddf.columns else ddf.iloc[0:0]
            if pit_in.empty:
                continue
            for _, in_lap in pit_in.iterrows():
                next_lap_number = int(in_lap["LapNumber"]) + 1
                out_lap = ddf[ddf["LapNumber"] == next_lap_number]
                if out_lap.empty or ("PitOutTime" in out_lap.columns and out_lap["PitOutTime"].isna().all()):
                    continue
                loss = (in_lap["LapTimeSeconds"] + out_lap.iloc[0]["LapTimeSeconds"]) - 2 * base
                if pd.notna(loss):
                    losses.append(loss)
    else:
        base = clean["LapTimeSeconds"].median()
        in_laps = df[df["PitInTime"].notna()] if "PitInTime" in df.columns else df.iloc[0:0]
        out_laps = df[df["PitOutTime"].notna()] if "PitOutTime" in df.columns else df.iloc[0:0]
        losses.extend(list(in_laps["LapTimeSeconds"] - base))
        losses.extend(list(out_laps["LapTimeSeconds"] - base))

    losses = _robust_filter(losses)
    if not losses:
        return None

    return float(np.median(losses))


def load_pit_loss(metrics_path: Path, round_number: int) -> float | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    row = df[df["round"] == round_number]
    if row.empty:
        return None
    value = row.iloc[0].get("pit_loss_median")
    return None if pd.isna(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate pit strategies using a trained model")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--features", type=str, default="data/features")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--metrics-dir", type=str, default="data/metrics")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--driver", type=str, required=True)
    parser.add_argument("--model", type=str, default="hgb", choices=["ridge", "hgb"])
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--pit-loss", type=str, default="auto")
    parser.add_argument("--max-stops", type=int, default=2)
    parser.add_argument("--min-stint", type=int, default=8)
    parser.add_argument("--max-stint", type=int, default=35)
    parser.add_argument("--stint-step", type=int, default=2)
    parser.add_argument("--include-wet", action="store_true")
    parser.add_argument("--allow-single-compound", action="store_true")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    features_path = Path(args.features) / f"features_{args.season}.parquet"
    df = pd.read_parquet(features_path)

    race_df = df[(df["RoundNumber"] == args.round) & (df["Driver"] == args.driver)].copy()
    if race_df.empty:
        raise ValueError("No laps found for that round/driver. Check driver code (e.g., VER, HAM).")

    race_df = race_df.sort_values("LapNumber").reset_index(drop=True)
    total_laps = int(race_df["LapNumber"].max())

    model_path = Path(args.models_dir) / f"{args.model}_model.joblib"
    model = joblib.load(model_path)

    if args.pit_loss.lower() != "auto":
        pit_loss = float(args.pit_loss)
    else:
        metrics_path = Path(args.metrics_dir) / f"pit_loss_{args.season}.csv"
        pit_loss = load_pit_loss(metrics_path, args.round)
        if pit_loss is None:
            raw_path = list((Path(args.raw_dir) / str(args.season)).glob(f"round_{args.round:02d}_*/laps.parquet"))
            if raw_path:
                raw_df = pd.read_parquet(raw_path[0])
                pit_loss = estimate_pit_loss_from_raw(raw_df)
        if pit_loss is None:
            pit_loss = 20.0

    if args.strategy:
        strategies = {"custom": parse_strategy(args.strategy)}
    else:
        compounds = available_compounds(race_df, args.include_wet)
        strategies = generate_strategies(
            total_laps,
            compounds,
            max_stops=args.max_stops,
            min_stint=args.min_stint,
            max_stint=args.max_stint,
            step=args.stint_step,
            require_two_compounds=not args.allow_single_compound,
        )

    results = []
    target_is_delta = "LapTimeDelta" in race_df.columns and "RaceMedianLap" in race_df.columns
    race_median = race_df["RaceMedianLap"].iloc[0] if "RaceMedianLap" in race_df.columns else race_df["LapTimeSeconds"].median()

    for name, strategy in strategies.items():
        laps = build_laps(race_df, strategy)
        X = laps.drop(columns=["LapTimeSeconds", "LapTimeDelta"], errors="ignore")
        preds = model.predict(X)
        if target_is_delta:
            preds = preds + race_median
        total_time = float(np.sum(preds)) + pit_loss * (len(strategy) - 1)
        results.append(
            {
                "strategy": name,
                "total_time_s": total_time,
                "pit_loss_s": pit_loss,
                "stops": len(strategy) - 1,
                "stints": strategy,
            }
        )

    out = pd.DataFrame(results).sort_values("total_time_s").head(args.top_n)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
