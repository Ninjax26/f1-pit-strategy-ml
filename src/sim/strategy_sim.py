import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(val) else val


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


def _filter_losses(losses: list[float]) -> list[float]:
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

    losses = _filter_losses(losses)
    if not losses:
        return None

    return float(np.median(losses))


def _fixed_pit_loss_stats(value: float) -> dict:
    return {
        "median": value,
        "mean": value,
        "std": 0.0,
        "p10": value,
        "p90": value,
    }


def load_pit_loss(metrics_path: Path, round_number: int) -> dict | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    row = df[df["round"] == round_number]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "median": _safe_float(row.get("pit_loss_median")),
        "mean": _safe_float(row.get("pit_loss_mean")),
        "std": _safe_float(row.get("pit_loss_std")),
        "p10": _safe_float(row.get("pit_loss_p10")),
        "p90": _safe_float(row.get("pit_loss_p90")),
    }


def sample_pit_loss(stats: dict | None, rng: np.random.Generator, mode: str) -> float:
    if stats is None:
        return 20.0
    median = stats.get("median")
    mean = stats.get("mean")
    std = stats.get("std")
    p10 = stats.get("p10")
    p90 = stats.get("p90")

    center = mean if mean is not None else (median if median is not None else 20.0)
    if mode == "fixed":
        return center

    if std is None or std <= 0:
        if p10 is not None and p90 is not None and p90 > p10:
            std = (p90 - p10) / 2.563
        else:
            std = 2.0

    value = float(rng.normal(center, std))
    return float(np.clip(value, 5.0, 60.0))


def load_residuals(residuals_path: Path) -> dict | None:
    if not residuals_path.exists():
        return None
    df = pd.read_parquet(residuals_path)
    if "residual" not in df.columns:
        return None
    df = df[pd.notna(df["residual"])].copy()
    if df.empty:
        return None

    q01, q99 = np.quantile(df["residual"].to_numpy(), [0.01, 0.99])
    df = df[(df["residual"] >= q01) & (df["residual"] <= q99)]
    if df.empty:
        return None

    global_residuals = df["residual"].to_numpy()
    by_compound: dict[str, np.ndarray] = {}
    if "Compound" in df.columns:
        for comp, grp in df.groupby("Compound"):
            arr = grp["residual"].dropna().to_numpy()
            if len(arr) >= 50:
                by_compound[str(comp).upper()] = arr

    return {"global": global_residuals, "by_compound": by_compound}


def sample_residual(compound: str, residuals: dict | None, rng: np.random.Generator) -> float:
    if residuals is None:
        return 0.0
    comp = str(compound).upper()
    arr = residuals.get("by_compound", {}).get(comp)
    if arr is None or len(arr) == 0:
        arr = residuals.get("global")
    if arr is None or len(arr) == 0:
        return 0.0
    return float(rng.choice(arr))


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
    parser.add_argument("--n-sims", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--residuals", type=str, default=None)
    parser.add_argument("--noise-sigma", type=float, default=None)
    parser.add_argument("--pit-loss-mode", type=str, default="sample", choices=["sample", "fixed"])
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
        pit_loss_stats = _fixed_pit_loss_stats(float(args.pit_loss))
    else:
        metrics_path = Path(args.metrics_dir) / f"pit_loss_{args.season}.csv"
        pit_loss_stats = load_pit_loss(metrics_path, args.round)
        if pit_loss_stats is None:
            raw_path = list((Path(args.raw_dir) / str(args.season)).glob(f"round_{args.round:02d}_*/laps.parquet"))
            if raw_path:
                raw_df = pd.read_parquet(raw_path[0])
                fallback = estimate_pit_loss_from_raw(raw_df)
                if fallback is not None:
                    pit_loss_stats = _fixed_pit_loss_stats(float(fallback))
        if pit_loss_stats is None:
            pit_loss_stats = _fixed_pit_loss_stats(20.0)

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

    rng = np.random.default_rng(args.seed)
    residuals = None
    if args.n_sims > 1 or args.noise_sigma is not None:
        if args.residuals:
            residuals = load_residuals(Path(args.residuals))
        else:
            default_residuals = Path(args.metrics_dir) / f"predictions_{args.model}.parquet"
            residuals = load_residuals(default_residuals)

    pit_loss_mode = args.pit_loss_mode
    if args.n_sims <= 1:
        pit_loss_mode = "fixed"

    results = []
    target_is_delta = "LapTimeDelta" in race_df.columns and "RaceMedianLap" in race_df.columns
    race_median = race_df["RaceMedianLap"].iloc[0] if "RaceMedianLap" in race_df.columns else race_df["LapTimeSeconds"].median()

    for name, strategy in strategies.items():
        laps = build_laps(race_df, strategy)
        X = laps.drop(columns=["LapTimeSeconds", "LapTimeDelta"], errors="ignore")
        base_preds = model.predict(X)
        if target_is_delta:
            base_preds = base_preds + race_median

        n_stops = len(strategy) - 1
        if args.n_sims <= 1:
            pit_loss_value = sample_pit_loss(pit_loss_stats, rng, pit_loss_mode)
            total_time = float(np.sum(base_preds)) + pit_loss_value * n_stops
            results.append(
                {
                    "strategy": name,
                    "total_time_s": total_time,
                    "pit_loss_s": pit_loss_value,
                    "stops": n_stops,
                    "stints": strategy,
                }
            )
        else:
            totals: list[float] = []
            pit_losses: list[float] = []
            compounds = laps.get("Compound", pd.Series([None] * len(laps)))
            for _ in range(args.n_sims):
                lap_preds = np.array(base_preds, dtype=float)
                if residuals is not None:
                    noise = np.array([sample_residual(c, residuals, rng) for c in compounds])
                    lap_preds = lap_preds + noise
                elif args.noise_sigma is not None and args.noise_sigma > 0:
                    lap_preds = lap_preds + rng.normal(0.0, args.noise_sigma, size=lap_preds.shape[0])

                pit_loss_total = sum(sample_pit_loss(pit_loss_stats, rng, pit_loss_mode) for _ in range(n_stops))
                pit_losses.append(pit_loss_total)
                totals.append(float(np.sum(lap_preds)) + pit_loss_total)

            totals_arr = np.array(totals, dtype=float)
            results.append(
                {
                    "strategy": name,
                    "total_time_mean_s": float(np.mean(totals_arr)),
                    "total_time_p10_s": float(np.quantile(totals_arr, 0.10)),
                    "total_time_p50_s": float(np.quantile(totals_arr, 0.50)),
                    "total_time_p90_s": float(np.quantile(totals_arr, 0.90)),
                    "pit_loss_mean_s": float(np.mean(pit_losses)) if pit_losses else 0.0,
                    "stops": n_stops,
                    "stints": strategy,
                }
            )

    if args.n_sims <= 1:
        out = pd.DataFrame(results).sort_values("total_time_s").head(args.top_n)
    else:
        out = pd.DataFrame(results).sort_values("total_time_mean_s").head(args.top_n)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
