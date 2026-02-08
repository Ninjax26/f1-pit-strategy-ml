import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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


def estimate_pit_loss(df: pd.DataFrame) -> dict:
    if "LapTime" not in df.columns:
        return {
            "pit_loss_median": np.nan,
            "pit_loss_mean": np.nan,
            "pit_loss_std": np.nan,
            "pit_loss_p10": np.nan,
            "pit_loss_p90": np.nan,
            "n_events": 0,
            "base_lap_median": np.nan,
        }

    df = df.copy()
    df["LapTimeSeconds"] = _to_seconds(df["LapTime"])

    # Clean laps for baseline pace
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
    if clean.empty or clean["LapTimeSeconds"].dropna().empty:
        clean = df[df["LapTimeSeconds"].notna()].copy()

    losses = []

    if "Driver" not in df.columns or "LapNumber" not in df.columns:
        base = clean["LapTimeSeconds"].median()
        in_laps = df[df["PitInTime"].notna()] if "PitInTime" in df.columns else df.iloc[0:0]
        out_laps = df[df["PitOutTime"].notna()] if "PitOutTime" in df.columns else df.iloc[0:0]
        losses.extend(list(in_laps["LapTimeSeconds"] - base))
        losses.extend(list(out_laps["LapTimeSeconds"] - base))
    else:
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

    losses = _robust_filter(losses)
    if not losses:
        base_median = float(clean["LapTimeSeconds"].median()) if not clean.empty else np.nan
        return {
            "pit_loss_median": np.nan,
            "pit_loss_mean": np.nan,
            "pit_loss_std": np.nan,
            "pit_loss_p10": np.nan,
            "pit_loss_p90": np.nan,
            "n_events": 0,
            "base_lap_median": base_median,
        }

    losses_arr = np.array(losses, dtype=float)
    loss_std = float(np.std(losses_arr, ddof=1)) if len(losses_arr) > 1 else 0.0
    return {
        "pit_loss_median": float(np.median(losses_arr)),
        "pit_loss_mean": float(np.mean(losses_arr)),
        "pit_loss_std": loss_std,
        "pit_loss_p10": float(np.quantile(losses_arr, 0.10)),
        "pit_loss_p90": float(np.quantile(losses_arr, 0.90)),
        "n_events": int(len(losses_arr)),
        "base_lap_median": float(clean["LapTimeSeconds"].median()) if not clean.empty else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pit loss distributions per 2024 race")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/metrics")
    args = parser.parse_args()

    raw_root = Path(args.raw_dir) / str(args.season)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(raw_root.glob("round_*_*/laps.parquet")):
        df = pd.read_parquet(path)
        event_name = df["EventName"].iloc[0] if "EventName" in df.columns else path.parent.name
        round_number = int(df["RoundNumber"].iloc[0]) if "RoundNumber" in df.columns else None
        stats = estimate_pit_loss(df)
        rows.append(
            {
                "season": args.season,
                "round": round_number,
                "event": event_name,
                **stats,
            }
        )

    out = pd.DataFrame(rows).sort_values("round")
    out_path = out_root / f"pit_loss_{args.season}.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
