import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def load_pit_loss(metrics_path: Path, round_number: int) -> float | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    row = df[df["round"] == round_number]
    if row.empty:
        return None
    value = row.iloc[0].get("pit_loss_median")
    return None if pd.isna(value) else float(value)


def actual_strategy(race_df: pd.DataFrame) -> list[dict]:
    rows = []
    for stint, grp in race_df.groupby("Stint"):
        compound = str(grp["Compound"].mode().iloc[0]) if "Compound" in grp.columns else "UNKNOWN"
        rows.append(
            {
                "stint": int(stint),
                "compound": compound,
                "laps": int(len(grp)),
                "lap_start": int(grp["LapNumber"].min()),
                "lap_end": int(grp["LapNumber"].max()),
            }
        )
    return sorted(rows, key=lambda x: x["stint"])


def make_plot(race_df: pd.DataFrame, preds: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(race_df["LapNumber"], race_df["LapTimeSeconds"], label="Actual", linewidth=1.5)
    plt.plot(race_df["LapNumber"], preds, label="Predicted", linewidth=1.5, alpha=0.8)

    if "IsPitLap" in race_df.columns:
        pit_laps = race_df[race_df["IsPitLap"]]["LapNumber"].tolist()
        for lap in pit_laps:
            plt.axvline(lap, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

    plt.title("Max Verstappen Case Study: Actual vs Predicted Lap Time")
    plt.xlabel("Lap")
    plt.ylabel("Lap Time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Max Verstappen case study report")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--round", type=int, default=14)
    parser.add_argument("--driver", type=str, default="VER")
    parser.add_argument("--features", type=str, default="data/features")
    parser.add_argument("--models-dir", type=str, default="data/models")
    parser.add_argument("--metrics-dir", type=str, default="data/metrics")
    parser.add_argument("--model", type=str, default="hgb", choices=["ridge", "hgb"])
    parser.add_argument("--out-report", type=str, default="reports/case_study_max_round14.md")
    parser.add_argument("--out-plot", type=str, default="figures/case_study_max_round14.png")
    parser.add_argument("--max-stops", type=int, default=2)
    parser.add_argument("--min-stint", type=int, default=8)
    parser.add_argument("--max-stint", type=int, default=35)
    parser.add_argument("--stint-step", type=int, default=2)
    parser.add_argument("--include-wet", action="store_true")
    parser.add_argument("--allow-single-compound", action="store_true")
    parser.add_argument("--top-n", type=int, default=5)
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

    X = race_df.drop(columns=["LapTimeSeconds", "LapTimeDelta"], errors="ignore")
    preds = model.predict(X)
    if "LapTimeDelta" in race_df.columns and "RaceMedianLap" in race_df.columns:
        preds = preds + race_df["RaceMedianLap"].to_numpy()

    actual_total = float(race_df["LapTimeSeconds"].sum())
    pred_total = float(np.sum(preds))

    pit_loss = load_pit_loss(Path(args.metrics_dir) / f"pit_loss_{args.season}.csv", args.round)
    pit_loss = pit_loss if pit_loss is not None else 20.0

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

    best = None
    for name, strategy in strategies.items():
        sim_laps = build_laps(race_df, strategy)
        X_sim = sim_laps.drop(columns=["LapTimeSeconds", "LapTimeDelta"], errors="ignore")
        sim_preds = model.predict(X_sim)
        if "LapTimeDelta" in sim_laps.columns and "RaceMedianLap" in sim_laps.columns:
            sim_preds = sim_preds + sim_laps["RaceMedianLap"].to_numpy()
        total_time = float(np.sum(sim_preds)) + pit_loss * (len(strategy) - 1)
        record = {"name": name, "total_time_s": total_time, "strategy": strategy}
        if best is None or total_time < best["total_time_s"]:
            best = record

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    actual_stints = actual_strategy(race_df)

    report_lines = []
    report_lines.append(f"# Max Verstappen Case Study (Round {args.round}, {args.season})")
    report_lines.append("")
    report_lines.append("## Actual Strategy")
    for stint in actual_stints:
        report_lines.append(
            f"- Stint {stint['stint']}: {stint['compound']} | laps {stint['lap_start']}-{stint['lap_end']} (n={stint['laps']})"
        )

    report_lines.append("")
    report_lines.append("## Model Summary")
    report_lines.append(f"- Model: {args.model}")
    report_lines.append(f"- Actual total race time (s): {actual_total:.2f}")
    report_lines.append(f"- Predicted total race time (s): {pred_total:.2f}")
    report_lines.append(f"- Pit-loss (auto, median) (s): {pit_loss:.2f}")

    if best is not None:
        report_lines.append("")
        report_lines.append("## Best Simulated Strategy")
        report_lines.append(f"- {best['name']} | total predicted time (s): {best['total_time_s']:.2f}")
        report_lines.append(f"- Stints: {best['strategy']}")

    out_report.write_text("\n".join(report_lines))

    out_plot = Path(args.out_plot)
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    make_plot(race_df, preds, out_plot)

    print(f"Wrote {out_report}")
    print(f"Wrote {out_plot}")


if __name__ == "__main__":
    main()
