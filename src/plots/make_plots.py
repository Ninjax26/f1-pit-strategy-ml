import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--metrics-dir", type=str, default="data/metrics")
    parser.add_argument("--model", type=str, default="hgb")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = metrics_dir / f"predictions_{args.model}.parquet"
    df = pd.read_parquet(pred_path)

    # Predicted vs actual
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x="LapTimeSeconds", y="pred", data=df, s=10, alpha=0.4)
    min_val = min(df["LapTimeSeconds"].min(), df["pred"].min())
    max_val = max(df["LapTimeSeconds"].max(), df["pred"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=1)
    plt.title(f"Predicted vs Actual ({args.model})")
    plt.xlabel("Actual Lap Time (s)")
    plt.ylabel("Predicted Lap Time (s)")
    plt.tight_layout()
    plt.savefig(out_dir / f"pred_vs_actual_{args.model}.png", dpi=150)
    plt.close()

    # Residuals
    plt.figure(figsize=(6, 4))
    sns.histplot(df["residual"], bins=50, kde=True)
    plt.title(f"Residual Distribution ({args.model})")
    plt.xlabel("Prediction Error (s)")
    plt.tight_layout()
    plt.savefig(out_dir / f"residuals_{args.model}.png", dpi=150)
    plt.close()

    # MAE by compound
    compound_path = metrics_dir / f"mae_by_compound_{args.model}.csv"
    if compound_path.exists():
        comp = pd.read_csv(compound_path)
        plt.figure(figsize=(6, 4))
        sns.barplot(x="Compound", y="mae", data=comp)
        plt.title(f"MAE by Compound ({args.model})")
        plt.xlabel("Compound")
        plt.ylabel("MAE (s)")
        plt.tight_layout()
        plt.savefig(out_dir / f"mae_by_compound_{args.model}.png", dpi=150)
        plt.close()

    # MAE by stint
    stint_path = metrics_dir / f"mae_by_stint_{args.model}.csv"
    if stint_path.exists():
        stint = pd.read_csv(stint_path)
        plt.figure(figsize=(6, 4))
        sns.lineplot(x="Stint", y="mae", data=stint, marker="o")
        plt.title(f"MAE by Stint ({args.model})")
        plt.xlabel("Stint")
        plt.ylabel("MAE (s)")
        plt.tight_layout()
        plt.savefig(out_dir / f"mae_by_stint_{args.model}.png", dpi=150)
        plt.close()

    # MAE by round
    round_path = metrics_dir / f"mae_by_round_{args.model}.csv"
    if round_path.exists():
        rounds = pd.read_csv(round_path)
        plt.figure(figsize=(7, 4))
        sns.lineplot(x="RoundNumber", y="mae", data=rounds, marker="o")
        plt.title(f"MAE by Round ({args.model})")
        plt.xlabel("Round")
        plt.ylabel("MAE (s)")
        plt.tight_layout()
        plt.savefig(out_dir / f"mae_by_round_{args.model}.png", dpi=150)
        plt.close()

    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
