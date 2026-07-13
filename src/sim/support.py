"""Historical-support profiles for conservative counterfactual strategies."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _summarize(grouped, quantile: float) -> dict:
    result = {}
    for key, group in grouped:
        tyre_life = pd.to_numeric(group["TyreLife"], errors="coerce").dropna()
        if tyre_life.empty:
            continue
        result[str(key).upper()] = {
            "n_laps": int(len(tyre_life)),
            "max_supported_tyre_life": int(np.ceil(tyre_life.quantile(quantile))),
            "observed_max_tyre_life": int(tyre_life.max()),
        }
    return result


def build_support_profile(history: pd.DataFrame, target_season: int, quantile: float = 0.99) -> dict:
    required = {"Season", "EventName", "Compound", "TyreLife"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Missing support columns: {sorted(missing)}")

    clean = history[history["Season"] < target_season].copy()
    if "IsPitLap" in clean.columns:
        clean = clean[~clean["IsPitLap"]]
    if "IsSafetyCar" in clean.columns:
        clean = clean[~clean["IsSafetyCar"]]
    clean = clean.dropna(subset=["EventName", "Compound", "TyreLife"])

    events = {}
    for event_name, event_df in clean.groupby("EventName"):
        events[str(event_name)] = _summarize(event_df.groupby("Compound"), quantile)

    return {
        "target_season": target_season,
        "source_seasons": sorted(clean["Season"].dropna().astype(int).unique().tolist()),
        "support_quantile": quantile,
        "global": _summarize(clean.groupby("Compound"), quantile),
        "events": events,
    }


def assess_strategy_support(strategy: list[tuple[str, int]], event_name: str, profile: dict | None) -> dict:
    if not profile:
        return {"unsupported_laps": 0, "support_source": "unavailable", "support_details": []}

    event_limits = profile.get("events", {}).get(str(event_name), {})
    global_limits = profile.get("global", {})
    unsupported_laps = 0
    details = []
    sources = set()

    for compound, length in strategy:
        compound = str(compound).upper()
        support = event_limits.get(compound)
        source = "event history"
        if support is None:
            support = global_limits.get(compound)
            source = "global history"
        if support is None:
            unsupported_laps += int(length)
            details.append(f"{compound}: no historical support")
            sources.add("missing")
            continue

        limit = int(support["max_supported_tyre_life"])
        excess = max(0, int(length) - limit)
        unsupported_laps += excess
        sources.add(source)
        details.append(f"{compound}: {length} laps vs {limit}-lap {source} limit")

    return {
        "unsupported_laps": unsupported_laps,
        "support_source": ", ".join(sorted(sources)),
        "support_details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical tyre-life support limits")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--features-dir", default="data/features")
    parser.add_argument("--metrics-dir", default="data/metrics")
    parser.add_argument("--quantile", type=float, default=0.99)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    frames = []
    for path in sorted(features_dir.glob("features_*.parquet")):
        suffix = path.stem.removeprefix("features_")
        if suffix.isdigit() and int(suffix) < args.season:
            frame = pd.read_parquet(path)
            if "Season" not in frame.columns:
                frame["Season"] = int(suffix)
            frames.append(frame)
    if not frames:
        raise RuntimeError("No prior-season features are available for support profiling")

    profile = build_support_profile(pd.concat(frames, ignore_index=True), args.season, args.quantile)
    output = Path(args.metrics_dir) / f"strategy_support_{args.season}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
