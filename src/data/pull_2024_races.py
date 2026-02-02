import argparse
import re
from pathlib import Path

import fastf1
import pandas as pd


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull 2024 F1 race data with FastF1")
    parser.add_argument("--season", type=int, default=2024)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="data/raw")
    args = parser.parse_args()

    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(args.cache_dir)

    schedule = fastf1.get_event_schedule(args.season)

    out_root = Path(args.out_dir) / str(args.season)
    out_root.mkdir(parents=True, exist_ok=True)

    for _, event in schedule.iterrows():
        event_name = event.get("EventName")
        round_number = int(event.get("RoundNumber")) if pd.notna(event.get("RoundNumber")) else None
        if not event_name or round_number is None:
            continue

        print(f"Loading {args.season} Round {round_number}: {event_name}")
        session = fastf1.get_session(args.season, event_name, "R")
        session.load(laps=True, telemetry=False, weather=True)

        laps = session.laps.reset_index(drop=True)
        weather = laps.get_weather_data().reset_index(drop=True)

        if "Time" in weather.columns:
            weather = weather.drop(columns=["Time"])

        df = pd.concat([laps, weather], axis=1)
        df["EventName"] = event_name
        df["RoundNumber"] = round_number
        df["SessionName"] = session.name

        slug = slugify(event_name)
        out_dir = out_root / f"round_{round_number:02d}_{slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "laps.parquet"
        df.to_parquet(out_path, index=False)

    print("Done.")


if __name__ == "__main__":
    main()
