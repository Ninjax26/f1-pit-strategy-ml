"""Regression tests for strategy and Monte Carlo simulation behavior."""

import numpy as np
import pandas as pd
import pytest

import app
from src.sim.strategies import build_laps, generate_strategies, parse_strategy, validate_strategy
from src.sim.uncertainty import sample_residual_series


def test_generated_strategies_enforce_every_stint_limit():
    strategies = generate_strategies(
        total_laps=30,
        compounds=["SOFT", "MEDIUM", "HARD"],
        max_stops=2,
        min_stint=5,
        max_stint=15,
        step=2,
        require_two_compounds=True,
    )
    assert strategies
    for strategy in strategies.values():
        validate_strategy(
            strategy,
            30,
            min_stint=5,
            max_stint=15,
            allowed_compounds=["SOFT", "MEDIUM", "HARD"],
            require_two_compounds=True,
        )
        assert all(5 <= length <= 15 for _, length in strategy)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("SOFT", "Expected COMPOUND:LAPS"),
        ("SOFT:abc", "Invalid lap count"),
        ("SOFT:10,HARD:-1", "positive integer"),
        ("SOFT:10,HARD:8", "covers 18 laps"),
    ],
)
def test_custom_strategy_validation_errors(spec, message):
    if ":" not in spec or "abc" in spec:
        with pytest.raises(ValueError, match=message):
            parse_strategy(spec)
        return
    strategy = parse_strategy(spec)
    with pytest.raises(ValueError, match=message):
        validate_strategy(strategy, 20, allowed_compounds=["SOFT", "HARD"])


def test_build_laps_rejects_partial_strategy():
    laps = pd.DataFrame({"Compound": ["MEDIUM"] * 10, "TyreLife": [1] * 10, "Stint": [1] * 10})
    with pytest.raises(ValueError, match="covers 8 laps"):
        build_laps(laps, [("MEDIUM", 4), ("HARD", 4)])


def test_residual_sampler_uses_contiguous_blocks():
    sequence = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    residuals = {
        "global_sequences": [sequence],
        "by_compound": {"HARD": [sequence]},
        "global_values": sequence,
    }
    sampled = sample_residual_series(["HARD"] * 10, residuals, np.random.default_rng(42), block_size=5)
    assert sampled.tolist() == sequence.tolist() * 2


def test_race_template_preserves_lap_varying_weather():
    features = pd.read_parquet("data/features/features_2024.parquet")
    round_df = features[features["RoundNumber"] == 24]
    template = app.build_race_template(round_df, "VER", int(round_df["LapNumber"].max()), 2024)
    assert template["TrackTemp"].nunique() > 1
    assert template["AirTemp"].notna().all()


def test_fixed_pit_loss_uses_displayed_median():
    stats = {"median": 21.0, "mean": 27.0, "std": 3.0, "p10": 18.0, "p90": 30.0}
    assert app.sample_pit_loss(stats, np.random.default_rng(42), "fixed") == 21.0
