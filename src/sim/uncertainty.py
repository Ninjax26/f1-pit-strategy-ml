"""Residual loading and moving-block bootstrap for race simulations."""

from pathlib import Path

import numpy as np
import pandas as pd


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

    sort_cols = [c for c in ["RoundNumber", "Driver", "Stint", "LapNumber"] if c in df.columns]
    group_cols = [c for c in ["RoundNumber", "Driver", "Stint"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)

    global_sequences = []
    by_compound: dict[str, list[np.ndarray]] = {}
    groups = df.groupby(group_cols, dropna=False) if group_cols else [(None, df)]
    for _, group in groups:
        values = group["residual"].to_numpy(dtype=float)
        if len(values) >= 2:
            global_sequences.append(values)
        if "Compound" in group.columns:
            for compound, compound_group in group.groupby("Compound"):
                sequence = compound_group["residual"].to_numpy(dtype=float)
                if len(sequence) >= 2:
                    by_compound.setdefault(str(compound).upper(), []).append(sequence)

    return {
        "global_sequences": global_sequences,
        "by_compound": by_compound,
        "global_values": df["residual"].to_numpy(dtype=float),
    }


def _draw_block(sequences: list[np.ndarray], fallback: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    eligible = [sequence for sequence in sequences if len(sequence) >= 2]
    if not eligible:
        return rng.choice(fallback, size=size, replace=True)
    sequence = eligible[int(rng.integers(0, len(eligible)))]
    block_length = min(size, len(sequence))
    start = int(rng.integers(0, len(sequence) - block_length + 1))
    return sequence[start:start + block_length]


def sample_residual_series(
    compounds: pd.Series | list,
    residuals: dict | None,
    rng: np.random.Generator,
    block_size: int = 5,
) -> np.ndarray:
    compounds = [str(compound).upper() for compound in compounds]
    if residuals is None or not compounds:
        return np.zeros(len(compounds), dtype=float)

    fallback = residuals.get("global_values")
    if fallback is None or len(fallback) == 0:
        return np.zeros(len(compounds), dtype=float)

    sampled = np.empty(len(compounds), dtype=float)
    cursor = 0
    while cursor < len(compounds):
        compound = compounds[cursor]
        segment_end = cursor + 1
        while segment_end < len(compounds) and compounds[segment_end] == compound:
            segment_end += 1
        segment_cursor = cursor
        sequences = residuals.get("by_compound", {}).get(compound, residuals.get("global_sequences", []))
        while segment_cursor < segment_end:
            requested = min(block_size, segment_end - segment_cursor)
            block = _draw_block(sequences, fallback, requested, rng)
            sampled[segment_cursor:segment_cursor + len(block)] = block
            segment_cursor += len(block)
        cursor = segment_end
    return sampled
