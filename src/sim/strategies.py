"""Shared strategy parsing, validation, generation, and lap construction."""

import pandas as pd


DRY_COMPOUNDS = {"SOFT", "MEDIUM", "HARD"}


def parse_strategy(spec: str) -> list[tuple[str, int]]:
    if not spec or not spec.strip():
        raise ValueError("Strategy cannot be empty")
    stints = []
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if ":" not in part:
            raise ValueError(f"Invalid stint '{part}'. Expected COMPOUND:LAPS")
        compound, raw_length = part.split(":", 1)
        compound = compound.strip().upper()
        if not compound:
            raise ValueError("Compound cannot be empty")
        try:
            length = int(raw_length.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid lap count '{raw_length}' for {compound}") from exc
        stints.append((compound, length))
    return stints


def validate_strategy(
    strategy: list[tuple[str, int]],
    total_laps: int,
    *,
    min_stint: int = 1,
    max_stint: int | None = None,
    allowed_compounds: list[str] | None = None,
    require_two_compounds: bool = False,
) -> None:
    if not strategy:
        raise ValueError("Strategy must contain at least one stint")
    if total_laps <= 0:
        raise ValueError("Race distance must be positive")

    allowed = {str(c).upper() for c in allowed_compounds} if allowed_compounds else None
    compounds = []
    lap_sum = 0
    for compound, length in strategy:
        compound = str(compound).upper()
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError(f"{compound} stint length must be a positive integer")
        if length < min_stint:
            raise ValueError(f"{compound} stint has {length} laps; minimum is {min_stint}")
        if max_stint is not None and length > max_stint:
            raise ValueError(f"{compound} stint has {length} laps; maximum is {max_stint}")
        if allowed is not None and compound not in allowed:
            raise ValueError(f"Compound {compound} is unavailable for this race")
        compounds.append(compound)
        lap_sum += length

    if lap_sum != total_laps:
        raise ValueError(f"Strategy covers {lap_sum} laps but race distance is {total_laps}")
    if require_two_compounds and set(compounds).issubset(DRY_COMPOUNDS) and len(set(compounds)) < 2:
        raise ValueError("Dry strategies must use at least two compounds")


def available_compounds(race_df: pd.DataFrame, include_wet: bool) -> list[str]:
    compounds = sorted(set(race_df.get("Compound", pd.Series(dtype=str)).dropna().astype(str).str.upper()))
    dry = [c for c in compounds if c in DRY_COMPOUNDS]
    wet = [c for c in compounds if c not in DRY_COMPOUNDS and c not in {"NAN", "NONE", "UNKNOWN"}]
    if include_wet:
        return dry + wet if dry else wet
    return dry if dry else wet


def generate_strategies(
    total_laps: int,
    compounds: list[str],
    max_stops: int,
    min_stint: int,
    max_stint: int,
    step: int,
    require_two_compounds: bool,
) -> dict[str, list[tuple[str, int]]]:
    if step <= 0:
        raise ValueError("Stint step must be positive")
    max_stint = total_laps if max_stint <= 0 else max_stint
    strategies = {}

    if max_stops == 0:
        for compound in compounds:
            strategy = [(compound, total_laps)]
            try:
                validate_strategy(
                    strategy, total_laps, min_stint=min_stint, max_stint=max_stint,
                    allowed_compounds=compounds, require_two_compounds=require_two_compounds,
                )
            except ValueError:
                continue
            strategies[f"0stop_{compound[0]}_{total_laps}"] = strategy

    if max_stops >= 1:
        for c1 in compounds:
            for c2 in compounds:
                for s1 in range(min_stint, total_laps - min_stint + 1, step):
                    strategy = [(c1, s1), (c2, total_laps - s1)]
                    try:
                        validate_strategy(
                            strategy, total_laps, min_stint=min_stint, max_stint=max_stint,
                            allowed_compounds=compounds, require_two_compounds=require_two_compounds,
                        )
                    except ValueError:
                        continue
                    strategies[f"1stop_{c1[0]}-{c2[0]}_{strategy[0][1]}-{strategy[1][1]}"] = strategy

    if max_stops >= 2:
        for c1 in compounds:
            for c2 in compounds:
                for c3 in compounds:
                    for s1 in range(min_stint, total_laps - 2 * min_stint + 1, step):
                        for s2 in range(min_stint, total_laps - s1 - min_stint + 1, step):
                            strategy = [(c1, s1), (c2, s2), (c3, total_laps - s1 - s2)]
                            try:
                                validate_strategy(
                                    strategy, total_laps, min_stint=min_stint, max_stint=max_stint,
                                    allowed_compounds=compounds, require_two_compounds=require_two_compounds,
                                )
                            except ValueError:
                                continue
                            lengths = "-".join(str(length) for _, length in strategy)
                            strategies[f"2stop_{c1[0]}-{c2[0]}-{c3[0]}_{lengths}"] = strategy
    return strategies


def build_laps(base_laps: pd.DataFrame, strategy: list[tuple[str, int]]) -> pd.DataFrame:
    validate_strategy(strategy, len(base_laps))
    laps = base_laps.copy().reset_index(drop=True)
    lap_idx = 0
    for stint_idx, (compound, length) in enumerate(strategy, start=1):
        end = lap_idx + length
        laps.loc[lap_idx:end - 1, "Compound"] = compound
        if "TyreLife" in laps.columns:
            laps.loc[lap_idx:end - 1, "TyreLife"] = range(1, length + 1)
        if "Stint" in laps.columns:
            laps.loc[lap_idx:end - 1, "Stint"] = stint_idx
        lap_idx = end
    return laps
