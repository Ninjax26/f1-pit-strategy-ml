# Multi-Season Generalization Testing

This evaluation measures absolute lap-time performance without using the completed target race as its own baseline. Each supported race uses the latest available earlier-season median for the same circuit as `PreRaceBaseline`; `Season`, baseline columns, targets, pit laps, and safety-car laps are excluded from model inputs.

## Experimental Design

1. **Within-season baseline:** train on 2024 Rounds 1–16 and test on Rounds 17–24.
2. **Cross-season evaluation:** train on supported laps from 2022–2024 and test on supported 2025 laps.
3. **Regulation-shift test:** intentionally skipped because 2021 has no earlier-season data in this repository. Running it with the completed 2021 race median would reintroduce the leakage this evaluation is designed to remove.

## Results

| Experiment | Train Set | Test Set | Model | MAE | RMSE | Test Laps | Unseen Track Rows |
|---|---|---|---|---|---|---|---|
| Baseline 2024 | 2024 Rounds 1–16 | 2024 Rounds 17–24 | `hgb` | **1.751s** | **2.795s** | 7,454 | 7,454 |
| Baseline 2024 | 2024 Rounds 1–16 | 2024 Rounds 17–24 | `ridge` | 3.175s | 3.978s | 7,454 | 7,454 |
| Cross-season | 2022+2023+2024 | 2025 | `hgb` | **3.802s** | 6.014s | 24,747 | 0 |
| Cross-season | 2022+2023+2024 | 2025 | `ridge` | 3.987s | 5.465s | 24,747 | 0 |

## Interpretation

The within-season HGB result remains useful, but it is weaker than the previous oracle-baseline result because the model must now absorb year-to-year circuit pace changes. The cross-season MAE rises to 3.802 seconds, showing meaningful distribution shift even within the same regulation era. This does not prove that the model learned complete tyre or fuel physics; it shows that historical relationships transfer only partially.

All 2024 holdout events are unseen categorical event names because the split is by race round. `OneHotEncoder(handle_unknown="ignore")` handles those names operationally, but the model lacks explicit circuit geometry, surface, and corner-profile features. The 2025 test has no unseen event names because those circuits appeared in the 2022–2024 training seasons.
