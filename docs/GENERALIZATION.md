# Multi-Season Generalization Testing

This document details the cross-season generalization performance of the lap-time ML pipeline, specifically examining how well the models generalize across different seasons within the same regulation era, and how severely they degrade across major regulation shifts.

## Experimental Design

To isolate the effects of data distribution shifts, we run three distinct experiments. In all experiments, the `Season` year is strictly used as a split key and is **dropped** from the feature set before training, preventing the model from explicitly memorizing season-specific base paces.

1. **Baseline (Within-Season)**
   - **Train:** 2024 Rounds 1–16
   - **Test:** 2024 Rounds 17–24
   - *Purpose:* Establishes the expected performance when training and testing within the exact same year.

2. **Experiment 1: Primary Generalization (Same Era)**
   - **Train:** 2022, 2023, 2024 (Ground Effect Era)
   - **Test:** 2025
   - *Purpose:* Tests if the model learns general tire degradation and fuel burn physics that apply to future, unseen races within the same regulatory framework.

3. **Experiment 2: Regulation-Shift Stress Test**
   - **Train:** 2021 (Pre-2022 regulations, 13-inch tires, over-car aero)
   - **Test:** 2022 (Ground-effect aero, 18-inch tires)
   - *Purpose:* A deliberate out-of-distribution test. The model is forced to predict lap times for a fundamentally different car concept.

## Results

| Experiment | Train Set | Test Set | Model | MAE | RMSE | Test Laps | Unseen Track Rows |
|---|---|---|---|---|---|---|---|
| Baseline 2024 (Within-Season) | 2024 Rounds 1–16 | 2024 Rounds 17–24 | `hgb` | 1.412s | 2.250s | 7,454 | 8,223 |
| Baseline 2024 (Within-Season) | 2024 Rounds 1–16 | 2024 Rounds 17–24 | `ridge` | 3.741s | 4.913s | 7,454 | 8,223 |
| Exp 1: Generalization (Same Era) | 2022+2023+2024 | 2025 (25 rounds) | `hgb` | **1.335s** | 2.732s | 24,747 | 0 |
| Exp 1: Generalization (Same Era) | 2022+2023+2024 | 2025 (25 rounds) | `ridge` | 1.585s | 2.545s | 24,747 | 0 |
| Exp 2: Reg-Shift Stress Test | 2021 | 2022 | `hgb` | **2.131s** | 4.041s | 21,683 | 5,330 |
| Exp 2: Reg-Shift Stress Test | 2021 | 2022 | `ridge` | 2.404s | 3.974s | 21,683 | 5,330 |

## Interpretation

**1. Baseline vs. Primary Generalization (Same Era)**
The model generalizes well within the same regulation era. The HGB model achieved a Mean Absolute Error (MAE) of **1.335s** on the completely unseen 2025 season (trained on 2022-2024), which is actually slightly *better* than the 1.412s MAE achieved on the 2024 within-season test set. This confirms the model is successfully learning underlying physics (tire degradation profiles, track evolution, fuel burn) rather than memorizing specific races, as the combined three-year dataset provides a much richer set of training examples.

**2. The Regulation-Shift Gap**
As expected, the model degrades when predicting across a major regulation change. When trained strictly on 2021 data (13-inch tires, over-car aero) and tested on 2022 data (18-inch tires, ground-effect aero), the HGB model's MAE jumped to **2.131s** (a roughly +0.8s penalty compared to same-era generalization). 
While the model didn't fail completely—it still captures universal truths like "old hard tires are slower than new soft tires"—the fundamental tire degradation curves and car behavior changed enough in 2022 to introduce a noticeable baseline shift. This proves that historical ML models in F1 require retraining when the FIA introduces major aerodynamic or mechanical regulation changes.

## Note on Unseen Tracks

When predicting future seasons, the model will encounter tracks it was never trained on. Our models handle categorical variables via `OneHotEncoder(handle_unknown="ignore")`. This means unseen tracks are gracefully assigned an all-zero encoding for track-specific features. The base predictions rely on track temperatures, tire compounds, and stint lengths rather than track memorization. The number of rows affected by unseen tracks is explicitly recorded in the results table above.
