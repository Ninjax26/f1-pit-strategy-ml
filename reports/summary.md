# 2024 F1 Pit Strategy ML + Simulation

## Overview
- Season: 2024 (main races only)
- Goal: predict lap time and simulate pit strategies

## Data
- Source: FastF1
- Races: All 2024 main races

## Modeling
- Baseline: Ridge regression
- Tree model: HistGradientBoostingRegressor

## Metrics
- HGB: MAE 1.751s, RMSE 2.795s
- Ridge: MAE 3.175s, RMSE 3.978s
- Absolute times use a prior-season circuit baseline, not the completed target race median

## Strategy Simulation
- Example races + drivers
- Compare predicted total time for candidate strategies

## Key Findings
- HGB is the stronger within-season model on the Rounds 17–24 holdout.
- Cross-season HGB MAE rises to 3.802s, showing material distribution shift.
- Strategy rankings penalize tyre life beyond prior-season historical support.
