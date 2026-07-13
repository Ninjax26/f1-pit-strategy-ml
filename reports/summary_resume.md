# F1 2024 Pit Strategy ML + Simulation (Resume Summary)

## Problem
Build an ML model to predict lap time and use it to simulate pit strategies for F1 2024 main races.

## Data
- Source: FastF1 race sessions (2024 main races)
- Features: tyre compound, tyre life, stint, lap number, weather, track status, driver/team

## Approach
- Target: `LapTimeDelta` = lap time minus the latest available prior-season circuit median
- Models: Ridge regression baseline + HistGradientBoostingRegressor (tree)
- Evaluation: time-based split (Rounds 1–16 train, 17–24 test) + rolling splits

## Results (Rounds 17–24)
- HGB MAE: **1.75s**, RMSE: **2.80s**
- Ridge MAE: **3.17s**, RMSE: **3.98s**
- Pit-loss estimated per race using robust median filtering

## Case Study (Max Verstappen, Round 14)
- Retrospective demonstration: **2-stop M→H→H**
- Predicted total time and conservative score: **~4801.7s**, with no unsupported tyre-life laps

## Deliverables
- Trained models, evaluation metrics, plots, and case study report
- Strategy simulator to compare candidate pit windows/compounds
