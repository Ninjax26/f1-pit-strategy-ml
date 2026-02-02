# F1 2024 Pit Strategy ML + Simulation (Resume Summary)

## Problem
Build an ML model to predict lap time and use it to simulate pit strategies for F1 2024 main races.

## Data
- Source: FastF1 race sessions (2024 main races)
- Features: tyre compound, tyre life, stint, lap number, weather, track status, driver/team

## Approach
- Target: `LapTimeDelta` = lap time minus race median (normalizes track-to-track pace)
- Models: Ridge regression baseline + HistGradientBoostingRegressor (tree)
- Evaluation: time-based split (Rounds 1–16 train, 17–24 test) + rolling splits

## Results (Rounds 17–24)
- HGB MAE: **1.49s**, RMSE: **2.30s**
- Ridge MAE: **3.74s**, RMSE: **4.91s**
- Pit-loss estimated per race using robust median filtering

## Case Study (Max Verstappen, Round 14)
- Best simulated strategy: **1‑stop M→H**, pit around lap 10
- Predicted total time: **~4796.7s** with pit‑loss **~19.6s**

## Deliverables
- Trained models, evaluation metrics, plots, and case study report
- Strategy simulator to compare candidate pit windows/compounds

