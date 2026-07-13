# Machine Learning Pipeline

```
FastF1 API
    │
    ▼
src/data/pull_2024_races.py
    │  Fetches lap + weather data for each 2024 race session
    │  Saves: data/raw/2024/round_XX_<name>/laps.parquet
    ▼
src/features/build_features.py
    │  Drops deleted laps and rows with no LapTime
    │  Flags pit laps (PitInTime/PitOutTime) and safety-car laps (TrackStatus 4/5)
    │  Uses the latest prior-season circuit median → PreRaceBaseline
    │  Computes LapTimeDelta without using the completed target race
    │  Saves: data/features/features_2024.parquet
    ▼
src/sim/compute_pit_loss.py
    │  Estimates pit-stop time cost per race from raw lap data
    │  Uses robust median filtering (5th–95th percentile, 5–60s bounds)
    │  Saves: data/metrics/pit_loss_2024.csv
    ▼
src/sim/support.py
    │  Builds per-circuit and global tyre-life support limits from prior seasons
    │  Saves: data/metrics/strategy_support_2024.json
    ▼
src/models/train_models.py
    │  Loads feature parquet, applies train/test round split (Rounds 1–16 / 17–24)
    │  Builds sklearn Pipeline: SimpleImputer → OneHotEncoder (cat) + median imputer (num) → model
    │  Trains Ridge (alpha=1.0) and HistGradientBoosting (max_depth=8, lr=0.1)
    │  Saves: data/models/{ridge,hgb}_model.joblib
    │  Saves: data/metrics/metrics.json
    ▼
src/models/evaluate.py
    │  Loads saved models, runs standard + rolling train/test evaluation
    │  Computes MAE/RMSE overall and grouped by: Compound, Stint, RoundNumber
    │  Saves: data/metrics/predictions_{model}.parquet (includes residuals)
    │         data/metrics/mae_by_{compound,stint,round}_{model}.csv
    │         data/metrics/rolling_metrics_{model}.json
    ▼
app.py  (Streamlit)
    │  Loads features, models, metrics, and pit-loss CSV from data/
    │  Renders Dashboard / Strategy Simulator / Model Performance tabs
    ▼
Monte Carlo Simulation (in app.py)
    │  For each strategy candidate, builds a per-lap feature DataFrame
    │  Runs the trained model to get base lap-time predictions
    │  For n_sims > 1: samples residual noise per lap (from predictions_{model}.parquet)
    │                   samples pit loss from per-race Normal distribution
    │  Aggregates total race time → mean, P10, P50, P90 across simulations
    │  Adds upper-tail uncertainty and historical-support penalties
    ▼
Strategy Rankings  →  displayed in UI / downloadable as CSV
```
