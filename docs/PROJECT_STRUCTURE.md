# Project Structure

```
f1/
│
├── app.py                    # Main Streamlit app — layout, simulation engine, UI logic
├── three_components.py       # Canvas-based animated components (hero, telemetry, loader, tire viz, gauge)
├── ui_helpers.py             # UI rendering functions (cards, tables, charts, insights)
├── ui_styles.py              # Full CSS design system (glassmorphism, sidebar, animations, compound colors)
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── data/
│   │   └── pull_2024_races.py        # FastF1 data fetcher — saves laps.parquet per race
│   ├── features/
│   │   └── build_features.py         # Cleaning, flagging, delta target, feature selection
│   ├── models/
│   │   ├── train_models.py           # Ridge + HGB training with sklearn Pipelines
│   │   └── evaluate.py               # Standard + rolling evaluation, group metrics, residual export
│   └── sim/
│       ├── compute_pit_loss.py       # Per-race pit-stop time cost estimation
│       └── strategy_sim.py           # CLI strategy simulator (Monte Carlo, strategy enumeration)
│
├── data/
│   ├── raw/
│   │   └── 2024/
│   │       └── round_XX_<name>/
│   │           └── laps.parquet      # Raw lap + weather data per race
│   ├── features/
│   │   └── features_2024.parquet     # Cleaned, engineered feature dataset
│   ├── models/
│   │   ├── hgb_model.joblib          # Trained HistGradientBoosting pipeline
│   │   └── ridge_model.joblib        # Trained Ridge Regression pipeline
│   └── metrics/
│       ├── metrics.json              # Overall MAE/RMSE for both models
│       ├── metrics_{hgb,ridge}.json  # Per-model metrics with train/test rounds
│       ├── predictions_{model}.parquet  # Test-set predictions with residuals
│       ├── mae_by_compound_{model}.csv
│       ├── mae_by_round_{model}.csv
│       ├── mae_by_stint_{model}.csv
│       ├── rolling_metrics_{model}.json
│       └── pit_loss_2024.csv         # Pit-stop time cost distribution per race
│
├── figures/                  # Pre-generated evaluation plots (PNG)
├── notebooks/
│   └── 03_evaluation.ipynb   # Evaluation notebook
├── reports/
│   ├── summary.md
│   ├── summary_resume.md
│   └── case_study_max_round14.md  # Verstappen Belgian GP case study
└── cache/                    # FastF1 local cache directory
```
