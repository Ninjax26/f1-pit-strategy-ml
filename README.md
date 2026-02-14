# 🏎️ F1 Pit Strategy ML + Simulation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

Lap-time prediction + **Monte Carlo pit strategy simulation** for F1 2024.

## ✨ Features

- **ML lap-time models**: Ridge + HistGradientBoosting (HGB), trained on `LapTimeDelta`
- **Strategy simulator**: 1-stop & 2-stop strategies with compound rules
- **Interactive web app**: Round/driver selection, run sims, see best strategy & charts
- **Pit-loss & residuals**: Race-specific pit loss and model residual noise

## 🚀 Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then click **▶ Run Simulation** to see top strategies and the best option.

## 📁 Pipeline (run once for data & models)

1. `python src/data/pull_2024_races.py --cache-dir cache`
2. `python src/features/build_features.py`
3. `python src/models/train_models.py --train-rounds 1-16 --test-rounds 17-24`
4. `python src/sim/compute_pit_loss.py`
5. `python src/models/evaluate.py --rolling`
6. `streamlit run app.py`

## 📊 Results (2024 test rounds 17–24)

| Model | MAE | RMSE |
|-------|-----|------|
| HGB   | **1.49s** | 2.30s |
| Ridge | 3.74s | 4.91s |

## 📂 Structure

- `app.py` — Streamlit app (F1-themed UI, model metrics, best strategy callout)
- `data/` — raw, features, models, metrics
- `src/` — data, features, models, sim, plots, reports

## 🌐 Deploy (Streamlit Cloud)

1. Push repo to GitHub  
2. [share.streamlit.io](https://share.streamlit.io) → connect repo → main file: `app.py`  
3. Deploy (app uses `data/` in the repo)
