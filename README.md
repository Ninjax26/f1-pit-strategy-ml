# 🏎️ F1 Pit Strategy ML + Simulation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

Lap-time prediction + **Monte Carlo pit strategy simulation** for F1 2024, wrapped in a sleek, F1-themed interactive dashboard powered by **Three.js** animations.

---

## ✨ Features

### 🧠 Machine Learning
- **Dual ML models** — Ridge Regression & HistGradientBoosting (HGB) trained on `LapTimeDelta`
- **20+ engineered features** — tire degradation curves, weather interpolation, safety car flags, race-normalised lap deltas
- **Time-based cross-validation** — rolling train/test splits that mimic real deployment (always tested on unseen future races)

### 🎲 Strategy Simulator
- **Monte Carlo engine** — up to 2 000 simulations per strategy with residual-based noise
- **1-stop & 2-stop strategies** with compound rules, customisable stint lengths, and optional wet compounds
- **Race-specific pit loss** distributions (median, mean, std, P10/P90)
- **Custom strategy input** — test any stint plan (e.g. `SOFT:18,MEDIUM:22,HARD:20`)

### 🎨 Interactive UI
- **Three.js particle hero** — animated speed-trail background with F1 branding
- **Live telemetry dashboard** — animated gauges and scrolling race data strip
- **Simulation loader** — racing-particle spinner while Monte Carlo runs
- **Tire compound visualiser** — animated sidebar tire icon per compound
- **Glassmorphism cards, Orbitron typography, micro-animations** throughout the app
- **Three-tab layout** — Dashboard · Strategy Simulator · Model Performance

---

## 📊 Results (2024 test rounds 17–24)

| Model | MAE | RMSE |
|-------|-----|------|
| HGB   | **1.49 s** | 2.30 s |
| Ridge | 3.74 s | 4.91 s |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run app.py
```

Select a **season → round → driver**, configure strategy parameters in the sidebar, and click **▶ Run Simulation** to see top strategies and the best option.

---

## 📁 Data & Model Pipeline (run once)

```bash
python src/data/pull_2024_races.py --cache-dir cache
python src/features/build_features.py
python src/models/train_models.py --train-rounds 1-16 --test-rounds 17-24
python src/sim/compute_pit_loss.py
python src/models/evaluate.py --rolling
streamlit run app.py
```

---

## 📂 Project Structure

```
f1/
├── app.py                  # Main Streamlit app — tabs, sidebar, simulation engine
├── three_components.py     # Three.js-powered components (hero, telemetry, loader, tire viz, gauge)
├── ui_helpers.py           # UI rendering functions (hero, cards, tables, insights, charts)
├── ui_styles.py            # Full CSS design system (glassmorphism, sidebar, animations)
├── requirements.txt        # Python dependencies
├── data/
│   ├── features/           # Engineered feature parquets per season
│   ├── models/             # Trained model .joblib files (HGB, Ridge)
│   └── metrics/            # Model metrics, pit-loss CSVs, residual parquets
├── figures/                # Pre-generated evaluation plots (residuals, rolling MAE, etc.)
├── src/
│   ├── data/               # Data pulling & cleaning scripts
│   ├── features/           # Feature engineering pipeline
│   ├── models/             # Training & evaluation scripts
│   └── sim/                # Pit-loss computation
├── notebooks/              # Exploratory analysis
└── reports/                # Generated analysis reports
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn (Ridge, HistGradientBoosting) |
| Data | FastF1, pandas, NumPy, PyArrow |
| Visualisation | Streamlit, Altair, Matplotlib, Seaborn |
| UI Animations | Three.js (via Streamlit HTML components) |
| Styling | Custom CSS (Orbitron + Inter fonts, glassmorphism) |

---

## 🌐 Deploy (Streamlit Cloud)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect repo → main file: `app.py`
3. Deploy (app uses `data/` committed in the repo)

---

## 📜 License

See [LICENSE](LICENSE) for details.
