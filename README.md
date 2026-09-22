# F1 pit-strategy decision support

**Compare pit windows and tyre plans with lap-time predictions and uncertainty, not a single magic answer.**

This Streamlit project combines historical Formula 1 lap data, two scikit-learn models, and Monte Carlo simulation. It estimates how alternative strategies might play out under the model's assumptions. It is a research and exploration tool, not a live race-strategy system.

[Try the app](https://f1-predictor-temp.streamlit.app/) · [Run locally](#quick-start) · [Read the evaluation](#results)

| Data | Model | Decision view |
| :--- | :--- | :--- |
| FastF1 race laps and engineered stint, compound, and weather features. | Ridge baseline and HistGradientBoosting lap-time model. | Compare strategy-time distributions and inspect model diagnostics. |

---

## At a glance

![Belgian Grand Prix case-study strategy comparison](figures/case_study_max_round14.png)

The figure above is a generated case study, not a screenshot of the interactive app. The [live Streamlit app](https://f1-predictor-temp.streamlit.app/) contains the dashboard, simulator, and model-performance views.

---

## ❓ Problem Statement

A Formula 1 race strategy engineer must decide:

- **When** to pit (which lap)
- **Which tyre compound** to fit next (Soft / Medium / Hard)
- **How many stops** to make in total

Getting this wrong costs seconds — sometimes entire race positions. The challenge is that the right answer depends on dozens of factors at once: how quickly a particular tyre degrades over a stint, the ambient and track temperature on race day, how long a pit stop costs at a specific circuit, and how all of these factors interact across 40–70 laps.

**Data helps** because historical lap times encode tyre degradation curves, track-specific pace differentials between compounds, and pit-stop time costs. A model trained on this data can predict how fast each lap will be under any hypothetical strategy — far faster than a human can reason through every combination by hand.

---

## 💡 Why this project?

Modern Formula 1 strategy decisions depend on balancing tyre degradation, weather, pit-stop costs, and race conditions. Evaluating every possible combination manually is impractical. This project demonstrates how machine learning and simulation can support engineers by narrowing down promising strategies quickly while providing confidence estimates rather than single predictions.

---

## ✨ Features

| Category | Feature |
|---|---|
| 🎮 **Interactive Strategy Simulator** | Compare strategies, view confidence intervals, and explore optimal pit windows |
| 📊 **Race Dashboard** | View live telemetry, model metrics, and season-at-a-glance stats |
| 📈 **Feature Importance** | Interactive permutation importance graph to explain which features influence predictions most |
| ✏️ **Custom Strategy Input** | Test any user-defined pit plan (e.g. `SOFT:18,MEDIUM:22,HARD:20`) |
| 🎲 **Monte Carlo Simulation** | Up to 2,000 simulations per strategy with residual-based lap noise and pit loss sampling. [Read more](docs/MONTE_CARLO.md) |
| 🧠 **Machine Learning Prediction** | Dual-model training (Ridge + HistGradientBoosting) with rolling train/test splits. [Read more](docs/ML_PIPELINE.md) |
| 🔬 **Feature Engineering** | Race-normalised lap delta target, safety-car flags, pit-lap flags, and weather features |
| 📥 **CSV Export** | Download Monte Carlo simulation results as CSV |
| 🧭 **Explainability** | Model Performance tab now includes rolling MAE, compound/round diagnostics, and feature importance artifacts |

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Programming** | Python |
| **UI** | Streamlit, Custom CSS (glassmorphism), HTML5 Canvas Animations |
| **Machine Learning** | Scikit-learn (`Ridge`, `HistGradientBoostingRegressor`) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Altair, Matplotlib, Seaborn |
| **Utilities** | Joblib, FastF1, PyArrow |

---

## 📐 Design Decisions

| Decision | Reason |
|---|---|
| **HistGradientBoosting** | Better performance on nonlinear tabular data compared to standard regressors |
| **Ridge Regression** | Baseline for comparison |
| **Monte Carlo** | Model uncertainty instead of single predictions |
| **Streamlit** | Rapid interactive prototyping and deployment |
| **Parquet** | Faster loading and smaller storage than CSV |

---

## 🚀 Quick Start

For detailed project structure and advanced usage, see the [Project Structure Docs](docs/PROJECT_STRUCTURE.md).

```bash
# 1. Clone the repository
git clone https://github.com/Ninjax26/f1-pit-strategy-ml.git
cd f1-pit-strategy-ml

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app (pre-built data and models included)
streamlit run app.py
```

---

## 📈 Results

Models are trained on **Rounds 1–16** (Bahrain → Italy) and evaluated on **Rounds 17–24** (Azerbaijan → Abu Dhabi) — races the model has never seen during training.

### Overall Performance

| Model | MAE (s) | RMSE (s) |
|---|---|---|
| **HGB (HistGradientBoosting)** | **1.41** | **2.25** |
| Ridge Regression | 3.74 | 4.91 |

_MAE and RMSE measure error relative to the true absolute lap time._

### Latest Evaluation Artifacts

- `data/metrics/metrics.json` stores the current split-wide metrics
- `data/metrics/feature_importance_hgb.csv` stores permutation feature importance for HGB
- `data/metrics/rolling_metrics_hgb.json` and `data/metrics/rolling_metrics_ridge.json` store rolling validation results
- `data/metrics/predictions_hgb.parquet` and `data/metrics/predictions_ridge.parquet` store per-lap predictions and residuals used by Monte Carlo noise sampling

### Case Study — Max Verstappen, Round 14 (Belgian GP)

In the Belgian GP case study, the simulator recommended a one-stop Medium → Hard strategy with a pit window comparable to the actual race strategy, demonstrating that the approach can generate realistic strategy recommendations under historical race conditions.

### Current App Behavior

- Streamlit dashboard with separate `Dashboard`, `Strategy Simulator`, and `Model Performance` tabs
- Sidebar tooltips for each simulation control
- Auto-run toggle for faster iteration during demo sessions
- Monte Carlo outputs mean and P10/P50/P90 strategy time bands when multiple simulations are enabled

## 📊 Generated Evaluation Plots

The training and evaluation workflow produces static plots for model diagnostics and strategy analysis. These can be regenerated with:

```bash
python src/plots/make_plots.py --model hgb
```

![Predicted vs Actual](figures/pred_vs_actual_hgb.png)

![Residual Distribution](figures/residuals_hgb.png)

![MAE by Compound](figures/mae_by_compound_hgb.png)

![MAE by Round](figures/mae_by_round_hgb.png)

![MAE by Stint](figures/mae_by_stint_hgb.png)

## ⚠️ Known Limitations

- Wet-weather laps show much higher MAE than dry-weather laps. The model is much more reliable for dry-race strategy decisions than for wet-race calls without a dedicated wet-weather model.
- Round 21 (Las Vegas) shows an MAE spike compared with the rest of the season, which is worth investigating for unusual track conditions, safety-car effects, or data-quality issues.
- The simulator still relies on historical lap-time patterns and sampled residuals, so it does not explicitly model live traffic, safety-car timing, or overtakes.

---

## 📚 What I Learned

- Designing maintainable Streamlit applications
- Building reproducible ML pipelines
- Separating UI from business logic
- Feature engineering for time-series datasets
- Simulating uncertainty using Monte Carlo methods
- Organizing reusable Python modules

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.
