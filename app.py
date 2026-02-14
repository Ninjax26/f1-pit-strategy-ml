"""
F1 Pit Strategy Simulator — Lap-time ML + Monte Carlo strategy simulation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

try:
    import altair as alt
except Exception:
    alt = None


DATA_DIR = Path("data")
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
METRICS_DIR = DATA_DIR / "metrics"

# ----- Custom theme: F1-inspired dark + red -----
st.set_page_config(
    page_title="F1 Strategy Simulator",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    .main { background: linear-gradient(180deg, #0e0e0e 0%, #1a1a1a 50%, #0e0e0e 100%); }
    .stApp { background: #0e0e0e; }
    h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; color: #e10600 !important; }
    .stMetric { background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #e10600; }
    .stMetric label { color: #b0b0b0 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #fff !important; font-family: 'JetBrains Mono', monospace !important; }
    div[data-testid="stVerticalBlock"] > div { border-radius: 8px; }
    .best-strategy { background: linear-gradient(135deg, #1a2e1a 0%, #0d1a0d 100%); padding: 1rem 1.5rem; border-radius: 12px; border: 1px solid #2e7d32; margin: 1rem 0; }
    .hero { text-align: center; padding: 1.5rem 0; border-bottom: 2px solid #e10600; margin-bottom: 1.5rem; }
    .hero h1 { font-size: 2.2rem !important; margin-bottom: 0.25rem !important; }
    .hero p { color: #888; font-size: 1rem; }
    .stButton > button { background: linear-gradient(90deg, #e10600 0%, #b80500 100%) !important; color: white !important; font-weight: 600 !important; border: none !important; padding: 0.6rem 1.5rem !important; border-radius: 8px !important; }
    .stButton > button:hover { background: linear-gradient(90deg, #ff1a0d 0%, #e10600 100%) !important; box-shadow: 0 0 20px rgba(225,6,0,0.4); }
    .model-badge { display: inline-block; background: #e10600; color: white; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.85rem; margin-right: 0.5rem; }
    .sidebar .sidebar-content { background: #141414; }
</style>
""", unsafe_allow_html=True)


def parse_strategy(spec: str) -> list[tuple[str, int]]:
    stints = []
    for part in spec.split(","):
        compound, length = part.split(":")
        stints.append((compound.strip().upper(), int(length)))
    return stints


def available_compounds(race_df: pd.DataFrame, include_wet: bool) -> list[str]:
    compounds = sorted(set(race_df.get("Compound", pd.Series([])).astype(str).str.upper().dropna()))
    dry = [c for c in compounds if c in {"SOFT", "MEDIUM", "HARD"}]
    wet = [c for c in compounds if c not in {"SOFT", "MEDIUM", "HARD"}]
    if include_wet:
        return dry + wet if dry else wet
    return dry if dry else compounds


def generate_strategies(
    total_laps: int,
    compounds: list[str],
    max_stops: int,
    min_stint: int,
    max_stint: int,
    step: int,
    require_two_compounds: bool,
) -> dict[str, list[tuple[str, int]]]:
    strategies: dict[str, list[tuple[str, int]]] = {}
    if max_stint <= 0:
        max_stint = total_laps
    if max_stops <= 0:
        if not require_two_compounds:
            for c1 in compounds:
                strategies[f"0stop_{c1[0]}_{total_laps}"] = [(c1, total_laps)]
        return strategies
    if max_stops >= 1:
        for c1 in compounds:
            for c2 in compounds:
                if require_two_compounds and c1 == c2:
                    continue
                for s1 in range(min_stint, total_laps - min_stint + 1, step):
                    s2 = total_laps - s1
                    if s2 < min_stint or s2 > max_stint:
                        continue
                    strategies[f"1stop_{c1[0]}-{c2[0]}_{s1}-{s2}"] = [(c1, s1), (c2, s2)]
    if max_stops >= 2:
        for c1 in compounds:
            for c2 in compounds:
                for c3 in compounds:
                    if require_two_compounds and len({c1, c2, c3}) < 2:
                        continue
                    for s1 in range(min_stint, total_laps - 2 * min_stint + 1, step):
                        for s2 in range(min_stint, total_laps - s1 - min_stint + 1, step):
                            s3 = total_laps - s1 - s2
                            if s3 < min_stint or s3 > max_stint:
                                continue
                            strategies[f"2stop_{c1[0]}-{c2[0]}-{c3[0]}_{s1}-{s2}-{s3}"] = [(c1, s1), (c2, s2), (c3, s3)]
    return strategies


def build_laps(base_laps: pd.DataFrame, strategy: list[tuple[str, int]]) -> pd.DataFrame:
    laps = base_laps.copy().reset_index(drop=True)
    lap_idx = 0
    stint_idx = 1
    for compound, length in strategy:
        for i in range(length):
            if lap_idx >= len(laps):
                break
            laps.loc[lap_idx, "Compound"] = compound
            if "TyreLife" in laps.columns:
                laps.loc[lap_idx, "TyreLife"] = i + 1
            if "Stint" in laps.columns:
                laps.loc[lap_idx, "Stint"] = stint_idx
            lap_idx += 1
        stint_idx += 1
    return laps


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(val) else val


def _fixed_pit_loss_stats(value: float) -> dict:
    return {"median": value, "mean": value, "std": 0.0, "p10": value, "p90": value}


def load_pit_loss(metrics_path: Path, round_number: int) -> dict | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    row = df[df["round"] == round_number]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "median": _safe_float(row.get("pit_loss_median")),
        "mean": _safe_float(row.get("pit_loss_mean")),
        "std": _safe_float(row.get("pit_loss_std")),
        "p10": _safe_float(row.get("pit_loss_p10")),
        "p90": _safe_float(row.get("pit_loss_p90")),
    }


def sample_pit_loss(stats: dict | None, rng: np.random.Generator, mode: str) -> float:
    if stats is None:
        return 20.0
    median = stats.get("median")
    mean = stats.get("mean")
    std = stats.get("std")
    p10, p90 = stats.get("p10"), stats.get("p90")
    center = mean if mean is not None else (median if median is not None else 20.0)
    if mode == "fixed":
        return center
    if std is None or std <= 0:
        std = (p90 - p10) / 2.563 if (p10 is not None and p90 is not None and p90 > p10) else 2.0
    return float(np.clip(float(rng.normal(center, std)), 5.0, 60.0))


def load_residuals(residuals_path: Path) -> dict | None:
    if not residuals_path.exists():
        return None
    df = pd.read_parquet(residuals_path)
    if "residual" not in df.columns or df.empty:
        return None
    df = df[pd.notna(df["residual"])].copy()
    if df.empty:
        return None
    q01, q99 = np.quantile(df["residual"].to_numpy(), [0.01, 0.99])
    df = df[(df["residual"] >= q01) & (df["residual"] <= q99)]
    if df.empty:
        return None
    global_residuals = df["residual"].to_numpy()
    by_compound = {}
    if "Compound" in df.columns:
        for comp, grp in df.groupby("Compound"):
            arr = grp["residual"].dropna().to_numpy()
            if len(arr) >= 50:
                by_compound[str(comp).upper()] = arr
    return {"global": global_residuals, "by_compound": by_compound}


def sample_residual(compound: str, residuals: dict | None, rng: np.random.Generator) -> float:
    if residuals is None:
        return 0.0
    comp = str(compound).upper()
    arr = residuals.get("by_compound", {}).get(comp) or residuals.get("global")
    if arr is None or len(arr) == 0:
        return 0.0
    return float(rng.choice(arr))


@st.cache_data(show_spinner=False)
def load_features(season: int) -> pd.DataFrame | None:
    path = FEATURES_DIR / f"features_{season}.parquet"
    return pd.read_parquet(path) if path.exists() else None


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    path = MODELS_DIR / f"{model_name}_model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_residuals_cached(model_name: str) -> dict | None:
    return load_residuals(METRICS_DIR / f"predictions_{model_name}.parquet")


@st.cache_data(show_spinner=False)
def load_model_metrics(season: int) -> dict | None:
    path = METRICS_DIR / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def simulate_strategies(
    race_df: pd.DataFrame,
    model,
    strategies: dict[str, list[tuple[str, int]]],
    pit_loss_stats: dict,
    n_sims: int,
    pit_loss_mode: str,
    residuals: dict | None,
    noise_sigma: float | None,
    seed: int | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    results = []
    target_is_delta = "LapTimeDelta" in race_df.columns and "RaceMedianLap" in race_df.columns
    race_median = race_df["RaceMedianLap"].iloc[0] if "RaceMedianLap" in race_df.columns else race_df["LapTimeSeconds"].median()
    if n_sims <= 1:
        pit_loss_mode = "fixed"

    for name, strategy in strategies.items():
        laps = build_laps(race_df, strategy)
        X = laps.drop(columns=["LapTimeSeconds", "LapTimeDelta"], errors="ignore")
        base_preds = model.predict(X)
        if target_is_delta:
            base_preds = base_preds + race_median
        n_stops = len(strategy) - 1

        if n_sims <= 1:
            pit_loss_value = sample_pit_loss(pit_loss_stats, rng, pit_loss_mode)
            total_time = float(np.sum(base_preds)) + pit_loss_value * n_stops
            results.append({"strategy": name, "total_time_s": total_time, "pit_loss_s": pit_loss_value, "stops": n_stops, "stints": json.dumps(strategy)})
        else:
            totals, pit_losses = [], []
            compounds = laps.get("Compound", pd.Series([None] * len(laps)))
            for _ in range(n_sims):
                lap_preds = np.array(base_preds, dtype=float)
                if residuals:
                    lap_preds += np.array([sample_residual(c, residuals, rng) for c in compounds])
                elif noise_sigma and noise_sigma > 0:
                    lap_preds += rng.normal(0.0, noise_sigma, size=lap_preds.shape[0])
                pit_total = sum(sample_pit_loss(pit_loss_stats, rng, pit_loss_mode) for _ in range(n_stops))
                pit_losses.append(pit_total)
                totals.append(float(np.sum(lap_preds)) + pit_total)
            totals_arr = np.array(totals, dtype=float)
            results.append({
                "strategy": name,
                "total_time_mean_s": float(np.mean(totals_arr)),
                "total_time_p10_s": float(np.quantile(totals_arr, 0.10)),
                "total_time_p50_s": float(np.quantile(totals_arr, 0.50)),
                "total_time_p90_s": float(np.quantile(totals_arr, 0.90)),
                "pit_loss_mean_s": float(np.mean(pit_losses)) if pit_losses else 0.0,
                "stops": n_stops,
                "stints": json.dumps(strategy),
            })

    key = "total_time_mean_s" if n_sims > 1 else "total_time_s"
    return pd.DataFrame(results).sort_values(key)


# ----- UI -----
st.markdown('<div class="hero"><h1>🏎️ F1 Pit Strategy Simulator</h1><p>Lap-time ML + Monte Carlo strategy simulation</p></div>', unsafe_allow_html=True)

st.info("💡 **Setup:** Run the pipeline first (pull data → build features → train models → compute pit-loss → evaluate). See README for the workflow.")


season = st.sidebar.number_input("Season", value=2024, step=1, min_value=2018, max_value=2026)
features_df = load_features(int(season))
if features_df is None:
    st.error(f"Missing features: {FEATURES_DIR / f'features_{season}.parquet'}")
    st.stop()

rounds = sorted(features_df["RoundNumber"].dropna().unique().astype(int))
if not rounds:
    st.error("No rounds in features.")
    st.stop()

selected_round = st.sidebar.selectbox("Round", rounds, index=len(rounds) - 1)
round_df = features_df[features_df["RoundNumber"] == selected_round].copy()
if round_df.empty:
    st.error("No data for selected round.")
    st.stop()

available_drivers = sorted(round_df["Driver"].dropna().unique().astype(str))
default_driver_idx = available_drivers.index("VER") if "VER" in available_drivers else 0
selected_driver = st.sidebar.selectbox("Driver", available_drivers, index=default_driver_idx)

model_name = st.sidebar.selectbox("Model", ["hgb", "ridge"], index=0)
model = load_model(model_name)
if model is None:
    st.error(f"Missing model: {MODELS_DIR / f'{model_name}_model.joblib'}")
    st.stop()

# Model metrics
model_metrics = load_model_metrics(int(season))
if model_metrics and model_name in model_metrics and isinstance(model_metrics[model_name], dict):
    m = model_metrics[model_name]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Model metrics (test)**")
    st.sidebar.markdown(f"MAE: **{m.get('mae', 0):.2f}s** · RMSE: **{m.get('rmse', 0):.2f}s**")

max_stops = st.sidebar.selectbox("Max Stops", [1, 2], index=1)
metrics_path = METRICS_DIR / f"pit_loss_{season}.csv"
pit_loss_stats = load_pit_loss(metrics_path, int(selected_round)) or _fixed_pit_loss_stats(20.0)

race_df = round_df[round_df["Driver"] == selected_driver].copy()
if race_df.empty:
    st.error("No laps for that driver.")
    st.stop()
race_df = race_df.sort_values("LapNumber").reset_index(drop=True)
total_laps = int(race_df["LapNumber"].max())

min_stint_max = max(1, min(20, total_laps))
min_stint_default = min(8, max(1, total_laps // (max_stops + 1)), min_stint_max)
min_stint = st.sidebar.slider("Min Stint", 1, min_stint_max, min_stint_default)
max_stint_min = max(1, min_stint)
max_stint_max = max(max_stint_min, min(50, total_laps))
max_stint = st.sidebar.slider("Max Stint", max_stint_min, max_stint_max, min(35, max_stint_max))
stint_step = st.sidebar.slider("Stint Step", 1, 5, 2)
include_wet = st.sidebar.checkbox("Include Wet Compounds", value=False)
allow_single = st.sidebar.checkbox("Allow Single Compound", value=False)

max_feasible_stops = max(0, total_laps // max(min_stint, 1) - 1)
effective_max_stops = min(max_stops, max_feasible_stops)
if effective_max_stops < max_stops:
    st.sidebar.warning(f"Max Stops reduced to {effective_max_stops}.")

n_sims = st.sidebar.slider("Simulations", 1, 2000, 200, step=50)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
pit_loss_mode = st.sidebar.selectbox("Pit Loss Mode", ["sample", "fixed"], index=0)
noise_sigma = st.sidebar.slider("Gaussian Noise Sigma", 0.0, 5.0, 0.0, step=0.1)
use_residuals = st.sidebar.checkbox("Use Residual Noise", value=True)
residuals = load_residuals_cached(model_name) if use_residuals else None
custom_strategy = st.sidebar.text_input("Custom Strategy (optional)", value="", placeholder="SOFT:18,MEDIUM:22,HARD:20")

st.subheader("📋 Inputs")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Laps", int(total_laps))
with col2:
    st.metric("Pit Loss Median (s)", f"{pit_loss_stats.get('median', 20.0):.2f}")
with col3:
    st.metric("Pit Loss Std (s)", f"{pit_loss_stats.get('std', 0.0):.2f}")

run = st.button("▶ Run Simulation", type="primary")

if run:
    with st.spinner("Running Monte Carlo simulation…"):
        try:
            if custom_strategy.strip():
                strategies = {"custom": parse_strategy(custom_strategy.strip())}
            else:
                compounds = available_compounds(race_df, include_wet)
                strategies = generate_strategies(
                    total_laps=int(race_df["LapNumber"].max()),
                    compounds=compounds,
                    max_stops=effective_max_stops,
                    min_stint=min_stint,
                    max_stint=max_stint,
                    step=stint_step,
                    require_two_compounds=not allow_single,
                )
                if not strategies:
                    st.error("No valid strategies. Try lowering Min Stint or enabling Allow Single Compound.")
                    st.stop()

            results = simulate_strategies(
                race_df=race_df,
                model=model,
                strategies=strategies,
                pit_loss_stats=pit_loss_stats,
                n_sims=n_sims,
                pit_loss_mode=pit_loss_mode,
                residuals=residuals,
                noise_sigma=noise_sigma if not use_residuals else None,
                seed=int(seed) if seed is not None else None,
            )

            st.success(f"✅ Simulated **{len(results)}** strategies with **{n_sims}** run(s) each.")

            # Best strategy callout
            if n_sims > 1 and "total_time_mean_s" in results.columns:
                best = results.iloc[0]
                st.markdown(f'<div class="best-strategy"><strong>🏆 Best strategy</strong>: <code>{best["strategy"]}</code> — Mean total time: <strong>{best["total_time_mean_s"]:.2f}s</strong> (P10–P90: {best["total_time_p10_s"]:.2f}s – {best["total_time_p90_s"]:.2f}s)</div>', unsafe_allow_html=True)
            elif "total_time_s" in results.columns:
                best = results.iloc[0]
                st.markdown(f'<div class="best-strategy"><strong>🏆 Best strategy</strong>: <code>{best["strategy"]}</code> — Total time: <strong>{best["total_time_s"]:.2f}s</strong></div>', unsafe_allow_html=True)

            st.subheader("📊 Top strategies")
            display_df = results.head(15)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            if alt and n_sims > 1 and "total_time_mean_s" in display_df.columns:
                chart = (
                    alt.Chart(display_df.head(10))
                    .mark_bar()
                    .encode(
                        x=alt.X("total_time_mean_s:Q", title="Mean total time (s)"),
                        y=alt.Y("strategy:N", sort="-x"),
                        tooltip=["strategy", "total_time_mean_s", "total_time_p10_s", "total_time_p90_s", "pit_loss_mean_s"],
                    )
                )
                st.altair_chart(chart, use_container_width=True)

            st.download_button(
                "Download results CSV",
                results.to_csv(index=False).encode("utf-8"),
                file_name="strategy_results.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.exception(e)
else:
    st.caption("👆 Click **Run Simulation** to see strategy rankings and best strategy.")

st.caption("Tip: Use residual noise with ≥200 sims for realistic uncertainty bands.")
