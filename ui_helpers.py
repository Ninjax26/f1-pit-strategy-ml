"""UI helper functions for the F1 Pit Strategy Simulator."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from ui_styles import COMPOUND_CSS, COMPOUND_COLORS
from three_components import render_particle_hero, render_live_telemetry, render_simulation_loader

FIGURES_DIR = Path("figures")


def format_race_time(seconds: float) -> str:
    """Convert seconds to human-readable race time like 1h 19m 56.7s."""
    if seconds <= 0 or np.isnan(seconds):
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:04.1f}s"
    if minutes > 0:
        return f"{minutes}m {secs:04.1f}s"
    return f"{secs:.2f}s"


def render_hero():
    render_particle_hero(height=420)


def render_how_it_works():
    st.markdown("### ⚙️ How It Works")
    cols = st.columns([2, 1, 2, 1, 2])
    steps = [
        ("📊", "DATA COLLECTION", "Race telemetry from all 2024 F1 races — lap times, tire compounds, weather, track conditions"),
        ("→", None, None),
        ("🤖", "ML PREDICTION", "Gradient Boosting model predicts lap times based on tire age, compound, weather & track factors"),
        ("→", None, None),
        ("🎯", "STRATEGY SIM", "Monte Carlo simulation tests thousands of pit strategies to find the optimal tire & pit window"),
    ]
    for i, (icon, title, desc) in enumerate(steps):
        with cols[i]:
            if title is None:
                st.markdown(f'<div class="flow-arrow">{icon}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="flow-step">
                    <div class="flow-icon">{icon}</div>
                    <div class="flow-title">{title}</div>
                    <div class="flow-desc">{desc}</div>
                </div>""", unsafe_allow_html=True)


def render_model_comparison(metrics: dict):
    st.markdown("### 🏆 Model Performance Comparison")
    col1, col2 = st.columns(2)
    for col, (name, label) in zip([col1, col2], [("hgb", "HGB (Gradient Boosting)"), ("ridge", "Ridge Regression")]):
        m = metrics.get(name, {})
        mae = m.get("mae", 0)
        rmse = m.get("rmse", 0)
        is_winner = name == "hgb"
        badge = '<span class="winner-badge">⭐ BEST MODEL</span>' if is_winner else '<span style="color:#666;font-size:0.8rem;">BASELINE</span>'
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                    <h3 style="margin:0!important;font-size:1.1rem!important;">{label}</h3>
                    {badge}
                </div>
                <div style="display:flex;gap:1.5rem;">
                    <div class="metric-card" style="flex:1;">
                        <div class="metric-label">MAE</div>
                        <div class="metric-value">{mae:.2f}<span class="metric-unit">sec</span></div>
                    </div>
                    <div class="metric-card" style="flex:1;">
                        <div class="metric-label">RMSE</div>
                        <div class="metric-value">{rmse:.2f}<span class="metric-unit">sec</span></div>
                    </div>
                </div>
                <div style="margin-top:1rem;color:#888;font-size:0.8rem;">
                    {"Avg prediction error of just ~1.5 seconds per lap" if is_winner else "Avg prediction error of ~3.7 seconds per lap"}
                </div>
            </div>""", unsafe_allow_html=True)


def render_season_stats(features_df: pd.DataFrame):
    st.markdown("### 📈 Season at a Glance")
    n_races = features_df["RoundNumber"].nunique()
    n_drivers = features_df["Driver"].nunique()
    n_teams = features_df["Team"].nunique()
    n_laps = len(features_df)
    cols = st.columns(4)
    labels = ["RACES ANALYZED", "DRIVERS", "TEAMS", "TOTAL LAPS"]
    values = [n_races, n_drivers, n_teams, f"{n_laps:,}"]
    icons = ["🏁", "👨‍✈️", "🏢", "🔄"]
    for col, icon, label, value in zip(cols, icons, labels, values):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{icon} {label}</div>
                <div class="metric-value">{value}</div>
            </div>""", unsafe_allow_html=True)


def render_stint_bar(strategy_stints: list, total_laps: int):
    """Render a colored bar showing tire stints."""
    segments = ""
    lap_counter = 0
    for compound, length in strategy_stints:
        compound_upper = compound.upper() if isinstance(compound, str) else compound
        css_class = COMPOUND_CSS.get(compound_upper, "stint-hard")
        pct = (length / total_laps) * 100
        segments += f'<div class="stint-segment {css_class}" style="width:{pct}%;" title="{compound_upper}: {length} laps">{compound_upper[0]} {length}L</div>'
        lap_counter += length
    return f'<div class="stint-bar">{segments}</div>'


def render_best_strategy(best_row, n_sims: int, total_laps: int):
    is_mc = n_sims > 1
    time_key = "total_time_mean_s" if is_mc else "total_time_s"
    time_val = best_row[time_key]
    time_str = format_race_time(time_val)

    stints = json.loads(best_row["stints"]) if isinstance(best_row["stints"], str) else best_row["stints"]
    stint_bar = render_stint_bar(stints, total_laps)

    detail = ""
    if is_mc:
        p10 = format_race_time(best_row.get("total_time_p10_s", 0))
        p90 = format_race_time(best_row.get("total_time_p90_s", 0))
        detail = f'<div class="strat-detail">Confidence range: {p10} — {p90} (P10–P90)</div>'

    st.markdown(f"""
    <div class="best-strategy-card">
        <div class="trophy">🏆</div>
        <div style="color:#81c784;font-size:0.85rem;text-transform:uppercase;letter-spacing:2px;">Optimal Strategy</div>
        <div class="strat-name">{best_row["strategy"]}</div>
        <div class="strat-time">{time_str}</div>
        {detail}
        <div style="max-width:500px;margin:1rem auto 0;">{stint_bar}</div>
        <div style="margin-top:0.5rem;color:#666;font-size:0.75rem;">
            {best_row["stops"]} stop{"s" if best_row["stops"] != 1 else ""} · {n_sims} simulation{"s" if n_sims > 1 else ""}
        </div>
    </div>""", unsafe_allow_html=True)


def generate_insights(results: pd.DataFrame, n_sims: int) -> list[str]:
    """Generate human-readable insights from simulation results."""
    insights = []
    is_mc = n_sims > 1
    time_key = "total_time_mean_s" if is_mc else "total_time_s"

    if len(results) < 2:
        return insights

    best = results.iloc[0]
    second = results.iloc[1]
    gap = second[time_key] - best[time_key]
    insights.append(f"The optimal strategy saves **{gap:.1f}s** ({format_race_time(gap)}) over the next best option.")

    stop_counts = results.head(5)["stops"].value_counts()
    dominant = stop_counts.idxmax()
    insights.append(f"**{dominant}-stop** strategies dominate the top 5 rankings for this race.")

    if len(results) >= 5:
        spread = results.iloc[4][time_key] - best[time_key]
        insights.append(f"The gap between rank 1 and rank 5 is **{spread:.1f}s** — {'a tight field' if spread < 5 else 'a significant difference'}.")

    return insights


def render_insights(insights: list[str]):
    if not insights:
        return
    text = "<br>".join(f"• {i}" for i in insights)
    st.markdown(f"""
    <div class="insight-box">
        <span class="insight-icon">💡</span> <strong style="color:#7eb8da;">Strategy Insights</strong>
        <div class="insight-text" style="margin-top:0.5rem;">{text}</div>
    </div>""", unsafe_allow_html=True)


def render_strategy_table(results: pd.DataFrame, n_sims: int, total_laps: int):
    """Render a formatted strategy results table."""
    is_mc = n_sims > 1
    display = results.head(15).copy()

    if is_mc:
        display["Rank"] = range(1, len(display) + 1)
        best_time = display["total_time_mean_s"].iloc[0]
        display["Δ to Best"] = display["total_time_mean_s"] - best_time
        display["Mean Time"] = display["total_time_mean_s"].apply(format_race_time)
        display["P10"] = display["total_time_p10_s"].apply(format_race_time)
        display["P90"] = display["total_time_p90_s"].apply(format_race_time)
        display["Δ (s)"] = display["Δ to Best"].apply(lambda x: f"+{x:.1f}s" if x > 0 else "—")
        show_cols = ["Rank", "strategy", "stops", "Mean Time", "P10", "P90", "Δ (s)"]
    else:
        display["Rank"] = range(1, len(display) + 1)
        best_time = display["total_time_s"].iloc[0]
        display["Δ to Best"] = display["total_time_s"] - best_time
        display["Total Time"] = display["total_time_s"].apply(format_race_time)
        display["Δ (s)"] = display["Δ to Best"].apply(lambda x: f"+{x:.1f}s" if x > 0 else "—")
        show_cols = ["Rank", "strategy", "stops", "Total Time", "Δ (s)"]

    st.dataframe(display[show_cols], use_container_width=True, hide_index=True)


def render_stint_gallery(results: pd.DataFrame, total_laps: int, top_n: int = 8):
    """Show visual tire stint bars for top strategies."""
    st.markdown("#### 🏁 Strategy Visual Breakdown")
    for i, row in results.head(top_n).iterrows():
        stints = json.loads(row["stints"]) if isinstance(row["stints"], str) else row["stints"]
        bar = render_stint_bar(stints, total_laps)
        rank = results.index.get_loc(i) + 1
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin:0.4rem 0;">
            <div style="min-width:160px;color:#888;font-size:0.8rem;font-family:'Orbitron',sans-serif;">
                #{rank} {row["strategy"][:20]}
            </div>
            <div style="flex:1;">{bar}</div>
        </div>""", unsafe_allow_html=True)


def render_model_performance_tab(metrics_dir: Path, figures_dir: Path):
    """Render Model Performance tab content."""
    import altair as alt

    st.markdown("### 📊 Model Accuracy Deep Dive")
    model_toggle = st.radio("Select Model", ["hgb", "ridge"], horizontal=True, index=0)
    label_map = {"hgb": "HGB (Gradient Boosting)", "ridge": "Ridge Regression"}
    st.markdown(f"**Showing results for: {label_map[model_toggle]}**")

    # Rolling metrics
    rolling_path = metrics_dir / f"rolling_metrics_{model_toggle}.json"
    if rolling_path.exists():
        with open(rolling_path) as f:
            rolling = json.load(f)
        if rolling:
            st.markdown("#### 📈 Rolling MAE Across Season")
            st.markdown('<div class="insight-box"><span class="insight-text">This chart shows how model accuracy changes as more training data becomes available. Each point represents a train/test split where the model is trained on earlier rounds and tested on later ones.</span></div>', unsafe_allow_html=True)
            rdf = pd.DataFrame(rolling)
            rdf["split"] = [f"R{r['test_rounds'][0]}–{r['test_rounds'][-1]}" for r in rolling]
            chart = alt.Chart(rdf).mark_line(point=alt.OverlayMarkDef(filled=True, size=80), strokeWidth=3, color="#e10600").encode(
                x=alt.X("split:N", title="Test Rounds", sort=None),
                y=alt.Y("mae:Q", title="MAE (seconds)", scale=alt.Scale(zero=True)),
                tooltip=["split", alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("rmse:Q", format=".2f")],
            ).properties(height=350).configure_axis(
                labelColor="#888", titleColor="#aaa", gridColor="#222"
            ).configure_view(strokeWidth=0)
            st.altair_chart(chart, use_container_width=True)

    # Per-compound
    compound_path = metrics_dir / f"mae_by_compound_{model_toggle}.csv"
    if compound_path.exists():
        cdf = pd.read_csv(compound_path)
        st.markdown("#### 🏎️ Accuracy by Tire Compound")
        cdf["color"] = cdf["Compound"].map(COMPOUND_COLORS)
        chart = alt.Chart(cdf).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Compound:N", sort="-y"),
            y=alt.Y("mae:Q", title="MAE (seconds)"),
            color=alt.Color("Compound:N", scale=alt.Scale(domain=list(COMPOUND_COLORS.keys()), range=list(COMPOUND_COLORS.values())), legend=None),
            tooltip=["Compound", alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("rmse:Q", format=".2f"), "n"],
        ).properties(height=300).configure_axis(labelColor="#888", titleColor="#aaa", gridColor="#222").configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)

    # Per-round
    round_path = metrics_dir / f"mae_by_round_{model_toggle}.csv"
    if round_path.exists():
        rdf = pd.read_csv(round_path).sort_values("RoundNumber")
        st.markdown("#### 🗓️ Accuracy by Race Round")
        chart = alt.Chart(rdf).mark_bar(cornerRadiusEnd=6, color="#e10600").encode(
            x=alt.X("RoundNumber:O", title="Round"),
            y=alt.Y("mae:Q", title="MAE (seconds)"),
            tooltip=["RoundNumber", alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("rmse:Q", format=".2f"), "n"],
        ).properties(height=300).configure_axis(labelColor="#888", titleColor="#aaa", gridColor="#222").configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)

    # Figures gallery
    figs = list(figures_dir.glob(f"*{model_toggle}*.png"))
    if figs:
        st.markdown("#### 🖼️ Evaluation Plots")
        fig_cols = st.columns(min(len(figs), 3))
        for i, fig in enumerate(figs):
            with fig_cols[i % len(fig_cols)]:
                st.image(str(fig), caption=fig.stem.replace("_", " ").title(), use_container_width=True)
