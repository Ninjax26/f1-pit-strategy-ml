"""UI helpers for the F1 strategy simulator."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from ui_styles import COMPOUND_CSS, COMPOUND_COLORS
from three_components import render_particle_hero, render_live_telemetry

FIGURES_DIR = Path("figures")


def format_race_time(seconds: float) -> str:
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
    st.markdown("### Workflow")
    steps = [
        ("Historical Race Data", "Raw lap times, tire compounds, weather, and track state from a selected race weekend."),
        ("Feature Engineering", "The pipeline builds lap-level features such as tyre age, track status, and race-normalized targets."),
        ("Machine Learning Model", "A trained regressor predicts lap-by-lap race pace for candidate strategies."),
        ("Monte Carlo Simulation", "The app perturbs those predictions to reflect uncertainty and pit-stop variability."),
        ("Strategy Evaluation", "Each candidate strategy is ranked by expected race time and uncertainty band."),
        ("Recommended Pit Strategy", "The best-ranked strategy is shown with its pit window and confidence range."),
    ]
    cols = st.columns(len(steps))
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="min-height:180px;">
                <div style="color:#e10600;font-size:0.8rem;font-weight:700;letter-spacing:0.2rem;text-transform:uppercase;">{title}</div>
                <div style="color:#ccc;font-size:0.92rem;line-height:1.5;margin-top:0.5rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)


def render_model_comparison(metrics: dict):
    st.markdown("### Model Comparison")
    col1, col2 = st.columns(2)
    model_specs = [
        ("hgb", "HGB (Gradient Boosting)", "Best default choice for this project because it captures nonlinear tyre-degradation patterns better than a linear baseline.", "Slightly less transparent and can be more sensitive to the feature set and training window.", "Best default"),
        ("ridge", "Ridge Regression", "Fast, stable, and interpretable as a baseline comparator.", "Less flexible for nonlinear pace changes and usually trails HGB on this task.", "Baseline comparator"),
    ]
    for col, (name, label, strength, weakness, default_note) in zip([col1, col2], model_specs):
        m = metrics.get(name, {})
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="margin:0 0 0.8rem 0!important;font-size:1rem!important;">{label}</h3>
                <div style="display:flex;gap:1rem;">
                    <div class="metric-card" style="flex:1;">
                        <div class="metric-label">MAE</div>
                        <div class="metric-value">{m.get('mae', 0):.2f}<span class="metric-unit">s</span></div>
                    </div>
                    <div class="metric-card" style="flex:1;">
                        <div class="metric-label">RMSE</div>
                        <div class="metric-value">{m.get('rmse', 0):.2f}<span class="metric-unit">s</span></div>
                    </div>
                </div>
                <ul style="color:#bdbdbd;font-size:0.9rem;line-height:1.6;margin:0.9rem 0 0 1rem;padding:0;">
                    <li><strong>Strength:</strong> {strength}</li>
                    <li><strong>Weakness:</strong> {weakness}</li>
                    <li><strong>Why it matters:</strong> {default_note}</li>
                </ul>
            </div>""", unsafe_allow_html=True)


def render_season_stats(features_df: pd.DataFrame, model_metrics: dict | None, season: int, metrics_dir: Path):
    st.markdown("### Key Statistics")
    n_races = features_df["RoundNumber"].nunique()
    n_drivers = features_df["Driver"].nunique()
    n_laps = len(features_df)
    model_count = len(model_metrics) if isinstance(model_metrics, dict) else 0
    best_mae = None
    if isinstance(model_metrics, dict):
        best_mae = min((m.get("mae") for m in model_metrics.values() if isinstance(m, dict) and m.get("mae") is not None), default=None)
    pit_loss_path = metrics_dir / f"pit_loss_{season}.csv"
    avg_pit_loss = None
    if pit_loss_path.exists():
        pit_loss_df = pd.read_csv(pit_loss_path)
        median_losses = [pd.to_numeric(v, errors="coerce") for v in pit_loss_df.get("pit_loss_median", []) if pd.notna(pd.to_numeric(v, errors="coerce"))]
        if median_losses:
            avg_pit_loss = float(np.mean(median_losses))
    cols = st.columns(3)
    stat_items = [
        ("Total Races", n_races, "Race weekends in the loaded feature set"),
        ("Drivers", n_drivers, "Unique drivers represented in the data"),
        ("Total Laps", f"{n_laps:,}", "Lap rows available for modeling and simulation"),
    ]
    for col, (label, value, helper_text) in zip(cols, stat_items):
        with col:
            st.metric(label, value, help=helper_text)
    cols2 = st.columns(3)
    for col, (label, value, helper_text) in zip(cols2, [
        ("Models Available", model_count, "Model artifacts available for the selected season"),
        ("Best MAE", f"{best_mae:.2f}s" if best_mae is not None else "N/A", "Lowest MAE among the available model metrics"),
        ("Average Pit Loss", f"{avg_pit_loss:.1f}s" if avg_pit_loss is not None else "N/A", "Average median pit-loss estimate from the pit-loss dataset"),
    ]):
        with col:
            st.metric(label, value, help=helper_text)


def render_stint_bar(stints: list, total_laps: int) -> str:
    segments = ""
    for compound, length in stints:
        compound_upper = compound.upper() if isinstance(compound, str) else compound
        css_class = COMPOUND_CSS.get(compound_upper, "stint-hard")
        pct = (length / total_laps) * 100
        segments += f'<div class="stint-segment {css_class}" style="width:{pct}%;" title="{compound_upper}: {length} laps">{compound_upper[0]} {length}L</div>'
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
        risk_score = format_race_time(best_row.get("risk_adjusted_time_s", time_val))
        detail = f'<div class="strat-detail">Outcome range: {p10} — {p90} (P10–P90) · Conservative score: {risk_score}</div>'

    st.markdown(f"""
    <div class="best-strategy-card">
        <div style="color:#81c784;font-size:0.85rem;text-transform:uppercase;letter-spacing:2px;margin-bottom:0.5rem;">Optimal Strategy</div>
        <div class="strat-name">{best_row["strategy"]}</div>
        <div class="strat-time">{time_str}</div>
        {detail}
        <div style="max-width:500px;margin:1rem auto 0;">{stint_bar}</div>
        <div style="margin-top:0.5rem;color:#666;font-size:0.75rem;">
            {best_row["stops"]} stop{"s" if best_row["stops"] != 1 else ""} · {n_sims} simulation{"s" if n_sims > 1 else ""}
        </div>
    </div>""", unsafe_allow_html=True)


def generate_insights(results: pd.DataFrame, n_sims: int) -> list[str]:
    insights = []
    is_mc = n_sims > 1
    time_key = "risk_adjusted_time_s"

    if len(results) < 2:
        return insights

    best = results.iloc[0]
    second = results.iloc[1]
    gap = second[time_key] - best[time_key]
    insights.append(f"The top strategy leads the next option by {gap:.1f}s on the conservative ranking score.")

    stop_counts = results.head(5)["stops"].value_counts()
    dominant = stop_counts.idxmax()
    insights.append(f"{dominant}-stop strategies dominate the top 5 rankings for this race.")

    if len(results) >= 5:
        spread = results.iloc[4][time_key] - best[time_key]
        insights.append(f"The gap between rank 1 and rank 5 is {spread:.1f}s — {'a tight field' if spread < 5 else 'a significant difference'}.")

    return insights


def render_insights(insights: list[str]):
    if not insights:
        return
    text = "<br>".join(f"• {i}" for i in insights)
    st.markdown(f"""
    <div class="insight-box">
        <strong style="color:#7eb8da;">Strategy Insights</strong>
        <div style="margin-top:0.5rem;">{text}</div>
    </div>""", unsafe_allow_html=True)


def render_strategy_table(results: pd.DataFrame, n_sims: int, total_laps: int):
    is_mc = n_sims > 1
    display = results.head(15).copy()

    if is_mc:
        display["Rank"] = range(1, len(display) + 1)
        best_time = display["risk_adjusted_time_s"].iloc[0]
        display["Delta to Best"] = display["risk_adjusted_time_s"] - best_time
        display["Mean Time"] = display["total_time_mean_s"].apply(format_race_time)
        display["Conservative Score"] = display["risk_adjusted_time_s"].apply(format_race_time)
        display["P10"] = display["total_time_p10_s"].apply(format_race_time)
        display["P90"] = display["total_time_p90_s"].apply(format_race_time)
        display["Delta"] = display["Delta to Best"].apply(lambda x: f"+{x:.1f}s" if x > 0 else "—")
        show_cols = ["Rank", "strategy", "stops", "Mean Time", "Conservative Score", "P10", "P90", "unsupported_laps", "Delta"]
    else:
        display["Rank"] = range(1, len(display) + 1)
        best_time = display["risk_adjusted_time_s"].iloc[0]
        display["Delta to Best"] = display["risk_adjusted_time_s"] - best_time
        display["Total Time"] = display["total_time_s"].apply(format_race_time)
        display["Conservative Score"] = display["risk_adjusted_time_s"].apply(format_race_time)
        display["Delta"] = display["Delta to Best"].apply(lambda x: f"+{x:.1f}s" if x > 0 else "—")
        show_cols = ["Rank", "strategy", "stops", "Total Time", "Conservative Score", "unsupported_laps", "Delta"]

    def highlight_best(row):
        if row.name == 0:
            return ["font-weight: bold; background-color: rgba(225, 6, 0, 0.18); color: white;"] * len(row)
        return [""] * len(row)

    styled = display[show_cols].style.apply(highlight_best, axis=1)
    st.dataframe(styled, width='stretch', hide_index=True)


def render_recommendation_explanation(best_row, results: pd.DataFrame, n_sims: int):
    is_mc = n_sims > 1
    time_key = "risk_adjusted_time_s"
    if results.empty:
        return

    best_time = best_row[time_key]
    second_time = results.iloc[1][time_key] if len(results) > 1 else None
    spread = None
    if second_time is not None:
        spread = second_time - best_time
    p10 = best_row.get("total_time_p10_s")
    p90 = best_row.get("total_time_p90_s")
    uncertainty_spread = None
    if p10 is not None and p90 is not None:
        uncertainty_spread = p90 - p10

    bullets = []
    bullets.append("This strategy has the lowest conservative score after expected time, uncertainty, and historical-support penalties are considered.")
    bullets.append(f"It uses {best_row['stops']} stop{'s' if best_row['stops'] != 1 else ''}, which keeps the pit-loss overhead in line with the other candidates.")
    if second_time is not None and spread is not None:
        bullets.append(f"It beats the next-best option by {spread:.1f}s, so it remains ahead even after accounting for the ranking gap.")
    if uncertainty_spread is not None:
        if uncertainty_spread <= 8:
            bullets.append("Its P10–P90 range is tight, so the recommendation is stable across most simulations.")
        elif uncertainty_spread <= 20:
            bullets.append("Its P10–P90 range is moderate, so the recommendation is reasonably stable but still sensitive to variability.")
        else:
            bullets.append("Its P10–P90 range is wider, so the recommendation is more sensitive to simulation noise and uncertainty.")
    if is_mc and "pit_loss_mean_s" in best_row.index:
        bullets.append(f"The simulation includes pit-loss variability with an average pit-loss contribution of {best_row['pit_loss_mean_s']:.1f}s.")
    if best_row.get("unsupported_laps", 0) == 0:
        bullets.append("Every stint remains within the historical tyre-life support limit used by the simulator.")

    st.markdown("#### Why was this strategy recommended?")
    st.markdown("<ul style='color:#d0d0d0;line-height:1.7;margin-top:0.4rem;'>" + "".join(f"<li>{b}</li>" for b in bullets[:6]) + "</ul>", unsafe_allow_html=True)

    if uncertainty_spread is not None or second_time is not None:
        st.markdown("#### How confident is this recommendation?")
        if uncertainty_spread is not None and uncertainty_spread <= 8 and second_time is not None and spread is not None and spread >= 5:
            confidence = "High confidence — most simulations produced similar race times and the recommended strategy stayed clearly ahead of the next-best option."
        elif uncertainty_spread is not None and uncertainty_spread <= 20 and second_time is not None and spread is not None and spread >= 2:
            confidence = "Medium confidence — the race-time range is moderate and several strategies remained close together."
        else:
            confidence = "Low confidence — there is meaningful spread across the simulations and the ranking is still sensitive to noise."
        st.markdown(f"<div style='color:#ccc;padding:0.8rem 1rem;border-left:3px solid #7eb8da;background:rgba(126,184,218,0.08);'> {confidence}</div>", unsafe_allow_html=True)


def render_stint_gallery(results: pd.DataFrame, total_laps: int, top_n: int = 8):
    st.markdown("#### Strategy Visual Breakdown")
    for i, row in results.head(top_n).iterrows():
        stints = json.loads(row["stints"]) if isinstance(row["stints"], str) else row["stints"]
        bar = render_stint_bar(stints, total_laps)
        rank = results.index.get_loc(i) + 1
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin:0.4rem 0;">
            <div style="min-width:160px;color:#888;font-size:0.8rem;">
                #{rank} {row["strategy"][:20]}
            </div>
            <div style="flex:1;">{bar}</div>
        </div>""", unsafe_allow_html=True)


def render_feature_importance(metrics_dir: Path, model_name: str = "hgb"):
    import altair as alt

    fi_path = metrics_dir / f"feature_importance_{model_name}.csv"
    if not fi_path.exists():
        st.info("Feature importance data not available. Run evaluation pipeline to generate it.")
        return

    st.markdown("### Feature Importance")

    fi_df = pd.read_csv(fi_path)
    top_fi = fi_df.sort_values("importance", ascending=False).head(15)

    chart = alt.Chart(top_fi).mark_bar(cornerRadiusEnd=4, color="#7eb8da").encode(
        x=alt.X("importance:Q", title="Importance Score"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")]
    ).properties(height=400).configure_axis(
        labelColor="#888", titleColor="#aaa", gridColor="#222"
    ).configure_view(strokeWidth=0)

    st.altair_chart(chart, width='stretch')

    st.markdown(
        '<div style="margin-top:0.5rem;color:#ccc;font-size:0.9rem;border-left:3px solid #7eb8da;padding-left:1rem;">Feature importance measures how much each input contributes to predicted lap times.</div>',
        unsafe_allow_html=True,
    )


def render_model_performance_tab(metrics_dir: Path, figures_dir: Path):
    import altair as alt

    st.markdown("### Model Accuracy")
    model_toggle = st.radio("Select Model", ["hgb", "ridge"], horizontal=True, index=0, key="perf_model_toggle")
    label_map = {"hgb": "HGB (Gradient Boosting)", "ridge": "Ridge Regression"}
    st.markdown(f"**Results for: {label_map[model_toggle]}**")
    render_feature_importance(metrics_dir, model_toggle)
    st.markdown("---")

    sections_rendered = 0

    rolling_path = metrics_dir / f"rolling_metrics_{model_toggle}.json"
    if rolling_path.exists():
        with open(rolling_path) as f:
            rolling = json.load(f)
        if rolling:
            st.markdown("#### Rolling MAE Across Season")
            rdf = pd.DataFrame(rolling)
            rdf["split"] = [f"R{r['test_rounds'][0]}–{r['test_rounds'][-1]}" for r in rolling]
            chart = alt.Chart(rdf).mark_line(point=alt.OverlayMarkDef(filled=True, size=80), strokeWidth=3, color="#e10600").encode(
                x=alt.X("split:N", title="Test Rounds", sort=None),
                y=alt.Y("mae:Q", title="MAE (seconds)", scale=alt.Scale(zero=True)),
                tooltip=["split", alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("rmse:Q", format=".2f")],
            ).properties(height=350).configure_axis(
                labelColor="#888", titleColor="#aaa", gridColor="#222"
            ).configure_view(strokeWidth=0)
            st.altair_chart(chart, width='stretch')
            sections_rendered += 1

    compound_path = metrics_dir / f"mae_by_compound_{model_toggle}.csv"
    if compound_path.exists():
        cdf = pd.read_csv(compound_path)
        st.markdown("#### Accuracy by Tire Compound")
        cdf["color"] = cdf["Compound"].map(COMPOUND_COLORS)
        chart = alt.Chart(cdf).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Compound:N", sort="-y"),
            y=alt.Y("mae:Q", title="MAE (seconds)"),
            color=alt.Color("Compound:N", scale=alt.Scale(domain=list(COMPOUND_COLORS.keys()), range=list(COMPOUND_COLORS.values())), legend=None),
            tooltip=["Compound", alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("rmse:Q", format=".2f"), "n"],
        ).properties(height=300).configure_axis(labelColor="#888", titleColor="#aaa", gridColor="#222").configure_view(strokeWidth=0)
        st.altair_chart(chart, width='stretch')
        sections_rendered += 1

    round_path = metrics_dir / f"mae_by_round_{model_toggle}.csv"
    if round_path.exists():
        rdf = pd.read_csv(round_path).sort_values("RoundNumber")
        st.markdown("#### Accuracy by Race Round")
        chart = alt.Chart(rdf).mark_bar(cornerRadiusEnd=6, color="#e10600").encode(
            x=alt.X("RoundNumber:O", title="Round"),
            y=alt.Y("mae:Q", title="MAE (seconds)"),
            tooltip=["RoundNumber", alt.Tooltip("mae:Q", format=".2f"), alt.Tooltip("rmse:Q", format=".2f"), "n"],
        ).properties(height=300).configure_axis(labelColor="#888", titleColor="#aaa", gridColor="#222").configure_view(strokeWidth=0)
        st.altair_chart(chart, width='stretch')
        sections_rendered += 1

    figs = list(figures_dir.glob(f"*{model_toggle}*.png"))
    if figs:
        st.markdown("#### Evaluation Plots")
        fig_cols = st.columns(min(len(figs), 3))
        for i, fig in enumerate(figs):
            with fig_cols[i % len(fig_cols)]:
                st.image(str(fig), caption=fig.stem.replace("_", " ").title(), width='stretch')
        sections_rendered += 1

    if sections_rendered == 0:
        st.info(
            "No evaluation data found for this model. "
            "Run the evaluation pipeline to generate metrics: "
            "`python -m src.models.evaluate`"
        )
