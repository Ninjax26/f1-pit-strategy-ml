"""CSS styles for the F1 Pit Strategy Simulator app."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

/* === Global === */
.main { background: linear-gradient(180deg, #0a0a0a 0%, #111 50%, #0a0a0a 100%); }
.stApp { background: #0a0a0a; color: #e0e0e0; }
h1, h2, h3 { font-family: 'Orbitron', sans-serif !important; }
h1 { color: #e10600 !important; }
h2 { color: #ff3333 !important; font-size: 1.4rem !important; }
h3 { color: #ff6666 !important; font-size: 1.15rem !important; }
p, span, label, li { font-family: 'Inter', sans-serif !important; }

/* === Hero === */
.hero-container {
    text-align: center; padding: 2.5rem 1rem 1.5rem; position: relative; overflow: hidden;
    border-bottom: 3px solid #e10600; margin-bottom: 2rem;
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0000 50%, #0a0a0a 100%);
}
.hero-container h1 { font-size: 2.6rem !important; letter-spacing: 2px; margin-bottom: 0.5rem !important;
    background: linear-gradient(90deg, #e10600, #ff4444, #e10600); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; animation: shimmer 3s ease-in-out infinite; background-size: 200% 100%; }
@keyframes shimmer { 0%,100% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } }
.hero-container p { color: #888; font-size: 1.05rem; margin: 0; }
.hero-glow { position: absolute; top: -50%; left: 50%; transform: translateX(-50%); width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(225,6,0,0.08) 0%, transparent 70%); pointer-events: none; }

/* === Glass Cards === */
.glass-card {
    background: linear-gradient(135deg, rgba(30,30,30,0.9) 0%, rgba(20,20,20,0.95) 100%);
    border: 1px solid rgba(225,6,0,0.15); border-radius: 16px; padding: 1.5rem;
    backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.3); transition: all 0.3s ease;
    margin-bottom: 1rem;
}
.glass-card:hover { border-color: rgba(225,6,0,0.4); box-shadow: 0 8px 32px rgba(225,6,0,0.1); transform: translateY(-2px); }
.glass-card h3 { margin-top: 0 !important; }

/* === Metric Cards === */
.metric-card {
    background: linear-gradient(135deg, #1a1a1a 0%, #222 100%); border-left: 4px solid #e10600;
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 0.8rem;
}
.metric-card .metric-label { color: #888; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.3rem; }
.metric-card .metric-value { color: #fff; font-size: 1.8rem; font-weight: 700; font-family: 'Orbitron', sans-serif !important; }
.metric-card .metric-unit { color: #666; font-size: 0.8rem; margin-left: 4px; }

/* === Winner Badge === */
.winner-badge { display: inline-block; background: linear-gradient(90deg, #ffd700, #ffaa00); color: #000;
    padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.8rem; font-weight: 700; animation: pulse 2s ease-in-out infinite; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 5px rgba(255,215,0,0.3); } 50% { box-shadow: 0 0 20px rgba(255,215,0,0.6); } }

/* === Best Strategy Box === */
.best-strategy-card {
    background: linear-gradient(135deg, #0d2b0d 0%, #1a3d1a 50%, #0d2b0d 100%);
    border: 2px solid #2e7d32; border-radius: 16px; padding: 2rem; margin: 1.5rem 0;
    text-align: center; box-shadow: 0 0 30px rgba(46,125,50,0.15);
}
.best-strategy-card .trophy { font-size: 3rem; margin-bottom: 0.5rem; }
.best-strategy-card .strat-name { color: #4caf50; font-family: 'Orbitron', sans-serif; font-size: 1.3rem; margin: 0.5rem 0; }
.best-strategy-card .strat-time { color: #fff; font-size: 2.5rem; font-weight: 800; font-family: 'Orbitron', sans-serif; }
.best-strategy-card .strat-detail { color: #81c784; font-size: 0.9rem; margin-top: 0.5rem; }

/* === Tire Stint Bar === */
.stint-bar { display: flex; height: 36px; border-radius: 8px; overflow: hidden; margin: 0.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
.stint-segment { display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 600;
    color: #000; transition: all 0.3s ease; position: relative; }
.stint-segment:hover { filter: brightness(1.2); }
.stint-soft { background: linear-gradient(180deg, #FF1801, #cc1300); }
.stint-medium { background: linear-gradient(180deg, #FFC906, #d4a800); }
.stint-hard { background: linear-gradient(180deg, #FFFFFF, #ddd); }
.stint-intermediate { background: linear-gradient(180deg, #43B02A, #358c22); color: #fff; }
.stint-wet { background: linear-gradient(180deg, #0067FF, #0052cc); color: #fff; }

/* === Flow Steps === */
.flow-step { text-align: center; padding: 1rem; }
.flow-icon { font-size: 2.5rem; margin-bottom: 0.5rem; }
.flow-title { color: #e10600; font-family: 'Orbitron', sans-serif; font-size: 0.9rem; margin-bottom: 0.3rem; }
.flow-desc { color: #999; font-size: 0.8rem; line-height: 1.4; }
.flow-arrow { display: flex; align-items: center; justify-content: center; color: #e10600; font-size: 2rem; }

/* === Buttons === */
.stButton > button { background: linear-gradient(90deg, #e10600 0%, #b80500 100%) !important; color: white !important;
    font-weight: 600 !important; border: none !important; padding: 0.7rem 2rem !important; border-radius: 10px !important;
    font-family: 'Orbitron', sans-serif !important; letter-spacing: 1px; transition: all 0.3s ease !important; }
.stButton > button:hover { background: linear-gradient(90deg, #ff1a0d 0%, #e10600 100%) !important;
    box-shadow: 0 0 25px rgba(225,6,0,0.5) !important; transform: translateY(-1px) !important; }

/* === Metrics Override === */
.stMetric { background: linear-gradient(135deg, #1a1a1a 0%, #252525 100%) !important; padding: 1rem !important;
    border-radius: 12px !important; border-left: 4px solid #e10600 !important; }
.stMetric label { color: #999 !important; font-family: 'Inter', sans-serif !important; text-transform: uppercase; font-size: 0.75rem !important; letter-spacing: 0.5px; }
.stMetric [data-testid="stMetricValue"] { color: #fff !important; font-family: 'Orbitron', sans-serif !important; }

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 2px solid #222; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #888; font-family: 'Orbitron', sans-serif;
    border-radius: 8px 8px 0 0; padding: 0.6rem 1.2rem; border: 1px solid transparent; transition: all 0.3s ease; }
.stTabs [data-baseweb="tab"]:hover { color: #e10600; background: rgba(225,6,0,0.05); }
.stTabs [aria-selected="true"] { color: #e10600 !important; border-bottom: 3px solid #e10600 !important; background: rgba(225,6,0,0.08) !important; }

/* === Insight Box === */
.insight-box { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid rgba(100,149,237,0.3);
    border-radius: 12px; padding: 1.2rem 1.5rem; margin: 1rem 0; }
.insight-box .insight-icon { font-size: 1.3rem; margin-right: 0.5rem; }
.insight-box .insight-text { color: #b8c9e8; font-size: 0.95rem; line-height: 1.5; }

/* ═══════════════════════════════════════ */
/*   SIDEBAR — High Contrast Overhaul     */
/* ═══════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d0d 0%, #0f0f0f 50%, #0d0d0d 100%) !important;
    border-right: 1px solid rgba(225,6,0,0.2) !important;
}

/* --- Section headers: bright white with red accent --- */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.95rem !important;
    color: #fff !important;
    -webkit-text-fill-color: #fff !important;
    background: none !important;
    letter-spacing: 1px;
    text-transform: uppercase;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid rgba(225,6,0,0.4);
    margin-bottom: 0.6rem !important;
}

/* --- All labels (dropdowns, sliders, checkboxes) --- */
section[data-testid="stSidebar"] label {
    color: #ccc !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* --- Captions --- */
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #bbb !important;
    font-size: 0.82rem !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #bbb !important;
}

/* --- All paragraph text in sidebar --- */
section[data-testid="stSidebar"] p {
    color: #ccc !important;
}

/* --- Sidebar markdown override for strong tags --- */
section[data-testid="stSidebar"] strong {
    color: #fff !important;
    font-weight: 700 !important;
}

/* --- Styled info badges in sidebar --- */
.sidebar-info {
    display: flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, rgba(225,6,0,0.08), rgba(225,6,0,0.03));
    border: 1px solid rgba(225,6,0,0.2);
    border-radius: 8px; padding: 0.5rem 0.8rem; margin: 0.3rem 0 0.8rem 0;
}
.sidebar-info .si-icon { font-size: 1rem; }
.sidebar-info .si-text { color: #ddd; font-size: 0.82rem; line-height: 1.3; font-family: 'Inter', sans-serif; }
.sidebar-info .si-value { color: #ff8a80; font-weight: 600; font-family: 'Orbitron', sans-serif; font-size: 0.8rem; }

/* --- Sidebar section dividers --- */
section[data-testid="stSidebar"] hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(225,6,0,0.3), transparent) !important;
    margin: 1rem 0 !important;
}

/* --- Sidebar selectbox & slider --- */
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background: #1a1a1a !important; border: 1px solid #333 !important; border-radius: 8px !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"]:hover {
    border-color: rgba(225,6,0,0.5) !important;
}
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #e10600 !important;
}

/* --- Sidebar checkbox text --- */
section[data-testid="stSidebar"] .stCheckbox label span {
    color: #ccc !important;
}

/* --- Sidebar number input --- */
section[data-testid="stSidebar"] input[type="number"] {
    color: #fff !important; background: #1a1a1a !important;
}
</style>
"""

COMPOUND_COLORS = {"SOFT": "#FF1801", "MEDIUM": "#FFC906", "HARD": "#FFFFFF", "INTERMEDIATE": "#43B02A", "WET": "#0067FF"}
COMPOUND_CSS = {"SOFT": "stint-soft", "MEDIUM": "stint-medium", "HARD": "stint-hard", "INTERMEDIATE": "stint-intermediate", "WET": "stint-wet"}
