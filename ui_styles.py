"""CSS styles for the F1 strategy simulator."""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global */
.stApp { background: #0a0a0a; color: #e0e0e0; }
h1, h2, h3 { font-family: 'Inter', sans-serif !important; }
h1 { color: #e10600 !important; font-weight: 700; }
h2 { color: #ff3333 !important; font-size: 1.3rem !important; }
h3 { color: #ff6666 !important; font-size: 1.1rem !important; }

/* Cards */
.glass-card {
    background: #151515;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Metric Cards */
.metric-card {
    background: #1a1a1a;
    border-left: 3px solid #e10600;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.metric-card .metric-label { color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.2rem; }
.metric-card .metric-value { color: #fff; font-size: 1.6rem; font-weight: 700; }
.metric-card .metric-unit { color: #666; font-size: 0.8rem; margin-left: 4px; }

/* Best Strategy Box */
.best-strategy-card {
    background: #0d2b0d;
    border: 1px solid #2e7d32;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
}
.best-strategy-card .strat-name { color: #4caf50; font-size: 1.2rem; margin: 0.5rem 0; }
.best-strategy-card .strat-time { color: #fff; font-size: 2rem; font-weight: 700; }
.best-strategy-card .strat-detail { color: #81c784; font-size: 0.85rem; margin-top: 0.5rem; }

/* Tire Stint Bar */
.stint-bar { display: flex; height: 32px; border-radius: 6px; overflow: hidden; margin: 0.5rem 0; }
.stint-segment { display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 600;
    color: #000; }
.stint-soft { background: linear-gradient(180deg, #FF1801, #cc1300); }
.stint-medium { background: linear-gradient(180deg, #FFC906, #d4a800); }
.stint-hard { background: linear-gradient(180deg, #FFFFFF, #ddd); }
.stint-intermediate { background: linear-gradient(180deg, #43B02A, #358c22); color: #fff; }
.stint-wet { background: linear-gradient(180deg, #0067FF, #0052cc); color: #fff; }

/* Flow Steps */
.flow-step { text-align: center; padding: 1rem; }
.flow-title { color: #e10600; font-weight: 600; font-size: 0.85rem; margin-bottom: 0.3rem; }
.flow-desc { color: #999; font-size: 0.8rem; line-height: 1.4; }
.flow-arrow { display: flex; align-items: center; justify-content: center; color: #e10600; font-size: 1.5rem; }

/* Buttons */
.stButton > button { background: linear-gradient(90deg, #e10600 0%, #b80500 100%) !important; color: white !important;
    font-weight: 600 !important; border: none !important; padding: 0.6rem 1.5rem !important; border-radius: 8px !important; }

/* Metrics Override */
.stMetric { background: #1a1a1a !important; padding: 1rem !important;
    border-radius: 8px !important; border-left: 3px solid #e10600 !important; }
.stMetric label { color: #999 !important; text-transform: uppercase; font-size: 0.75rem !important; }
.stMetric [data-testid="stMetricValue"] { color: #fff !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #222; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #888;
    border-radius: 8px 8px 0 0; padding: 0.5rem 1rem; border: 1px solid transparent; }
.stTabs [data-baseweb="tab"]:hover { color: #e10600; }
.stTabs [aria-selected="true"] { color: #e10600 !important; border-bottom: 2px solid #e10600 !important; }

/* Insight Box */
.insight-box { background: #1a1a2e; border: 1px solid rgba(100,149,237,0.2);
    border-radius: 8px; padding: 1rem 1.2rem; margin: 1rem 0; }
.insight-box .insight-text { color: #b8c9e8; font-size: 0.9rem; line-height: 1.5; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d0d0d !important;
    border-right: 1px solid rgba(225,6,0,0.15) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    font-size: 0.9rem !important;
    color: #fff !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(225,6,0,0.3);
    margin-bottom: 0.6rem !important;
}
section[data-testid="stSidebar"] label { color: #ccc !important; font-size: 0.85rem !important; }
section[data-testid="stSidebar"] p { color: #ccc !important; }
section[data-testid="stSidebar"] strong { color: #fff !important; }
.sidebar-info {
    display: flex; align-items: center; gap: 0.5rem;
    background: rgba(225,6,0,0.05);
    border: 1px solid rgba(225,6,0,0.15);
    border-radius: 6px; padding: 0.4rem 0.7rem; margin: 0.3rem 0 0.8rem 0;
}
.sidebar-info .si-text { color: #ddd; font-size: 0.8rem; }
.sidebar-info .si-value { color: #ff8a80; font-weight: 600; font-size: 0.8rem; }
section[data-testid="stSidebar"] hr { border: none !important;
    height: 1px !important; background: rgba(225,6,0,0.2) !important;
    margin: 0.8rem 0 !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] { background: #1a1a1a !important; border: 1px solid #333 !important; border-radius: 6px !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] { background: #e10600 !important; }
section[data-testid="stSidebar"] .stCheckbox label span { color: #ccc !important; }
section[data-testid="stSidebar"] input[type="number"] { color: #fff !important; background: #1a1a1a !important; }
</style>
"""

COMPOUND_COLORS = {"SOFT": "#FF1801", "MEDIUM": "#FFC906", "HARD": "#FFFFFF", "INTERMEDIATE": "#43B02A", "WET": "#0067FF"}
COMPOUND_CSS = {"SOFT": "stint-soft", "MEDIUM": "stint-medium", "HARD": "stint-hard", "INTERMEDIATE": "stint-intermediate", "WET": "stint-wet"}
