"""
EcoLoop Dashboard — Hackathon Edition
Optimised for immediate judge comprehension.
Baseline vs AI comparison across energy, comfort, peak demand, carbon, and cost.
"""

import json
import os
import re
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoLoop: AI HVAC Control",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROJECT_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH      = os.path.join(PROJECT_ROOT, "docs",  "results.json")
DECISION_LOG_PATH = os.path.join(PROJECT_ROOT, "logs",  "decision_log.csv")
BASELINE_LOG_PATH = os.path.join(PROJECT_ROOT, "logs",  "baseline_log.csv")
HEALTH_PATH       = os.path.join(PROJECT_ROOT, "logs",  "health.json")
EXPLANATION_PATH  = os.path.join(PROJECT_ROOT, "logs",  "explanations.jsonl")
ERRORS_PATH       = os.path.join(PROJECT_ROOT, "logs",  "ecoloop_errors.log")

# Emission & cost factors
CARBON_KG_PER_KWH   = 0.233   # US average grid
ELECTRICITY_USD_KWH = 0.12    # Average US residential

# Palette
C_AI        = "#3DD6F5"   # cyan — AI system
C_BASELINE  = "#F5793A"   # orange — baseline
C_GREEN     = "#2ECC71"
C_RED       = "#E74C3C"
C_YELLOW    = "#F39C12"
C_PURPLE    = "#9B59B6"
C_BG        = "#0D1117"
C_CARD      = "#161B26"
C_BORDER    = "rgba(255,255,255,0.07)"
C_TEXT      = "#E6EDF3"
C_SUBTEXT   = "#8B949E"

ACTION_COLORS = {
    "off":    "#4A5568",
    "eco":    "#2ECC71",
    "normal": "#3DD6F5",
    "boost":  "#E74C3C",
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: #0D1117; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem; max-width: 1600px; margin: 0 auto; }

/* Hero header */
.hero-banner {
    background: linear-gradient(135deg, #0f2744 0%, #0d1117 60%, #0a1a1a 100%);
    border: 1px solid rgba(61,214,245,0.15);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(61,214,245,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #3DD6F5, #7C6CF8, #3DD6F5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0; letter-spacing: -0.5px;
}
.hero-sub {
    color: #8B949E; font-size: 0.95rem; margin: 0; font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(61,214,245,0.12); border: 1px solid rgba(61,214,245,0.3);
    border-radius: 20px; padding: 4px 14px; font-size: 0.78rem;
    color: #3DD6F5; font-weight: 600; margin-right: 8px; margin-top: 10px;
}

/* Metric cards */
.kpi-grid { display: grid; grid-template-columns: repeat(6,1fr); gap: 14px; margin-bottom: 28px; }
.kpi-card {
    background: linear-gradient(145deg, #161B26, #0F1319);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 20px 18px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.18s, border-color 0.18s;
    position: relative; overflow: hidden;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(61,214,245,0.3); }
.kpi-card::after {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    border-radius: 16px 16px 0 0;
}
.kpi-card.green::after  { background: #2ECC71; }
.kpi-card.cyan::after   { background: #3DD6F5; }
.kpi-card.orange::after { background: #F5793A; }
.kpi-card.red::after    { background: #E74C3C; }
.kpi-card.purple::after { background: #9B59B6; }
.kpi-card.yellow::after { background: #F39C12; }

.kpi-label  { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: #8B949E; margin-bottom: 8px; }
.kpi-value  { font-size: 2.0rem; font-weight: 800; line-height: 1; margin-bottom: 4px; }
.kpi-sub    { font-size: 0.74rem; color: #8B949E; line-height: 1.4; }
.kpi-delta  { font-size: 0.8rem; font-weight: 600; margin-top: 6px; }

/* Section headers */
.section-title {
    font-size: 1.1rem; font-weight: 700; color: #E6EDF3;
    margin: 28px 0 14px; display: flex; align-items: center; gap: 8px;
}
.section-title::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.1), transparent);
}

/* Reasoning log table */
.reasoning-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.reasoning-table th {
    background: #161B26; color: #8B949E; font-weight: 600;
    padding: 10px 12px; text-align: left; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.reasoning-table td { padding: 9px 12px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #E6EDF3; vertical-align: top; }
.reasoning-table tr:hover td { background: rgba(255,255,255,0.025); }
.badge { display:inline-block; border-radius:4px; padding:2px 8px; font-size:0.72rem; font-weight:600; }
.badge-off    { background:rgba(74,85,104,0.4);  color:#A0AEC0; }
.badge-eco    { background:rgba(46,204,113,0.2); color:#2ECC71; }
.badge-normal { background:rgba(61,214,245,0.2); color:#3DD6F5; }
.badge-boost  { background:rgba(231,76,60,0.2);  color:#E74C3C; }
.badge-ok     { background:rgba(46,204,113,0.2); color:#2ECC71; }
.badge-fail   { background:rgba(231,76,60,0.2);  color:#E74C3C; }
.badge-corr   { background:rgba(243,156,18,0.2); color:#F39C12; }

.conf-bar-bg { background:rgba(255,255,255,0.08); border-radius:4px; height:6px; width:80px; display:inline-block; overflow:hidden; vertical-align:middle; margin-left:6px; }
.conf-bar-fg { height:6px; border-radius:4px; background: linear-gradient(90deg,#3DD6F5,#2ECC71); }

/* Alert boxes */
.alert-box {
    border-radius: 10px; padding: 14px 18px; margin-bottom: 14px;
    display: flex; align-items: flex-start; gap: 12px;
}
.alert-box.green { background: rgba(46,204,113,0.1); border:1px solid rgba(46,204,113,0.3); }
.alert-box.red   { background: rgba(231,76,60,0.1); border:1px solid rgba(231,76,60,0.3); }
.alert-box.cyan  { background: rgba(61,214,245,0.08); border:1px solid rgba(61,214,245,0.2); }
.alert-box-icon  { font-size: 1.4rem; line-height:1; }
.alert-box-text  { color:#E6EDF3; font-size:0.88rem; }
.alert-box-title { font-weight:700; font-size:0.95rem; margin-bottom:2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)

@st.cache_data(ttl=30)
def load_decision_log():
    if not os.path.exists(DECISION_LOG_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(DECISION_LOG_PATH, low_memory=False, on_bad_lines="skip")
    except Exception:
        try:
            records = []
            with open(DECISION_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("timestamp") and row.get("timestamp") != "timestamp":
                        records.append(row)
            df = pd.DataFrame(records)
        except Exception:
            return pd.DataFrame()

    df = _parse_timestamp(df)
    # Ensure numeric columns
    for col in ["zone_temp","heating_sp","cooling_sp","outdoor_temp",
                "coil_speed","confidence","energy_kwh","comfort_deviation",
                "conf_historical","conf_sensor","conf_weather","conf_comfort","conf_stability"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Bool columns
    for col in ["success","llm_ok"]:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False, "true": True, "false": False})
    # Fill optional columns
    for col in ["confidence","energy_kwh","comfort_deviation","outcome","risk_level",
                "expected_savings_pct","rejection_reasoning","violations"]:
        if col not in df.columns:
            df[col] = None
    return df

@st.cache_data(ttl=30)
def load_baseline_log():
    if not os.path.exists(BASELINE_LOG_PATH):
        return pd.DataFrame()
    df = pd.read_csv(BASELINE_LOG_PATH, low_memory=False)
    df = _parse_timestamp(df)
    for col in ["zone_temp","heating_sp","cooling_sp","outdoor_temp","coil_speed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Compute comfort deviation for baseline
    if "zone_temp" in df.columns and "cooling_sp" in df.columns:
        df["comfort_deviation"] = (
            df[["zone_temp"]].assign(
                cd=lambda x: (x["zone_temp"] - df["cooling_sp"]).clip(lower=0),
                hd=(df["heating_sp"] - df["zone_temp"]).clip(lower=0)
            ).apply(lambda r: max(
                max(0.0, df.loc[r.name, "zone_temp"] - df.loc[r.name, "cooling_sp"]),
                max(0.0, df.loc[r.name, "heating_sp"] - df.loc[r.name, "zone_temp"])
            ), axis=1)
        )
    return df

@st.cache_data(ttl=15)
def load_health():
    if not os.path.exists(HEALTH_PATH):
        return {}
    try:
        with open(HEALTH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

@st.cache_data(ttl=15)
def load_explanations_data():
    if not os.path.exists(EXPLANATION_PATH):
        return []
    records = []
    try:
        with open(EXPLANATION_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return list(reversed(records))  # Newest first

@st.cache_data(ttl=15)
def load_errors_data():
    if not os.path.exists(ERRORS_PATH):
        return []
    records = []
    try:
        with open(ERRORS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return list(reversed(records))  # Newest first

def _parse_timestamp(df):
    if "timestamp" not in df.columns:
        return df
    # Try parsing "MM/DD HH:MM" format
    def safe_parse(ts):
        ts = str(ts)
        m = re.match(r"(\d{2})/(\d{2})\s+(\d{2}):(\d{2})", ts)
        if m:
            mo, d, h, mi = m.groups()
            return pd.Timestamp(f"2024-{int(mo):02d}-{int(d):02d} {int(h):02d}:{int(mi):02d}")
        try:
            return pd.to_datetime(ts)
        except Exception:
            return pd.NaT
    df["dt"] = df["timestamp"].apply(safe_parse)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# Derived Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_derived(r: dict, ai_df: pd.DataFrame, base_df: pd.DataFrame) -> dict:
    base_kwh = r.get("baseline_energy_kwh", 0)
    ai_kwh   = r.get("ai_energy_kwh", 0)

    carbon_saved  = (base_kwh - ai_kwh) * CARBON_KG_PER_KWH
    cost_saved    = (base_kwh - ai_kwh) * ELECTRICITY_USD_KWH
    pct_energy    = r.get("pct_energy_savings", 0)
    pct_peak      = r.get("pct_peak_demand_reduction", 0)

    # Action distribution
    action_dist = {}
    if not ai_df.empty and "action" in ai_df.columns:
        action_dist = ai_df["action"].value_counts().to_dict()

    # Success / failure / recovery counts
    total_decisions  = len(ai_df)
    failed_decisions = 0
    recovered        = 0
    if not ai_df.empty:
        if "success" in ai_df.columns:
            failed_decisions = int((~ai_df["success"].fillna(True)).sum())
        if "outcome" in ai_df.columns:
            recovered = int((ai_df["outcome"] == "CORRECTED").sum())

    # Avg confidence
    avg_conf = 0.0
    if not ai_df.empty and "confidence" in ai_df.columns:
        avg_conf = ai_df["confidence"].dropna().mean()

    # Hourly energy series (proxy from coil_speed)
    ai_energy_series   = pd.Series(dtype=float)
    base_energy_series = pd.Series(dtype=float)
    if not ai_df.empty and "coil_speed" in ai_df.columns and "dt" in ai_df.columns:
        ai_energy_series = (
            ai_df.set_index("dt")["coil_speed"].dropna()
            * (3600 / 1.9) * (600 / 3_600_000)
        )
    if not base_df.empty and "coil_speed" in base_df.columns and "dt" in base_df.columns:
        base_energy_series = (
            base_df.set_index("dt")["coil_speed"].dropna()
            * (3600 / 1.9) * (600 / 3_600_000)
        )

    return {
        "carbon_saved_kg":    round(carbon_saved, 2),
        "cost_saved_usd":     round(cost_saved, 2),
        "pct_energy":         pct_energy,
        "pct_peak":           pct_peak,
        "action_dist":        action_dist,
        "total_decisions":    total_decisions,
        "failed_decisions":   failed_decisions,
        "recovered":          recovered,
        "avg_confidence":     round(avg_conf, 3),
        "ai_energy_series":   ai_energy_series,
        "base_energy_series": base_energy_series,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=C_TEXT, size=12),
    margin=dict(l=8, r=8, t=32, b=8),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(0,0,0,0)"),
)

def apply_base_layout(fig, **kwargs):
    fig.update_layout(**{**LAYOUT_BASE, **kwargs})
    return fig

def card(content: str):
    """Wrapper div for a chart card."""
    st.markdown(
        f'<div style="background:#161B26;border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:16px;padding:20px;margin-bottom:16px;">{content}</div>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

results  = load_results()
ai_df    = load_decision_log()
base_df  = load_baseline_log()
derived  = compute_derived(results, ai_df, base_df)
health_data       = load_health()
explanations_data = load_explanations_data()
errors_data       = load_errors_data()

# ── HERO BANNER ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
    <div>
      <div class="hero-title">EcoLoop: AI-Driven Building Control</div>
      <div class="hero-sub">
        Autonomous LLM controller (qwen2.5:3b via Ollama) vs Rule-Based Baseline &mdash;
        7-day EnergyPlus simulation &middot; Jul 1&ndash;Jul 7
      </div>
      <div style="margin-top:10px;">
        <span class="hero-badge">Multi-Agent System</span>
        <span class="hero-badge">Short-Term Memory</span>
        <span class="hero-badge">Multi-Candidate Planner</span>
        <span class="hero-badge">Confidence Engine</span>
        <span class="hero-badge">Explainable AI</span>
        <span class="hero-badge">Fault-Tolerant</span>
      </div>
    </div>
    <div style="text-align:right;color:#8B949E;font-size:0.82rem;margin-top:8px;">
      <div style="color:#3DD6F5;font-weight:700;font-size:1.1rem;">Phase 5</div>
      <div>Observe &rarr; Reason &rarr; Plan &rarr; Validate &rarr; Act</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI METRIC CARDS ─────────────────────────────────────────────────────────
def _delta_html(val, invert=False, unit=""):
    good = val > 0 if not invert else val < 0
    color = C_GREEN if good else C_RED
    arrow = "▲" if val > 0 else "▼"
    return f'<span style="color:{color};font-size:0.8rem;font-weight:600;">{arrow} {abs(val):.1f}{unit}</span>'

pct_e   = results.get("pct_energy_savings", 0)
pct_p   = results.get("pct_peak_demand_reduction", 0)
ai_kwh  = results.get("ai_energy_kwh", 0)
b_kwh   = results.get("baseline_energy_kwh", 0)
ai_peak = results.get("peak_demand_w_ai", 0)
b_peak  = results.get("peak_demand_w_baseline", 0)
ai_cd   = results.get("comfort_deviation_ai", 0)
b_cd    = results.get("comfort_deviation_baseline", 0)

kpis = [
    {
        "label": "Energy Savings",
        "value": f"{pct_e:.1f}%",
        "sub":   f"AI: {ai_kwh:.1f} kWh vs Base: {b_kwh:.1f} kWh",
        "delta": f'{_delta_html(pct_e, unit="%")}',
        "color": "green" if pct_e > 0 else "red",
    },
    {
        "label": "Peak Demand",
        "value": f"{abs(pct_p):.1f}%",
        "sub":   f"AI: {ai_peak:.0f} W vs Base: {b_peak:.0f} W",
        "delta": f'<span style="color:{C_RED if pct_p < 0 else C_GREEN};font-size:0.8rem;">{"▼ higher" if pct_p < 0 else "▲ lower"}</span>',
        "color": "green" if pct_p >= 0 else "orange",
    },
    {
        "label": "AI Comfort Dev.",
        "value": f"{ai_cd:.3f}°C",
        "sub":   f"Baseline: {b_cd:.3f}°C",
        "delta": f'<span style="color:{C_RED if ai_cd > b_cd else C_GREEN};font-size:0.8rem;">{"▲ wider" if ai_cd > b_cd else "▼ tighter"}</span>',
        "color": "cyan",
    },
    {
        "label": "Carbon Saved",
        "value": f"{abs(derived["carbon_saved_kg"]):.1f} kg",
        "sub":   f"CO₂ {'reduced' if derived['carbon_saved_kg'] > 0 else 'increased'} vs baseline",
        "delta": f'<span style="color:{C_GREEN if derived["carbon_saved_kg"] > 0 else C_RED};">{("▼ reduced" if derived["carbon_saved_kg"] > 0 else "▲ added")}</span>',
        "color": "green" if derived["carbon_saved_kg"] > 0 else "red",
    },
    {
        "label": "Cost Saved",
        "value": f"${abs(derived['cost_saved_usd']):.2f}",
        "sub":   "Electricity cost delta (7-day period)",
        "delta": f'<span style="color:{C_GREEN if derived["cost_saved_usd"] > 0 else C_RED};">{"▼ saved" if derived["cost_saved_usd"] > 0 else "▲ extra"}</span>',
        "color": "purple",
    },
    {
        "label": "Avg Confidence",
        "value": f"{derived['avg_confidence']:.0%}",
        "sub":   f"{derived['total_decisions']} decisions  ·  {derived['failed_decisions']} corrected  ·  {derived['recovered']} recovered",
        "delta": "",
        "color": "yellow",
    },
]

kpi_html = '<div class="kpi-grid">'
for k in kpis:
    kpi_html += f"""
    <div class="kpi-card {k['color']}">
      <div class="kpi-label">{k['label']}</div>
      <div class="kpi-value" style="color:{C_TEXT};">{k['value']}</div>
      <div class="kpi-sub">{k['sub']}</div>
      <div class="kpi-delta">{k['delta']}</div>
    </div>"""
kpi_html += "</div>"
st.markdown(kpi_html, unsafe_allow_html=True)

# ── SECTION 1: Performance Overview (bar charts) ──────────────────────────────
st.markdown('<div class="section-title">⚡ Performance Overview — Baseline vs AI</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Baseline", "AI Agent"],
        y=[b_kwh, ai_kwh],
        marker_color=[C_BASELINE, C_AI],
        text=[f"{b_kwh:.1f} kWh", f"{ai_kwh:.1f} kWh"],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=13, family="Inter"),
        width=0.45,
    ))
    apply_base_layout(fig, title=dict(text="Total Energy Consumption", font=dict(size=13, color=C_SUBTEXT)),
                      yaxis_title="kWh", height=280)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col2:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Baseline", "AI Agent"],
        y=[b_peak / 1000, ai_peak / 1000],
        marker_color=[C_BASELINE, C_AI],
        text=[f"{b_peak/1000:.2f} kW", f"{ai_peak/1000:.2f} kW"],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=13),
        width=0.45,
    ))
    apply_base_layout(fig, title=dict(text="Peak Demand", font=dict(size=13, color=C_SUBTEXT)),
                      yaxis_title="kW", height=280)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col3:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Baseline", "AI Agent"],
        y=[b_cd, ai_cd],
        marker_color=[C_BASELINE, C_AI],
        text=[f"{b_cd:.3f}°C", f"{ai_cd:.3f}°C"],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=13),
        width=0.45,
    ))
    apply_base_layout(fig, title=dict(text="Comfort Deviation (avg °C from setpoint)", font=dict(size=13, color=C_SUBTEXT)),
                      yaxis_title="°C", height=280)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── SECTION 2: Temperature Timeline ──────────────────────────────────────────
if not ai_df.empty and not base_df.empty and "dt" in ai_df.columns:
    st.markdown('<div class="section-title">🌡️ Indoor Temperature: Baseline vs AI Agent</div>', unsafe_allow_html=True)

    fig = go.Figure()
    # Outdoor temp (shared)
    if "outdoor_temp" in ai_df.columns:
        fig.add_trace(go.Scatter(
            x=ai_df["dt"], y=ai_df["outdoor_temp"],
            name="Outdoor Temp", line=dict(color="#F39C12", width=1.2, dash="dot"),
            opacity=0.7,
        ))
    # Setpoints
    if "cooling_sp" in ai_df.columns:
        fig.add_trace(go.Scatter(
            x=ai_df["dt"], y=ai_df["cooling_sp"],
            name="Cooling Setpoint", line=dict(color="#E74C3C", width=1, dash="dash"),
            opacity=0.5,
        ))
    if "heating_sp" in ai_df.columns:
        fig.add_trace(go.Scatter(
            x=ai_df["dt"], y=ai_df["heating_sp"],
            name="Heating Setpoint", line=dict(color="#3DD6F5", width=1, dash="dash"),
            opacity=0.5,
        ))
    # Baseline zone temp
    if "zone_temp" in base_df.columns:
        fig.add_trace(go.Scatter(
            x=base_df["dt"], y=base_df["zone_temp"],
            name="Baseline Zone Temp", line=dict(color=C_BASELINE, width=2),
        ))
    # AI zone temp
    if "zone_temp" in ai_df.columns:
        fig.add_trace(go.Scatter(
            x=ai_df["dt"], y=ai_df["zone_temp"],
            name="AI Zone Temp", line=dict(color=C_AI, width=2),
        ))
    apply_base_layout(fig,
        title=dict(text="Zone Temperature vs Setpoints vs Outdoor", font=dict(size=13, color=C_SUBTEXT)),
        yaxis_title="Temperature (°C)", height=320,
        xaxis=dict(tickformat="%b %d", dtick=86400000, gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── SECTION 3: Comfort Deviation Timeline ────────────────────────────────────
if not ai_df.empty and "dt" in ai_df.columns:
    st.markdown('<div class="section-title">😊 Comfort Deviation Over Time</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure()
        if "comfort_deviation" in base_df.columns and "dt" in base_df.columns:
            fig.add_trace(go.Scatter(
                x=base_df["dt"], y=base_df["comfort_deviation"],
                name="Baseline Comfort Dev.", fill="tozeroy",
                fillcolor="rgba(245,121,58,0.12)",
                line=dict(color=C_BASELINE, width=2),
            ))
        if "comfort_deviation" in ai_df.columns:
            fig.add_trace(go.Scatter(
                x=ai_df["dt"], y=ai_df["comfort_deviation"],
                name="AI Comfort Dev.", fill="tozeroy",
                fillcolor="rgba(61,214,245,0.10)",
                line=dict(color=C_AI, width=2),
            ))
        # Comfort violation threshold line
        fig.add_hline(y=0.8, line_dash="dot", line_color="rgba(231,76,60,0.5)",
                      annotation_text="Violation threshold (0.8°C)",
                      annotation_font=dict(color="rgba(231,76,60,0.8)", size=10))
        apply_base_layout(fig,
            title=dict(text="Comfort Deviation from Setpoint (°C)", font=dict(size=13, color=C_SUBTEXT)),
            yaxis_title="°C deviation", height=280,
            xaxis=dict(tickformat="%b %d", dtick=86400000, gridcolor="rgba(255,255,255,0.05)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col2:
        # Action distribution donut
        ad = derived["action_dist"]
        if ad:
            labels = list(ad.keys())
            values = list(ad.values())
            colors = [ACTION_COLORS.get(a, "#888") for a in labels]
            fig2 = go.Figure(go.Pie(
                labels=labels, values=values,
                marker_colors=colors,
                hole=0.60, textinfo="label+percent",
                textfont=dict(size=10, color=C_TEXT),
            ))
            fig2.add_annotation(
                text=f"<b>{sum(values)}</b><br>decisions",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=12, color=C_TEXT),
            )
            apply_base_layout(fig2,
                title=dict(text="Action Distribution", font=dict(size=13, color=C_SUBTEXT)),
                height=280, showlegend=False,
                margin=dict(l=4, r=4, t=32, b=4),
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── SECTION 4: Decision Timeline + Confidence ─────────────────────────────────
if not ai_df.empty and "dt" in ai_df.columns and "action" in ai_df.columns:
    st.markdown('<div class="section-title">🧠 AI Decision Timeline + Confidence Score</div>', unsafe_allow_html=True)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.04,
    )

    # Action color lane (scatter with big markers)
    for action, color in ACTION_COLORS.items():
        mask = ai_df["action"] == action
        if mask.sum() == 0:
            continue
        sub = ai_df[mask]
        fig.add_trace(go.Scatter(
            x=sub["dt"], y=[action] * len(sub),
            mode="markers",
            marker=dict(color=color, size=9, symbol="square"),
            name=action,
            showlegend=True,
        ), row=1, col=1)

    # Confidence line
    if "confidence" in ai_df.columns:
        conf_clean = ai_df.dropna(subset=["confidence", "dt"])
        fig.add_trace(go.Scatter(
            x=conf_clean["dt"], y=conf_clean["confidence"],
            name="Confidence", line=dict(color="#F39C12", width=2),
            fill="tozeroy", fillcolor="rgba(243,156,18,0.08)",
        ), row=2, col=1)
        # Confidence floor reference
        fig.add_hline(y=0.35, line_dash="dot", line_color="rgba(231,76,60,0.4)",
                      row=2, col=1,
                      annotation_text="Confidence floor",
                      annotation_font=dict(color="rgba(231,76,60,0.7)", size=9))

    apply_base_layout(fig,
        title=dict(text="Decision Timeline — Action Selection per Hour",
                   font=dict(size=13, color=C_SUBTEXT)),
        height=380,
        xaxis2=dict(tickformat="%b %d", dtick=86400000, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(categoryorder="array", categoryarray=["off","eco","normal","boost"],
                   gridcolor="rgba(255,255,255,0.05)"),
        yaxis2=dict(title="Confidence", range=[0, 1], gridcolor="rgba(255,255,255,0.05)"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── SECTION 5: Confidence Breakdown Radar ────────────────────────────────────
conf_cols = ["conf_historical", "conf_sensor", "conf_weather", "conf_comfort", "conf_stability"]
if not ai_df.empty and all(c in ai_df.columns for c in conf_cols):
    non_null = ai_df.dropna(subset=conf_cols)
    if len(non_null) > 0:
        st.markdown('<div class="section-title">📊 Confidence Signal Breakdown (Avg over Run)</div>', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])

        avgs = {
            "Historical Success":   non_null["conf_historical"].mean(),
            "Sensor Consistency":   non_null["conf_sensor"].mean(),
            "Weather Certainty":    non_null["conf_weather"].mean(),
            "Comfort Prediction":   non_null["conf_comfort"].mean(),
            "Simulation Stability": non_null["conf_stability"].mean(),
        }

        with col1:
            labels = list(avgs.keys())
            vals   = list(avgs.values())
            radar_vals = vals + [vals[0]]   # close polygon
            radar_labels = labels + [labels[0]]

            fig = go.Figure(go.Scatterpolar(
                r=radar_vals, theta=radar_labels,
                fill="toself", fillcolor="rgba(61,214,245,0.15)",
                line=dict(color=C_AI, width=2),
                name="Avg Confidence Signals",
            ))
            apply_base_layout(fig,
                title=dict(text="Confidence Signal Radar", font=dict(size=13, color=C_SUBTEXT)),
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1],
                                    gridcolor="rgba(255,255,255,0.1)",
                                    tickfont=dict(size=9, color=C_SUBTEXT)),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                                     tickfont=dict(size=10, color=C_TEXT)),
                    bgcolor="rgba(0,0,0,0)",
                ),
                height=300, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        with col2:
            # Stacked bar over time
            sample = non_null[["dt"] + conf_cols].dropna().copy()
            sample = sample.sort_values("dt")

            fig = go.Figure()
            names = {
                "conf_historical": "Historical Success",
                "conf_sensor":     "Sensor Consistency",
                "conf_weather":    "Weather Certainty",
                "conf_comfort":    "Comfort Prediction",
                "conf_stability":  "Sim. Stability",
            }
            colors_conf = ["#2ECC71","#3DD6F5","#F39C12","#9B59B6","#F5793A"]
            for col_name, color in zip(conf_cols, colors_conf):
                fig.add_trace(go.Bar(
                    x=sample["dt"], y=sample[col_name],
                    name=names[col_name],
                    marker_color=color,
                ))
            apply_base_layout(fig,
                title=dict(text="Confidence Signal Components per Cycle",
                           font=dict(size=13, color=C_SUBTEXT)),
                barmode="stack", yaxis_title="Signal Score",
                height=300,
                xaxis=dict(tickformat="%b %d %H:%M", gridcolor="rgba(255,255,255,0.05)"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── SECTION 6: Carbon & Cost ─────────────────────────────────────────────────
st.markdown('<div class="section-title">🌿 Carbon & Cost Impact</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    carbon_ai   = ai_kwh  * CARBON_KG_PER_KWH
    carbon_base = b_kwh   * CARBON_KG_PER_KWH
    fig = go.Figure(go.Bar(
        x=["Baseline", "AI Agent"],
        y=[carbon_base, carbon_ai],
        marker_color=[C_BASELINE, C_AI],
        text=[f"{carbon_base:.2f} kg CO₂", f"{carbon_ai:.2f} kg CO₂"],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=12),
        width=0.4,
    ))
    if carbon_base > carbon_ai:
        fig.add_annotation(
            x=1, y=carbon_ai,
            text=f"▼ {(carbon_base - carbon_ai):.2f} kg saved",
            showarrow=False, yshift=-20,
            font=dict(color=C_GREEN, size=11, family="Inter"),
        )
    apply_base_layout(fig, title=dict(text="Carbon Emissions (kg CO₂, 7-day)",
                                       font=dict(size=13, color=C_SUBTEXT)),
                      yaxis_title="kg CO₂", height=260)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col2:
    cost_ai   = ai_kwh  * ELECTRICITY_USD_KWH
    cost_base = b_kwh   * ELECTRICITY_USD_KWH
    fig = go.Figure(go.Bar(
        x=["Baseline", "AI Agent"],
        y=[cost_base, cost_ai],
        marker_color=[C_BASELINE, C_AI],
        text=[f"${cost_base:.2f}", f"${cost_ai:.2f}"],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=12),
        width=0.4,
    ))
    if cost_base > cost_ai:
        fig.add_annotation(
            x=1, y=cost_ai,
            text=f"▼ ${cost_base - cost_ai:.2f} saved",
            showarrow=False, yshift=-20,
            font=dict(color=C_GREEN, size=11),
        )
    apply_base_layout(fig, title=dict(text="Electricity Cost (USD, 7-day @ $0.12/kWh)",
                                       font=dict(size=13, color=C_SUBTEXT)),
                      yaxis_title="USD", height=260)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ── SECTION 7: Agent Reasoning Log ───────────────────────────────────────────
if not ai_df.empty:
    st.markdown('<div class="section-title">📋 Agent Reasoning Log</div>', unsafe_allow_html=True)

    tabs = st.tabs(["All Decisions", "Failed / Corrected", "High Confidence", "Low Confidence"])

    def _action_badge(action):
        cls = f"badge-{action}" if action in ["off","eco","normal","boost"] else "badge-off"
        return f'<span class="badge {cls}">{action}</span>'

    def _outcome_badge(outcome):
        if str(outcome) == "SUCCESS":
            return f'<span class="badge badge-ok">SUCCESS</span>'
        elif str(outcome) == "CORRECTED":
            return f'<span class="badge badge-corr">CORRECTED</span>'
        elif str(outcome) == "FALLBACK":
            return f'<span class="badge badge-fail">FALLBACK</span>'
        else:
            return f'<span class="badge badge-fail">{outcome}</span>'

    def _conf_html(conf):
        try:
            v = float(conf)
            pct = int(v * 100)
            color = C_GREEN if v >= 0.7 else (C_YELLOW if v >= 0.5 else C_RED)
            bar = f'<div class="conf-bar-bg"><div class="conf-bar-fg" style="width:{pct}%;background:{color};"></div></div>'
            return f'<span style="color:{color};font-weight:600;">{v:.2f}</span>{bar}'
        except Exception:
            return "—"

    def _render_table(df_sub, max_rows=50):
        cols = [c for c in ["timestamp","zone_temp","outdoor_temp","action",
                             "coil_speed","confidence","comfort_deviation","outcome","reasoning"]
                if c in df_sub.columns]
        rows_html = ""
        for _, row in df_sub.head(max_rows).iterrows():
            ts     = str(row.get("timestamp",""))
            zt     = f"{row['zone_temp']:.2f}°C" if pd.notna(row.get("zone_temp")) else "—"
            oat    = f"{row['outdoor_temp']:.1f}°C" if pd.notna(row.get("outdoor_temp")) else "—"
            action = _action_badge(str(row.get("action","")))
            spd    = f"{row['coil_speed']:.1f}" if pd.notna(row.get("coil_speed")) else "—"
            conf   = _conf_html(row.get("confidence"))
            dev    = f"{row['comfort_deviation']:.3f}°C" if pd.notna(row.get("comfort_deviation")) else "—"
            out    = _outcome_badge(str(row.get("outcome","")))
            rsn    = str(row.get("reasoning",""))[:100]
            rows_html += f"""
            <tr>
              <td style="color:{C_SUBTEXT};font-size:0.78rem;">{ts}</td>
              <td>{zt}</td>
              <td style="color:{C_SUBTEXT};">{oat}</td>
              <td>{action}</td>
              <td style="color:{C_SUBTEXT};">{spd}</td>
              <td>{conf}</td>
              <td>{dev}</td>
              <td>{out}</td>
              <td style="color:{C_SUBTEXT};font-size:0.78rem;max-width:280px;">{rsn}</td>
            </tr>"""
        header = """<tr>
          <th>Timestamp</th><th>Zone</th><th>Outdoor</th>
          <th>Action</th><th>Speed</th><th>Confidence</th>
          <th>Comfort Dev.</th><th>Outcome</th><th>Reasoning</th>
        </tr>"""
        return f'<div style="overflow-x:auto;"><table class="reasoning-table">{header}{rows_html}</table></div>'

    with tabs[0]:
        st.markdown(_render_table(ai_df.sort_values("dt") if "dt" in ai_df.columns else ai_df),
                    unsafe_allow_html=True)

    with tabs[1]:
        fail_mask = pd.Series([False] * len(ai_df), index=ai_df.index)
        if "outcome" in ai_df.columns:
            fail_mask = ai_df["outcome"].isin(["CORRECTED","FALLBACK","ROLLED_BACK"])
        failed = ai_df[fail_mask]
        if len(failed) == 0:
            st.markdown('<div class="alert-box green"><div class="alert-box-icon">✅</div><div class="alert-box-text"><div class="alert-box-title">No Failed Decisions</div>All decisions executed without correction.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(failed)} decisions** required correction or fallback:")
            st.markdown(_render_table(failed), unsafe_allow_html=True)

    with tabs[2]:
        if "confidence" in ai_df.columns:
            high = ai_df[ai_df["confidence"] >= 0.75]
            st.markdown(f"**{len(high)} high-confidence decisions** (≥0.75):")
            st.markdown(_render_table(high), unsafe_allow_html=True)
        else:
            st.info("No confidence data in log.")

    with tabs[3]:
        if "confidence" in ai_df.columns:
            low = ai_df[ai_df["confidence"] < 0.50]
            st.markdown(f"**{len(low)} low-confidence decisions** (<0.50):")
            if len(low) == 0:
                st.markdown('<div class="alert-box green"><div class="alert-box-icon">✅</div><div class="alert-box-text"><div class="alert-box-title">No Low-Confidence Decisions</div>System operated above the 0.50 confidence floor throughout the run.</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(_render_table(low), unsafe_allow_html=True)

# ── SECTION 8: Explainable AI (Decision Deep-Dive) ───────────────────────────
st.markdown('<div class="section-title">📖 Explainable AI (Decision Deep-Dive)</div>', unsafe_allow_html=True)
if not explanations_data:
    st.info("No detailed JSONL explanations recorded yet. When running with explanation_engine enabled, complete decision records with reasoning, trade-offs, and structured JSON will appear here.")
else:
    options = [f"Cycle #{e.get('cycle_number','?')} - {e.get('timestamp','?')} - {e.get('chosen_action','').upper()} ({e.get('outcome','')})" for e in explanations_data]
    sel_idx = st.selectbox("Select Decision Cycle to Inspect:", range(len(options)), format_func=lambda i: options[i])
    sel_exp = explanations_data[sel_idx]
    
    exp_tabs = st.tabs(["Human-Readable Explanation", "Structured JSON Record", "Candidate Trade-offs"])
    with exp_tabs[0]:
        hr_text = sel_exp.get("human_readable", "")
        if hr_text:
            st.code(hr_text, language="text")
        else:
            st.write(f"**Reasoning:** {sel_exp.get('reasoning_chain','N/A')}")
            st.write(f"**Primary Risk:** {sel_exp.get('primary_risk','N/A')}")
    with exp_tabs[1]:
        st.json(sel_exp)
    with exp_tabs[2]:
        cands = sel_exp.get("candidates", [])
        if cands:
            c_df = pd.DataFrame(cands)
            st.dataframe(c_df, use_container_width=True)
        else:
            st.write("No candidate trade-off details recorded for this cycle.")

# ── SECTION 9: System Health & Reliability Audit ─────────────────────────────
st.markdown('<div class="section-title">🛡️ System Health & Reliability Audit</div>', unsafe_allow_html=True)
if not health_data and not errors_data:
    st.info("No system health or reliability logs found. Run the resilience-enabled pipeline to generate live health tracking.")
else:
    h_col1, h_col2 = st.columns([1, 2])
    with h_col1:
        overall = health_data.get("overall_status", "unknown").upper()
        o_color = "green" if overall == "HEALTHY" else ("orange" if overall == "DEGRADED" else "red")
        o_icon  = "🟢" if overall == "HEALTHY" else ("🟠" if overall == "DEGRADED" else "🔴")
        st.markdown(f"""
        <div class="alert-box {o_color}">
          <div class="alert-box-icon">{o_icon}</div>
          <div class="alert-box-text">
            <div class="alert-box-title">System Status: {overall}</div>
            Session Start: {health_data.get('session_start', 'N/A')}<br>
            Last Updated: {health_data.get('last_updated', 'N/A')}<br>
            Frozen: {'YES ⚠️' if health_data.get('frozen', False) else 'NO ✅'}
          </div>
        </div>
        """, unsafe_allow_html=True)
    with h_col2:
        comps = health_data.get("components", {})
        if comps:
            c_list = []
            for k, v in comps.items():
                c_list.append({
                    "Component": k,
                    "Status": v.get("status", "unknown").upper(),
                    "Failures": v.get("failure_count", 0),
                    "Last Success": v.get("last_success_ts", "")[:19],
                    "Note": v.get("note", "")
                })
            st.dataframe(pd.DataFrame(c_list), use_container_width=True)
        else:
            st.write("No component health records available.")
    
    if errors_data:
        with st.expander(f"⚠️ Recent System Errors / Fallback Events ({len(errors_data)})", expanded=False):
            st.dataframe(pd.DataFrame(errors_data), use_container_width=True)

# ── SECTION 10: System Analysis Summary ──────────────────────────────────────
st.markdown('<div class="section-title">🔍 System Analysis Summary</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    energy_verdict = "green" if pct_e > 0 else "red"
    energy_icon    = "⬇️" if pct_e > 0 else "⬆️"
    st.markdown(f"""
    <div class="alert-box {energy_verdict}">
      <div class="alert-box-icon">{energy_icon}</div>
      <div class="alert-box-text">
        <div class="alert-box-title">Energy: {pct_e:+.2f}%</div>
        AI agent used {ai_kwh:.1f} kWh vs baseline {b_kwh:.1f} kWh over 7 days.
        {'Savings driven by graduated coil-speed selection — avoids over-cooling.' if pct_e > 0
         else 'Slightly higher energy due to more frequent comfort corrections.'}
      </div>
    </div>
    <div class="alert-box {'green' if pct_p >= 0 else 'red'}">
      <div class="alert-box-icon">{'⬇️' if pct_p >= 0 else '⬆️'}</div>
      <div class="alert-box-text">
        <div class="alert-box-title">Peak Demand: {pct_p:+.2f}%</div>
        AI peak: {ai_peak:.0f} W vs baseline: {b_peak:.0f} W.
        {'Peak demand successfully reduced.' if pct_p >= 0
         else 'Peak demand slightly elevated — agent occasionally co-runs with peak hours.'}
      </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    comfort_better = ai_cd < b_cd
    st.markdown(f"""
    <div class="alert-box {'green' if comfort_better else 'orange'}">
      <div class="alert-box-icon">{'😊' if comfort_better else '😐'}</div>
      <div class="alert-box-text">
        <div class="alert-box-title">Comfort: {'Better' if comfort_better else 'Wider'} deviation than baseline</div>
        AI: {ai_cd:.3f}°C avg deviation vs baseline: {b_cd:.3f}°C.
        {'Tighter comfort band — AI corrects deviations more precisely.' if comfort_better
         else 'Wider comfort deviation — trade-off for energy savings.'}
      </div>
    </div>
    <div class="alert-box cyan">
      <div class="alert-box-icon">🤖</div>
      <div class="alert-box-text">
        <div class="alert-box-title">Multi-Candidate Planner Active</div>
        Each hour: evaluate 4 candidates → select optimal → record rejection reasoning.
        Confidence Engine computes 5-signal prior (historical success, sensor consistency,
        weather certainty, comfort prediction, simulation stability).
      </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:32px 0 12px;color:#4A5568;font-size:0.80rem;">
  EcoLoop · Honeywell Hackathon 2024 · EnergyPlus + Ollama (qwen2.5:3b) · Phase 5
</div>
""", unsafe_allow_html=True)
