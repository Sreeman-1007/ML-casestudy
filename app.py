import os
import sys
import json
import time
import io
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath("."))
from src.predict import CongestionPredictor

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Internet Traffic Congestion Prediction | ML Analytics Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DESIGN SYSTEM: SOPHISTICATED DARK TECHNOLOGY / ACADEMIC ML AESTHETIC
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Root Variables & Typography */
    :root {
        --bg-main: #080d1a;
        --bg-surface: #0f172a;
        --bg-card: #131e36;
        --border-color: rgba(51, 65, 85, 0.6);
        --accent-blue: #38bdf8;
        --accent-primary: #2563eb;
        --color-low: #10b981;
        --color-med: #f59e0b;
        --color-high: #ef4444;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    /* Streamlit block container padding normalization */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1240px !important;
    }

    /* Subtle Technical Dot-Grid Pattern */
    .stApp {
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(37, 99, 235, 0.07) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(56, 189, 248, 0.05) 0%, transparent 45%),
            radial-gradient(rgba(51, 65, 85, 0.25) 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 28px 28px;
    }

    /* Sidebar Refinement */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1326 0%, #080d18 100%) !important;
        border-right: 1px solid rgba(51, 65, 85, 0.5) !important;
    }
    section[data-testid="stSidebar"] div.stMarkdown h3 {
        color: var(--accent-blue) !important;
        font-size: 11px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        font-weight: 700 !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
    }

    /* Sidebar Navigation Pills */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        gap: 5px;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background-color: rgba(19, 30, 54, 0.4) !important;
        border: 1px solid rgba(51, 65, 85, 0.4) !important;
        border-radius: 9px !important;
        padding: 9px 14px !important;
        margin-bottom: 3px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background-color: rgba(56, 189, 248, 0.1) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        transform: translateX(2px) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.28) 0%, rgba(14, 165, 233, 0.18) 100%) !important;
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2) !important;
    }

    /* Refined Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0e1a38 0%, #0a1329 65%, #070c1a 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 28px 34px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -4px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #2563eb 0%, #38bdf8 50%, #10b981 100%);
    }
    .hero-badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 12px;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 9999px;
        padding: 4px 11px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #7dd3fc;
        text-transform: uppercase;
    }
    .hero-heading {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 8px 0;
        letter-spacing: -0.6px;
        line-height: 1.25;
    }
    .hero-subtext {
        font-size: 14.5px;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
        line-height: 1.5;
        max-width: 880px;
    }

    /* Metric Cards */
    .metric-card-box {
        background: linear-gradient(145deg, #111d38 0%, #0d162d 100%);
        border: 1px solid rgba(51, 65, 85, 0.55);
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
    }
    .metric-card-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
    }
    .metric-border-blue::before { background: #38bdf8; }
    .metric-border-green::before { background: #10b981; }
    .metric-border-purple::before { background: #a78bfa; }
    .metric-border-amber::before { background: #f59e0b; }

    .metric-tag-label {
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.7px;
        margin-bottom: 6px;
    }
    .metric-number-display {
        font-size: 27px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-helper-caption {
        font-size: 12px;
        color: #64748b;
        font-weight: 500;
    }

    /* Section Cards */
    .dashboard-panel {
        background: linear-gradient(145deg, rgba(16, 26, 48, 0.6) 0%, rgba(11, 19, 36, 0.7) 100%);
        border: 1px solid rgba(51, 65, 85, 0.55);
        border-radius: 12px;
        padding: 22px 24px;
        margin-bottom: 18px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
    }
    .panel-header-title {
        font-size: 16.5px;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 12px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Taxonomy Cards */
    .taxonomy-item {
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease;
    }
    .taxonomy-item:hover {
        transform: translateX(3px);
    }
    .tax-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 78, 59, 0.2) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 4px solid #10b981;
    }
    .tax-med {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(120, 53, 15, 0.2) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 4px solid #f59e0b;
    }
    .tax-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(127, 29, 29, 0.2) 100%);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-left: 4px solid #ef4444;
    }

    /* Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-green { background-color: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #6ee7b7; }
    .badge-amber { background-color: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fde68a; }
    .badge-red { background-color: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; }
    .badge-cyan { background-color: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #7dd3fc; }

    /* Prediction Result Component */
    .prediction-display-card {
        border-radius: 14px;
        padding: 26px 30px;
        margin: 18px 0;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4);
    }
    .pred-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(9, 23, 44, 0.95) 100%);
        border: 2px solid #10b981;
    }
    .pred-card-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(9, 23, 44, 0.95) 100%);
        border: 2px solid #f59e0b;
    }
    .pred-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(9, 23, 44, 0.95) 100%);
        border: 2px solid #ef4444;
    }
    .pred-hero-label {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 6px 0 10px 0;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Pipeline Workflow Tracker */
    .workflow-tracker {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 20px;
        font-size: 12px;
        font-weight: 600;
        color: #94a3b8;
    }
    .step-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .step-idx {
        background: #1e293b;
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        width: 20px;
        height: 20px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10.5px;
        font-weight: 700;
    }

    /* Pulsing Status Radar */
    .radar-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        display: inline-block;
        box-shadow: 0 0 0 rgba(16, 185, 129, 0.7);
        animation: radar-pulse 2s infinite;
        margin-right: 6px;
    }
    @keyframes radar-pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid rgba(51, 65, 85, 0.7);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border-radius: 7px 7px 0 0 !important;
        color: #94a3b8 !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-bottom: none !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #172554 100%) !important;
        color: #38bdf8 !important;
        border-color: #38bdf8 !important;
    }

    /* Primary Buttons & Form Submit */
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #2563eb 0%, #0284c7 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 9px 22px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(56, 189, 248, 0.4) !important;
        border-color: #38bdf8 !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 7px !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }
    
    /* Tables and dataframes */
    .stDataFrame {
        border: 1px solid rgba(51, 65, 85, 0.5);
        border-radius: 9px;
        overflow: hidden;
    }

    /* LaTeX Math Container */
    .math-panel {
        background: rgba(11, 19, 36, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 9px;
        padding: 14px 18px;
        margin: 10px 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CACHED RESOURCE LOADERS (UNTOUCHED BACKEND / DATA LOGIC)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    """Initializes and caches the production CongestionPredictor instance."""
    return CongestionPredictor(
        model_path="./models/random_forest.joblib",
        feature_columns_path="./models/feature_columns.json",
        target_config_path="./models/congestion_target_parameters.json"
    )

@st.cache_data
def load_demonstration_presets():
    """Loads realistic pre-extracted demonstration rows from processed_test.csv."""
    presets_path = "./models/demonstration_presets.json"
    if os.path.exists(presets_path):
        with open(presets_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}

@st.cache_data
def load_evaluation_data():
    """Loads evaluation comparison table and structured metrics."""
    comp_path = "./results/model_comparison.csv"
    metrics_path = "./results/evaluation_metrics.json"
    
    comp_df = pd.read_csv(comp_path) if os.path.exists(comp_path) else None
    
    metrics_dict = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fp:
            metrics_dict = json.load(fp)
            
    return comp_df, metrics_dict

@st.cache_data
def load_feature_importance():
    """Loads the 58-feature importance ranking table."""
    fi_path = "./results/feature_importance.csv"
    if os.path.exists(fi_path):
        return pd.read_csv(fi_path)
    return None

# Load global predictor instance safely
try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    predictor = None
    model_loaded = False
    model_error = str(e)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION (PROFESSIONAL CONTROL CONSOLE)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 14px 0; border-bottom: 1px solid rgba(51, 65, 85, 0.5); margin-bottom: 16px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #2563eb, #38bdf8); width: 36px; height: 36px; border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 19px;">
                🌐
            </div>
            <div>
                <div style="font-weight: 800; font-size: 15px; color: #ffffff; letter-spacing: -0.2px;">NETFLOW AI</div>
                <div style="font-size: 10.5px; color: #38bdf8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px;">Traffic Congestion Lab</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Navigation")
    page = st.radio(
        "Select Dashboard View:",
        [
            "📊 Dashboard Overview",
            "🔮 Congestion Prediction",
            "📈 Model Performance",
            "🔍 Feature Importance",
            "ℹ️ About the Project"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### System Telemetry")
    
    if model_loaded:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 78, 59, 0.22) 100%); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 9px; padding: 13px; margin-bottom: 12px;">
            <div style="display: flex; align-items: center; margin-bottom: 7px;">
                <span class="radar-indicator"></span>
                <span style="color: #6ee7b7; font-weight: 700; font-size: 12.5px;">MODEL ONLINE</span>
            </div>
            <div style="font-size: 11.5px; color: #cbd5e1; margin-bottom: 2px;">• Architecture: <b>Random Forest</b></div>
            <div style="font-size: 11.5px; color: #cbd5e1; margin-bottom: 2px;">• Test Accuracy: <b style="color: #38bdf8;">85.39%</b></div>
            <div style="font-size: 11.5px; color: #cbd5e1; margin-bottom: 2px;">• Macro F1-Score: <b style="color: #a78bfa;">85.53%</b></div>
            <div style="font-size: 11.5px; color: #cbd5e1;">• Active Features: <b>58 Telemetry Inputs</b></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error(f"● Model Offline: {model_error}")
        
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.55); border: 1px solid rgba(51, 65, 85, 0.4); border-radius: 8px; padding: 11px; font-size: 11px; color: #94a3b8; line-height: 1.6;">
        <div><b>Project:</b> University ML Capstone</div>
        <div><b>Dataset:</b> CIC-IDS2017 Benchmark</div>
        <div><b>Memory Footprint:</b> 8 GB RAM Optimized</div>
        <div><b>Target Scope:</b> 3-Class TIS Proxy</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# -----------------------------------------------------------------------------
if page == "📊 Dashboard Overview":
    # Refined Hero Section
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge-container">
            <span class="hero-badge">🎓 Academic ML Capstone</span>
            <span class="hero-badge">🌐 CIC-IDS2017 Benchmark</span>
            <span class="hero-badge">🛡️ Zero-Leakage Quarantine Active</span>
        </div>
        <h1 class="hero-heading">Internet Traffic Congestion Prediction</h1>
        <p class="hero-subtext">
            Machine Learning-Based Network Traffic Analysis & Congestion Estimation using 
            58 physical packet morphology, buffering, and duty-cycle telemetry indicators.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prominent Metric Cards Row (Exact Historical Metrics Preserved)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card-box metric-border-blue">
            <div class="metric-tag-label">Selected Model</div>
            <div class="metric-number-display" style="color: #38bdf8;">Random Forest</div>
            <div class="metric-helper-caption">100 Trees Ensemble (Depth 16)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card-box metric-border-green">
            <div class="metric-tag-label">Test Set Accuracy</div>
            <div class="metric-number-display" style="color: #34d399;">85.39%</div>
            <div class="metric-helper-caption">Tested on 40,000 Unseen Flows</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card-box metric-border-purple">
            <div class="metric-tag-label">Macro F1-Score</div>
            <div class="metric-number-display" style="color: #a78bfa;">85.53%</div>
            <div class="metric-helper-caption">Balanced Across All 3 Tiers</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card-box metric-border-amber">
            <div class="metric-tag-label">Predictor Features</div>
            <div class="metric-number-display" style="color: #fbbf24;">58 Features</div>
            <div class="metric-helper-caption">Zero Target Leakage Quarantine</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two-Column Overview Layout
    c_left, c_right = st.columns([3, 2])
    
    with c_left:
        st.markdown("""
        <div class="dashboard-panel">
            <div class="panel-header-title">🎯 Project Purpose & Mathematical Formulation</div>
            <p style="color: #cbd5e1; font-size: 13.5px; line-height: 1.6; margin-bottom: 10px;">
                This system estimates network traffic congestion states from statistical flow telemetry 
                derived from the benchmark <b>CIC-IDS2017</b> dataset.
            </p>
            <p style="color: #cbd5e1; font-size: 13.5px; line-height: 1.6;">
                Because CIC-IDS2017 is primarily an intrusion dataset without native hardware queue drop counters, 
                the ground-truth target is formulated as a mathematically defensible <b>Traffic Intensity Score (TIS)</b> 
                combining normalized logarithmic bandwidth demand and packet processing load:
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="math-panel">
            <span style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px;">Traffic Intensity Score (TIS) Formulation</span>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"\text{TIS} = 0.5 \cdot \text{norm}(\log_{10}(1 + \text{Flow Bytes/s})) + 0.5 \cdot \text{norm}(\log_{10}(1 + \text{Flow Packets/s}))")
        
        st.markdown("""
        <div class="dashboard-panel" style="margin-top: 14px;">
            <div class="panel-header-title" style="color: #38bdf8;">🛡️ Strict Target Leakage Quarantine</div>
            <p style="color: #cbd5e1; font-size: 13px; line-height: 1.6;">
                To guarantee academic integrity and prevent the ML model from merely inverting an arithmetic formula:
            </p>
            <ul style="color: #cbd5e1; font-size: 12.5px; line-height: 1.7; padding-left: 18px; margin-bottom: 0;">
                <li><b style="color: #f87171;">Quarantined (Excluded):</b> <code>Flow Bytes/s</code>, <code>Flow Packets/s</code>, <code>Flow Duration</code>, and all 14 <code>Flow/Fwd/Bwd IAT</code> features.</li>
                <li><b style="color: #34d399;">Allowed Predictors (58 features):</b> Packet morphology distributions, TCP control flags, window buffer headroom, and active/idle duty cycles.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c_right:
        st.markdown("""
        <div class="dashboard-panel">
            <div class="panel-header-title">🚦 Congestion Level Taxonomy</div>
            <div class="taxonomy-item tax-low">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <b style="color: #34d399; font-size: 14px;">🟢 LOW CONGESTION</b>
                    <span class="badge-pill badge-green">TIS &le; 0.2894</span>
                </div>
                <div style="color: #cbd5e1; font-size: 12px; line-height: 1.5;">
                    Quiescent baseline traffic; prolonged idle pauses (median ~5.86s), high client window buffer headroom, and large MTU payloads.
                </div>
            </div>
            
            <div class="taxonomy-item tax-med">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <b style="color: #fbbf24; font-size: 14px;">🟡 MEDIUM CONGESTION</b>
                    <span class="badge-pill badge-amber">0.2894 &lt; TIS &le; 0.5255</span>
                </div>
                <div style="color: #cbd5e1; font-size: 12px; line-height: 1.5;">
                    Moderate transmission load; intermediate burst activity, emerging queuing strain, transitional buffer occupancy.
                </div>
            </div>
            
            <div class="taxonomy-item tax-high">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <b style="color: #f87171; font-size: 14px;">🔴 HIGH CONGESTION</b>
                    <span class="badge-pill badge-red">TIS &gt; 0.5255</span>
                </div>
                <div style="color: #cbd5e1; font-size: 12px; line-height: 1.5;">
                    Severe network saturation; continuous zero-idle transmission, small-packet flood dynamics (median ~16 bytes), reduced TCP window capacity.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Quick Navigation Guide
    st.markdown("""
    <div class="dashboard-panel">
        <div class="panel-header-title">🚀 How to Obtain Predictions</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
            <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(51, 65, 85, 0.4); border-radius: 9px; padding: 15px;">
                <b style="color: #38bdf8; font-size: 13.5px;">⚡ 1. Single-Flow Telemetry Prediction</b>
                <p style="color: #94a3b8; font-size: 12px; line-height: 1.5; margin: 5px 0 0 0;">
                    Navigate to <b>🔮 Congestion Prediction</b> &rarr; Option B. Load one-click authentic test presets or input custom telemetry to receive instantaneous classification and confidence estimates.
                </p>
            </div>
            <div style="background: rgba(15, 23, 42, 0.45); border: 1px solid rgba(51, 65, 85, 0.4); border-radius: 9px; padding: 15px;">
                <b style="color: #38bdf8; font-size: 13.5px;">📁 2. Batch Flow CSV Upload</b>
                <p style="color: #94a3b8; font-size: 12px; line-height: 1.5; margin: 5px 0 0 0;">
                    Upload any CSV containing the 58 standard predictor columns (e.g. <code>dataset/processed_test.csv</code>). The system executes batch evaluation and exports an annotated CSV.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: CONGESTION PREDICTION
# -----------------------------------------------------------------------------
elif page == "🔮 Congestion Prediction":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge-container">
            <span class="hero-badge">⚡ Real-Time Inference</span>
            <span class="hero-badge">🤖 Random Forest (100 Trees)</span>
        </div>
        <h1 class="hero-heading">Network Congestion Inference Engine</h1>
        <p class="hero-subtext">
            Evaluate individual network telemetry flow vectors or process batch CSV datasets 
            using the trained production Random Forest classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Workflow Pipeline Tracker
    st.markdown("""
    <div class="workflow-tracker">
        <div class="step-tag"><span class="step-idx">1</span> Ingest Telemetry</div>
        <div style="color: #475569;">&rarr;</div>
        <div class="step-tag"><span class="step-idx">2</span> Quarantine Schema Check</div>
        <div style="color: #475569;">&rarr;</div>
        <div class="step-tag"><span class="step-idx">3</span> Random Forest Voting</div>
        <div style="color: #475569;">&rarr;</div>
        <div class="step-tag"><span class="step-idx">4</span> Probabilistic Classification</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not model_loaded:
        st.error(f"Prediction model could not be loaded: {model_error}")
        st.stop()
        
    mode = st.radio(
        "Select Prediction Mode:",
        ["Option A: Batch CSV Upload", "Option B: Single-Flow Inference"],
        horizontal=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------
    # OPTION A: BATCH CSV UPLOAD
    # -------------------------------------------------------------------------
    if mode == "Option A: Batch CSV Upload":
        st.markdown("""
        <div class="dashboard-panel">
            <div class="panel-header-title">📁 Batch Telemetry Ingestion</div>
            <p style="color: #94a3b8; font-size: 13px; line-height: 1.5; margin-bottom: 8px;">
                Upload a CSV file containing network flow records. The file must supply the <b>58 required predictor columns</b>.
            </p>
            <div style="background: rgba(30, 41, 59, 0.4); border-left: 3px solid #38bdf8; padding: 7px 12px; border-radius: 4px; font-size: 11.5px; color: #cbd5e1;">
                💡 <b>Quick Demonstration Tip:</b> You can test this feature using the included test file <code>dataset/processed_test.csv</code>!
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Network Flow CSV", type=["csv"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file, low_memory=False)
                st.success(f"Successfully loaded CSV with **{len(df_upload):,} rows** and **{len(df_upload.columns)} columns**.")
                
                # Check for target column in input and drop safely
                if "Congestion_Level" in df_upload.columns:
                    st.warning("⚠️ Column 'Congestion_Level' detected in uploaded file. It will be quarantined and ignored during inference.")
                    df_upload = df_upload.drop(columns=["Congestion_Level"])
                    
                st.markdown("##### 🔍 Uploaded Telemetry Preview (First 5 Rows):")
                st.dataframe(df_upload.head(5), height=180, width="stretch")
                
                if st.button("🚀 Run Batch Congestion Prediction", type="primary"):
                    with st.spinner(f"Evaluating {len(df_upload):,} flow records through 100 Random Forest estimators..."):
                        t_start = time.perf_counter()
                        batch_results = predictor.predict(df_upload, return_probabilities=True)
                        t_elapsed = time.perf_counter() - t_start
                        
                        preds = [r["predicted_congestion_level"] for r in batch_results]
                        confs = [r["confidence"] for r in batch_results]
                        
                        df_result = df_upload.copy()
                        df_result.insert(0, "Predicted_Congestion_Level", preds)
                        df_result.insert(1, "Prediction_Confidence", confs)
                        
                        st.success(f"Batch prediction completed in **{t_elapsed:.3f} seconds** ({t_elapsed/len(df_upload)*1e6:.1f} \u03bcs/sample)!")
                        
                        # Summary Metrics Grid
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Flows Evaluated", f"{len(df_result):,}")
                        c2.metric("LOW Congestion Flows", f"{(df_result['Predicted_Congestion_Level']=='LOW').sum():,}")
                        c3.metric("MEDIUM Congestion Flows", f"{(df_result['Predicted_Congestion_Level']=='MEDIUM').sum():,}")
                        c4.metric("HIGH Congestion Flows", f"{(df_result['Predicted_Congestion_Level']=='HIGH').sum():,}")
                        
                        st.markdown("##### 📋 Annotated Predictions Preview:")
                        st.dataframe(df_result.head(10), height=250, width="stretch")
                        
                        csv_buffer = io.StringIO()
                        df_result.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="📥 Download Annotated Predictions CSV",
                            data=csv_buffer.getvalue(),
                            file_name="congestion_predictions_output.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Batch Processing Error: {str(e)}")
                
    # -------------------------------------------------------------------------
    # OPTION B: SINGLE-RECORD INFERENCE
    # -------------------------------------------------------------------------
    else:
        st.markdown("""
        <div class="dashboard-panel">
            <div class="panel-header-title">⚡ Single-Flow Telemetry Configuration</div>
            <p style="color: #94a3b8; font-size: 13px; line-height: 1.5; margin: 0;">
                Select an authentic demonstration profile extracted from the unseen test partition, or enter custom parameters below:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        presets = load_demonstration_presets()
        preset_names = ["Custom / Manual Input"] + list(presets.keys())
        
        selected_preset = st.selectbox(
            "Load Demonstration Telemetry Profile:",
            preset_names,
            index=1 if len(preset_names) > 1 else 0
        )
        
        base_vals = presets.get(selected_preset, {}) if selected_preset != "Custom / Manual Input" else {}
        
        if selected_preset != "Custom / Manual Input":
            st.info(f"Loaded telemetry profile: **{selected_preset}** (Authentic holdout test vector)")
            
        with st.form("single_prediction_form"):
            input_dict = {}
            
            # Category 1: Network Port
            with st.expander("1. Network Transport Service Port", expanded=True):
                col_p, _ = st.columns([1, 2])
                with col_p:
                    input_dict["Destination Port"] = st.number_input(
                        "Destination Port",
                        min_value=0, max_value=65535,
                        value=int(base_vals.get("Destination Port", 80)),
                        help="Target server port (80=HTTP, 443=HTTPS, 53=DNS, etc.)"
                    )
                    
            # Category 2: Packet Counts & Volume
            with st.expander("2. Packet Counts & Volumetric Metrics", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    input_dict["Total Fwd Packets"] = st.number_input(
                        "Total Fwd Packets", min_value=0, value=int(base_vals.get("Total Fwd Packets", 2))
                    )
                    input_dict["Subflow Fwd Packets"] = st.number_input(
                        "Subflow Fwd Packets", min_value=0, value=int(base_vals.get("Subflow Fwd Packets", 2))
                    )
                with c2:
                    input_dict["Total Backward Packets"] = st.number_input(
                        "Total Backward Packets", min_value=0, value=int(base_vals.get("Total Backward Packets", 1))
                    )
                    input_dict["Subflow Bwd Packets"] = st.number_input(
                        "Subflow Bwd Packets", min_value=0, value=int(base_vals.get("Subflow Bwd Packets", 1))
                    )
                with c3:
                    input_dict["Total Length of Fwd Packets"] = st.number_input(
                        "Total Length of Fwd Packets", min_value=0.0, value=float(base_vals.get("Total Length of Fwd Packets", 26.0))
                    )
                    input_dict["Subflow Fwd Bytes"] = st.number_input(
                        "Subflow Fwd Bytes", min_value=0.0, value=float(base_vals.get("Subflow Fwd Bytes", 26.0))
                    )
                with c4:
                    input_dict["Total Length of Bwd Packets"] = st.number_input(
                        "Total Length of Bwd Packets", min_value=0.0, value=float(base_vals.get("Total Length of Bwd Packets", 26.0))
                    )
                    input_dict["Subflow Bwd Bytes"] = st.number_input(
                        "Subflow Bwd Bytes", min_value=0.0, value=float(base_vals.get("Subflow Bwd Bytes", 26.0))
                    )
                input_dict["act_data_pkt_fwd"] = st.number_input(
                    "act_data_pkt_fwd (Actual Data Packets Forward)", min_value=0, value=int(base_vals.get("act_data_pkt_fwd", 1))
                )

            # Category 3: Packet Morphology & Length Statistics
            with st.expander("3. Packet Morphology & Length Statistics", expanded=False):
                m1, m2, m3 = st.columns(3)
                with m1:
                    input_dict["Average Packet Size"] = st.number_input(
                        "Average Packet Size", min_value=0.0, value=float(base_vals.get("Average Packet Size", 116.2))
                    )
                    input_dict["Packet Length Mean"] = st.number_input(
                        "Packet Length Mean", min_value=0.0, value=float(base_vals.get("Packet Length Mean", 87.4))
                    )
                    input_dict["Min Packet Length"] = st.number_input(
                        "Min Packet Length", min_value=0.0, value=float(base_vals.get("Min Packet Length", 0.0))
                    )
                    input_dict["Max Packet Length"] = st.number_input(
                        "Max Packet Length", min_value=0.0, value=float(base_vals.get("Max Packet Length", 232.0))
                    )
                    input_dict["Packet Length Std"] = st.number_input(
                        "Packet Length Std", min_value=0.0, value=float(base_vals.get("Packet Length Std", 150.8))
                    )
                    input_dict["Packet Length Variance"] = st.number_input(
                        "Packet Length Variance", min_value=0.0, value=float(base_vals.get("Packet Length Variance", 22740.0))
                    )
                with m2:
                    input_dict["Fwd Packet Length Max"] = st.number_input(
                        "Fwd Packet Length Max", min_value=0.0, value=float(base_vals.get("Fwd Packet Length Max", 26.0))
                    )
                    input_dict["Fwd Packet Length Min"] = st.number_input(
                        "Fwd Packet Length Min", min_value=0.0, value=float(base_vals.get("Fwd Packet Length Min", 0.0))
                    )
                    input_dict["Fwd Packet Length Mean"] = st.number_input(
                        "Fwd Packet Length Mean", min_value=0.0, value=float(base_vals.get("Fwd Packet Length Mean", 13.0))
                    )
                    input_dict["Fwd Packet Length Std"] = st.number_input(
                        "Fwd Packet Length Std", min_value=0.0, value=float(base_vals.get("Fwd Packet Length Std", 18.3))
                    )
                    input_dict["Avg Fwd Segment Size"] = st.number_input(
                        "Avg Fwd Segment Size", min_value=0.0, value=float(base_vals.get("Avg Fwd Segment Size", 13.0))
                    )
                with m3:
                    input_dict["Bwd Packet Length Max"] = st.number_input(
                        "Bwd Packet Length Max", min_value=0.0, value=float(base_vals.get("Bwd Packet Length Max", 26.0))
                    )
                    input_dict["Bwd Packet Length Min"] = st.number_input(
                        "Bwd Packet Length Min", min_value=0.0, value=float(base_vals.get("Bwd Packet Length Min", 0.0))
                    )
                    input_dict["Bwd Packet Length Mean"] = st.number_input(
                        "Bwd Packet Length Mean", min_value=0.0, value=float(base_vals.get("Bwd Packet Length Mean", 26.0))
                    )
                    input_dict["Bwd Packet Length Std"] = st.number_input(
                        "Bwd Packet Length Std", min_value=0.0, value=float(base_vals.get("Bwd Packet Length Std", 0.0))
                    )
                    input_dict["Avg Bwd Segment Size"] = st.number_input(
                        "Avg Bwd Segment Size", min_value=0.0, value=float(base_vals.get("Avg Bwd Segment Size", 26.0))
                    )

            # Category 4: TCP Control Flags
            with st.expander("4. TCP Control Flags", expanded=False):
                f1, f2, f3, f4 = st.columns(4)
                with f1:
                    input_dict["FIN Flag Count"] = st.number_input("FIN Flag Count", min_value=0, max_value=1, value=int(base_vals.get("FIN Flag Count", 0)))
                    input_dict["SYN Flag Count"] = st.number_input("SYN Flag Count", min_value=0, max_value=1, value=int(base_vals.get("SYN Flag Count", 0)))
                    input_dict["RST Flag Count"] = st.number_input("RST Flag Count", min_value=0, max_value=1, value=int(base_vals.get("RST Flag Count", 0)))
                with f2:
                    input_dict["PSH Flag Count"] = st.number_input("PSH Flag Count", min_value=0, max_value=1, value=int(base_vals.get("PSH Flag Count", 0)))
                    input_dict["ACK Flag Count"] = st.number_input("ACK Flag Count", min_value=0, max_value=1, value=int(base_vals.get("ACK Flag Count", 1)))
                    input_dict["URG Flag Count"] = st.number_input("URG Flag Count", min_value=0, max_value=1, value=int(base_vals.get("URG Flag Count", 0)))
                with f3:
                    input_dict["CWE Flag Count"] = st.number_input("CWE Flag Count", min_value=0, max_value=1, value=int(base_vals.get("CWE Flag Count", 0)))
                    input_dict["ECE Flag Count"] = st.number_input("ECE Flag Count", min_value=0, max_value=1, value=int(base_vals.get("ECE Flag Count", 0)))
                    input_dict["Fwd PSH Flags"] = st.number_input("Fwd PSH Flags", min_value=0, max_value=1, value=int(base_vals.get("Fwd PSH Flags", 0)))
                with f4:
                    input_dict["Bwd PSH Flags"] = st.number_input("Bwd PSH Flags", min_value=0, max_value=1, value=int(base_vals.get("Bwd PSH Flags", 0)))
                    input_dict["Fwd URG Flags"] = st.number_input("Fwd URG Flags", min_value=0, max_value=1, value=int(base_vals.get("Fwd URG Flags", 0)))
                    input_dict["Bwd URG Flags"] = st.number_input("Bwd URG Flags", min_value=0, max_value=1, value=int(base_vals.get("Bwd URG Flags", 0)))

            # Category 5: TCP Window & Buffering Dynamics
            with st.expander("5. TCP Window & Buffering Dynamics", expanded=False):
                w1, w2 = st.columns(2)
                with w1:
                    input_dict["Init_Win_bytes_forward"] = st.number_input(
                        "Init_Win_bytes_forward", value=int(base_vals.get("Init_Win_bytes_forward", 406))
                    )
                    input_dict["min_seg_size_forward"] = st.number_input(
                        "min_seg_size_forward", value=int(base_vals.get("min_seg_size_forward", 20))
                    )
                with w2:
                    input_dict["Init_Win_bytes_backward"] = st.number_input(
                        "Init_Win_bytes_backward", value=int(base_vals.get("Init_Win_bytes_backward", 136))
                    )
                    input_dict["Down/Up Ratio"] = st.number_input(
                        "Down/Up Ratio", min_value=0, value=int(base_vals.get("Down/Up Ratio", 1))
                    )

            # Category 6: Header & Bulk Settings
            with st.expander("6. Header & Bulk Metrics", expanded=False):
                h1, h2 = st.columns(2)
                with h1:
                    input_dict["Fwd Header Length"] = st.number_input("Fwd Header Length", value=int(base_vals.get("Fwd Header Length", 40)))
                    input_dict["Fwd Avg Bytes/Bulk"] = st.number_input("Fwd Avg Bytes/Bulk", value=int(base_vals.get("Fwd Avg Bytes/Bulk", 0)))
                    input_dict["Fwd Avg Packets/Bulk"] = st.number_input("Fwd Avg Packets/Bulk", value=int(base_vals.get("Fwd Avg Packets/Bulk", 0)))
                    input_dict["Fwd Avg Bulk Rate"] = st.number_input("Fwd Avg Bulk Rate", value=int(base_vals.get("Fwd Avg Bulk Rate", 0)))
                with h2:
                    input_dict["Bwd Header Length"] = st.number_input("Bwd Header Length", value=int(base_vals.get("Bwd Header Length", 20)))
                    input_dict["Bwd Avg Bytes/Bulk"] = st.number_input("Bwd Avg Bytes/Bulk", value=int(base_vals.get("Bwd Avg Bytes/Bulk", 0)))
                    input_dict["Bwd Avg Packets/Bulk"] = st.number_input("Bwd Avg Packets/Bulk", value=int(base_vals.get("Bwd Avg Packets/Bulk", 0)))
                    input_dict["Bwd Avg Bulk Rate"] = st.number_input("Bwd Avg Bulk Rate", value=int(base_vals.get("Bwd Avg Bulk Rate", 0)))

            # Category 7: Duty Cycle & Burst Timing
            with st.expander("7. Active / Idle Duty Cycle & Silence Timing", expanded=False):
                d1, d2 = st.columns(2)
                with d1:
                    input_dict["Active Mean"] = st.number_input("Active Mean (\u03bcs)", min_value=0.0, value=float(base_vals.get("Active Mean", 0.0)))
                    input_dict["Active Std"] = st.number_input("Active Std (\u03bcs)", min_value=0.0, value=float(base_vals.get("Active Std", 0.0)))
                    input_dict["Active Max"] = st.number_input("Active Max (\u03bcs)", min_value=0.0, value=float(base_vals.get("Active Max", 0.0)))
                    input_dict["Active Min"] = st.number_input("Active Min (\u03bcs)", min_value=0.0, value=float(base_vals.get("Active Min", 0.0)))
                with d2:
                    input_dict["Idle Mean"] = st.number_input("Idle Mean (\u03bcs)", min_value=0.0, value=float(base_vals.get("Idle Mean", 0.0)))
                    input_dict["Idle Std"] = st.number_input("Idle Std (\u03bcs)", min_value=0.0, value=float(base_vals.get("Idle Std", 0.0)))
                    input_dict["Idle Max"] = st.number_input("Idle Max (\u03bcs)", min_value=0.0, value=float(base_vals.get("Idle Max", 0.0)))
                    input_dict["Idle Min"] = st.number_input("Idle Min (\u03bcs)", min_value=0.0, value=float(base_vals.get("Idle Min", 0.0)))

            submit_btn = st.form_submit_button("⚡ Predict Congestion Level", type="primary")
            
        if submit_btn:
            try:
                with st.spinner("Executing inference across 100 decision trees..."):
                    pred_res = predictor.predict(input_dict, return_probabilities=True)
                    
                p_class = pred_res["predicted_congestion_level"]
                p_conf = pred_res["confidence"]
                p_probs = pred_res["probabilities"]
                
                # Dynamic Visual Result Card
                card_class = f"pred-card-{p_class.lower()}"
                badge_class = "badge-green" if p_class == "LOW" else ("badge-amber" if p_class == "MEDIUM" else "badge-red")
                tier_color = "#34d399" if p_class == "LOW" else ("#fbbf24" if p_class == "MEDIUM" else "#f87171")
                
                st.markdown(f"""
                <div class="prediction-display-card {card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <span class="badge-pill {badge_class}">PREDICTED CONGESTION STATE</span>
                            <span style="color: #94a3b8; font-size: 13px; margin-left: 12px; font-weight: 600;">
                                Confidence Estimate: <b style="color: #ffffff;">{p_conf*100:.1f}%</b>
                            </span>
                        </div>
                        <span class="badge-pill badge-cyan">RANDOM FOREST MODEL</span>
                    </div>
                    <div class="pred-hero-label" style="color: {tier_color};">
                        {p_class} CONGESTION
                    </div>
                    <p style="margin: 0; color: #cbd5e1; font-size: 14px; line-height: 1.6;">
                        {'The flow demonstrates large payloads and prolonged inter-burst silence consistent with quiescent link behavior.' if p_class=='LOW' else ('The flow displays moderate packet activity with intermediate queuing strain.' if p_class=='MEDIUM' else 'The flow demonstrates continuous zero-idle transmission and small packet flood dynamics consistent with high link saturation.')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence Breakdown Chart
                st.markdown("##### 📊 Model Class Probabilities (Tree Voting Breakdown)")
                st.caption("*Note: Probabilities represent Random Forest voting consensus across 100 trees, not direct physical queue overflow guarantees.*")
                
                prob_df = pd.DataFrame({
                    "Congestion Level": ["LOW", "MEDIUM", "HIGH"],
                    "Probability": [p_probs["LOW"], p_probs["MEDIUM"], p_probs["HIGH"]]
                })
                
                chart = alt.Chart(prob_df).mark_bar(cornerRadiusEnd=5, height=22).encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Voting Probability", axis=alt.Axis(format=".1%")),
                    y=alt.Y("Congestion Level:N", sort=["LOW", "MEDIUM", "HIGH"], title=""),
                    color=alt.Color("Congestion Level:N", scale=alt.Scale(
                        domain=["LOW", "MEDIUM", "HIGH"],
                        range=["#10b981", "#f59e0b", "#ef4444"]
                    ), legend=None),
                    tooltip=["Congestion Level", alt.Tooltip("Probability:Q", format=".2%")]
                ).properties(height=140)
                
                st.altair_chart(chart, width="stretch")
                
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

# -----------------------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "📈 Model Performance":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge-container">
            <span class="hero-badge">📊 Empirical Evaluation</span>
            <span class="hero-badge">🔒 40,000 Unseen Holdout Flows</span>
        </div>
        <h1 class="hero-heading">Multi-Model Performance Evaluation</h1>
        <p class="hero-subtext">
            Comprehensive benchmarking comparing Random Forest, Decision Tree, and Logistic Regression 
            strictly on holdout flows from <code>dataset/processed_test.csv</code>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    comp_df, metrics_dict = load_evaluation_data()
    
    if comp_df is None:
        st.warning("Model comparison data not found at `./results/model_comparison.csv`.")
    else:
        # Top Model Benchmark Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class="metric-card-box metric-border-blue">
                <div class="metric-tag-label">Production Model</div>
                <div class="metric-number-display" style="color: #38bdf8;">Random Forest</div>
                <div style="font-size: 12.5px; color: #cbd5e1; margin-top: 4px;">
                    Accuracy: <b>85.39%</b> | Macro F1: <b>85.53%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="metric-card-box metric-border-purple">
                <div class="metric-tag-label">Non-Linear Baseline</div>
                <div class="metric-number-display" style="color: #a78bfa;">Decision Tree</div>
                <div style="font-size: 12.5px; color: #cbd5e1; margin-top: 4px;">
                    Accuracy: <b>84.85%</b> | Macro F1: <b>84.91%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="metric-card-box metric-border-amber">
                <div class="metric-tag-label">Linear Baseline</div>
                <div class="metric-number-display" style="color: #cbd5e1;">Logistic Reg.</div>
                <div style="font-size: 12.5px; color: #cbd5e1; margin-top: 4px;">
                    Accuracy: <b>70.38%</b> | Macro F1: <b>70.59%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Benchmark Table
        st.markdown("""
        <div class="dashboard-panel">
            <div class="panel-header-title">📊 Multi-Model Metric Comparison</div>
            <p style="color: #94a3b8; font-size: 12.5px; margin-bottom: 12px;">
                Evaluated against the exact same 40,000 test partition under strict feature quarantine:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            comp_df.style.format({
                "Accuracy": "{:.2%}",
                "Macro Precision": "{:.2%}",
                "Macro Recall": "{:.2%}",
                "Macro F1": "{:.2%}",
                "Weighted Precision": "{:.2%}",
                "Weighted Recall": "{:.2%}",
                "Weighted F1": "{:.2%}",
                "Testing Time": "{:.4f} s"
            }).highlight_max(subset=["Accuracy", "Macro F1"], color="#1e3a8a"),
            width="stretch"
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Comparison Chart
        comp_chart_path = "./results/model_performance_comparison.png"
        if os.path.exists(comp_chart_path):
            st.markdown("""
            <div class="dashboard-panel">
                <div class="panel-header-title">📈 Metric Radar & Bar Benchmarks</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(comp_chart_path, caption="Comparative Metric Performance Across All 3 Algorithms", width="stretch")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confusion Matrices Section
        st.markdown("""
        <div class="dashboard-panel">
            <div class="panel-header-title">🔍 Confusion Matrix Diagnostics</div>
            <p style="color: #94a3b8; font-size: 12.5px; margin: 0 0 10px 0;">
                Analyze class-by-class classifications across actual holdout ground-truth tiers:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        cm_tabs = st.tabs(["Random Forest (Production)", "Decision Tree", "Logistic Regression"])
        
        with cm_tabs[0]:
            cm_rf_path = "./results/confusion_matrices/random_forest_confusion_matrix.png"
            if os.path.exists(cm_rf_path):
                st.image(cm_rf_path, width=650)
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; padding: 12px 16px; border-radius: 6px; font-size: 12.5px; color: #cbd5e1; margin-top: 10px;">
                <b>Key Validation Takeaways (Random Forest):</b><br>
                • <b>Near-Zero Severe Errors:</b> Only <b>21 out of 13,176 true HIGH flows (0.16%)</b> were misclassified as LOW.<br>
                • <b>Exceptional LOW Precision:</b> Reached <b>96.05%</b>, preventing false alarms during normal quiescent network traffic.<br>
                • <b>Balanced F1:</b> Macro F1 of <b>85.53%</b> verifies equal discriminative power across all three classes.
            </div>
            """, unsafe_allow_html=True)
            
        with cm_tabs[1]:
            cm_dt_path = "./results/confusion_matrices/decision_tree_confusion_matrix.png"
            if os.path.exists(cm_dt_path):
                st.image(cm_dt_path, width=650)
            st.markdown("""
            <div style="background: rgba(167, 139, 250, 0.08); border-left: 4px solid #a78bfa; padding: 12px 16px; border-radius: 6px; font-size: 12.5px; color: #cbd5e1; margin-top: 10px;">
                <b>Key Validation Takeaways (Decision Tree):</b><br>
                • <b>Fast Inference:</b> Evaluated 40,000 flows in 0.094s.<br>
                • <b>Boundary Variance:</b> Slightly higher dispersion between transitional MEDIUM and HIGH congestion boundaries.
            </div>
            """, unsafe_allow_html=True)
            
        with cm_tabs[2]:
            cm_lr_path = "./results/confusion_matrices/logistic_regression_confusion_matrix.png"
            if os.path.exists(cm_lr_path):
                st.image(cm_lr_path, width=650)
            st.markdown("""
            <div style="background: rgba(148, 163, 184, 0.08); border-left: 4px solid #94a3b8; padding: 12px 16px; border-radius: 6px; font-size: 12.5px; color: #cbd5e1; margin-top: 10px;">
                <b>Key Validation Takeaways (Logistic Regression):</b><br>
                • <b>Linear Limitation:</b> Linear hyperplanes struggle with non-linear duty cycle and packet distribution boundaries, topping out at 70.38% accuracy.
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 4: FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
elif page == "🔍 Feature Importance":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge-container">
            <span class="hero-badge">🌲 Gini Impurity Reduction</span>
            <span class="hero-badge">🔍 100 Tree Consensus</span>
        </div>
        <h1 class="hero-heading">Random Forest Feature Importance Analysis</h1>
        <p class="hero-subtext">
            Quantitative analysis of feature influence across 100 decision tree estimators, 
            verifying the physical drivers of traffic congestion classification.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    fi_df = load_feature_importance()
    
    if fi_df is None:
        st.warning("Feature importance table not found at `./results/feature_importance.csv`.")
    else:
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.45); border-left: 4px solid #38bdf8; padding: 11px 15px; border-radius: 6px; font-size: 12.5px; color: #cbd5e1; margin-bottom: 18px;">
            <b>Scientific Disclaimer:</b> Feature importance describes the <b>statistical reliance</b> of the Random Forest model during recursive tree splitting. It denotes correlation and predictive influence, not direct physical causality.
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive Altair Top 10 Horizontal Bar Chart
        st.markdown("##### 🏆 Top 10 Most Influential Predictors")
        top10_df = fi_df.head(10).copy()
        top10_chart = alt.Chart(top10_df).mark_bar(cornerRadiusEnd=5, color="#38bdf8", height=20).encode(
            x=alt.X("Importance:Q", title="Gini Importance Score", axis=alt.Axis(format=".3%")),
            y=alt.Y("Feature:N", sort="-x", title="Network Predictor"),
            tooltip=["Rank:Q", "Feature:N", alt.Tooltip("Importance:Q", format=".4%")]
        ).properties(height=320)
        st.altair_chart(top10_chart, width="stretch")

        fi_chart_path = "./results/graphs/random_forest_feature_importance.png"
        if os.path.exists(fi_chart_path):
            with st.expander("🖼️ View High-Resolution Publication Chart", expanded=False):
                st.image(fi_chart_path, caption="Top 10 Feature Importance Ranking (Gini Metric)", width="stretch")
            
        # Analysis Cards Row
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="dashboard-panel">
                <div class="panel-header-title">📌 Dominant Feature Categories</div>
                <ul style="color: #cbd5e1; font-size: 12.5px; line-height: 1.7; padding-left: 18px; margin-bottom: 0;">
                    <li><b>Packet Morphology (13.17% of Top 10):</b><br><code>Average Packet Size</code>, <code>Packet Length Mean</code>, and <code>Max Packet Length</code> distinguish small control floods from bulk payload transfers.</li>
                    <li><b>Duty Cycle & Silence (17.98% of Top 10):</b><br><code>Idle Max</code>, <code>Idle Min</code>, <code>Active Max</code>, and <code>Idle Mean</code> separate prolonged quiet periods from saturated links.</li>
                    <li><b>Transmission Volume (12.87% of Top 10):</b><br><code>Fwd Header Length</code>, <code>Subflow Fwd Packets</code>, and <code>Total Fwd Packets</code> capture cumulative channel pressure.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown("""
            <div class="dashboard-panel">
                <div class="panel-header-title" style="color: #34d399;">🛡️ Leakage Verification Audit</div>
                <div style="font-size: 12.5px; color: #cbd5e1; line-height: 1.7;">
                    <div>• <code>Flow Bytes/s</code>: <b style="color: #f87171;">Quarantined (Excluded)</b></div>
                    <div>• <code>Flow Packets/s</code>: <b style="color: #f87171;">Quarantined (Excluded)</b></div>
                    <div>• <code>Flow Duration</code>: <b style="color: #f87171;">Quarantined (Excluded)</b></div>
                    <div>• All <code>Flow/Fwd/Bwd IAT</code>: <b style="color: #f87171;">Quarantined (Excluded)</b></div>
                    <div>• Original <code>Label</code>: <b style="color: #f87171;">Quarantined (Excluded)</b></div>
                </div>
                <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 7px 11px; margin-top: 12px; font-size: 11.5px; color: #6ee7b7;">
                    ✓ Audit Passed: Zero target-defining rate metrics contaminate the feature space.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📋 Complete 58-Feature Importance Ranking")
        st.dataframe(
            fi_df.style.format({"Importance": "{:.6f}"}),
            width="stretch",
            height=350
        )

# -----------------------------------------------------------------------------
# PAGE 5: ABOUT THE PROJECT
# -----------------------------------------------------------------------------
elif page == "ℹ️ About the Project":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-badge-container">
            <span class="hero-badge">🎓 Academic Documentation</span>
            <span class="hero-badge">🔬 Capstone Methodology</span>
        </div>
        <h1 class="hero-heading">About the Project</h1>
        <p class="hero-subtext">
            System architecture, mathematical definitions, feature quarantine protocols, 
            and engineering considerations for internet traffic congestion prediction.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns(2)
    
    with col_a1:
        st.markdown("""
        <div class="dashboard-panel">
            <div class="panel-header-title">🎓 Project Overview & Objectives</div>
            <p style="color: #cbd5e1; font-size: 13px; line-height: 1.6; margin-bottom: 0;">
                <b>Title:</b> Internet Traffic Congestion Prediction Using Machine Learning<br>
                <b>Goal:</b> Predict network congestion states (<code>LOW</code>, <code>MEDIUM</code>, <code>HIGH</code>) 
                from passive statistical flow features without relying on invasive hardware probes or direct rate arithmetic.
            </p>
        </div>
        
        <div class="dashboard-panel">
            <div class="panel-header-title">📊 Dataset Telemetry (CIC-IDS2017)</div>
            <p style="color: #cbd5e1; font-size: 13px; line-height: 1.6; margin-bottom: 0;">
                The benchmark dataset includes packet captures from the Canadian Institute for Cybersecurity (CIC). 
                Because CIC-IDS2017 is an intrusion dataset with attack labels rather than congestion labels, 
                an objective proxy target was required to model congestion scientifically.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_a2:
        st.markdown("""
        <div class="dashboard-panel">
            <div class="panel-header-title">🛡️ Methodological Rigor & Target Fitting</div>
            <p style="color: #cbd5e1; font-size: 13px; line-height: 1.6; margin-bottom: 0;">
                <b>1. Strict Train-First Fitting:</b> TIS normalization boundaries and tercile thresholds 
                (&tau;<sub>33.3</sub> = 0.2894, &tau;<sub>66.7</sub> = 0.5255) were computed <b>strictly on the 80% training partition (160,000 flows)</b> 
                and projected out-of-sample onto the 20% test partition (40,000 flows).<br><br>
                <b>2. Zero-Leakage Quarantine:</b> Quarantining all rate/duration/IAT features prevented trivial inverse formula memorization, 
                forcing the model to learn true packet morphology and buffer dynamics.
            </p>
        </div>
        
        <div class="dashboard-panel">
            <div class="panel-header-title">💻 8 GB RAM Memory-Safe Engineering</div>
            <p style="color: #cbd5e1; font-size: 13px; line-height: 1.6; margin-bottom: 0;">
                Every pipeline stage operates within an 8 GB RAM physical memory boundary:
                chunked reading, in-place dtype casting, streaming batch evaluation, and constrained tree depth (16).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(51, 65, 85, 0.5); border-radius: 10px; padding: 18px 22px; margin-top: 10px;">
        <div style="color: #f59e0b; font-weight: 700; font-size: 13.5px; margin-bottom: 5px;">⚠️ Academic Disclaimer</div>
        <div style="color: #94a3b8; font-size: 12px; line-height: 1.6;">
            This software is an academic capstone demonstration platform evaluating statistical flow-level traffic intensity proxies. 
            It does not interface directly with physical router queue counters or execute active network packet modification.
        </div>
    </div>
    """, unsafe_allow_html=True)
