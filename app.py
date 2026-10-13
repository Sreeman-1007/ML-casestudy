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
    page_title="Internet Traffic Congestion Prediction",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS (DARK NAVY & BLUE THEME)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Dark Navy & Slate Theme */
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] .stMarkdown h1, 
    section[data-testid="stSidebar"] .stMarkdown h2, 
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #60a5fa;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .header-title {
        font-size: 28px;
        font-weight: 800;
        color: #38bdf8;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 15px;
        color: #94a3b8;
        margin: 0;
        font-weight: 400;
    }
    
    /* Metric Cards */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 2px;
    }
    .metric-desc {
        font-size: 12px;
        color: #64748b;
    }
    
    /* Result Cards */
    .result-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #10b981;
        border-radius: 12px;
        padding: 22px;
        margin-top: 16px;
    }
    .result-card-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #f59e0b;
        border-radius: 12px;
        padding: 22px;
        margin-top: 16px;
    }
    .result-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1.5px solid #ef4444;
        border-radius: 12px;
        padding: 22px;
        margin-top: 16px;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .badge-blue { background-color: #0369a1; color: #e0f2fe; }
    .badge-green { background-color: #065f46; color: #d1fae5; }
    .badge-amber { background-color: #78350f; color: #fef3c7; }
    .badge-red { background-color: #7f1d1d; color: #fee2e2; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 6px 6px 0 0;
        color: #94a3b8;
        border: 1px solid #334155;
        border-bottom: none;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid #334155;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CACHED RESOURCE LOADERS
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

# Load global predictor
try:
    predictor = load_predictor()
    model_loaded = True
except Exception as e:
    predictor = None
    model_loaded = False
    model_error = str(e)

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌐 Navigation")
    page = st.radio(
        "Select Dashboard View:",
        [
            "📊 Dashboard Overview",
            "🔮 Congestion Prediction",
            "📈 Model Performance",
            "🔍 Feature Importance",
            "ℹ️ About the Project"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    if model_loaded:
        st.markdown(
            '<div style="background-color: #064e3b; border: 1px solid #059669; border-radius: 6px; padding: 10px;">'
            '<span style="color: #34d399; font-weight: 700;">● Model Online</span><br>'
            '<small style="color: #a7f3d0;">Random Forest (100 Trees)</small><br>'
            '<small style="color: #a7f3d0;">Test Accuracy: <b>85.39%</b></small>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.error("Model Offline")
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("College Machine Learning Capstone")
    st.caption("CIC-IDS2017 Dataset Telemetry")
    st.caption("Environment: 8 GB RAM Optimized")

# -----------------------------------------------------------------------------
# PAGE 1: DASHBOARD OVERVIEW
# -----------------------------------------------------------------------------
if page == "📊 Dashboard Overview":
    # Header Banner
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">Internet Traffic Congestion Prediction</h1>
        <p class="header-subtitle">Machine Learning-Based Network Traffic Flow Analysis & Queue Strain Estimation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Selected Model</div>
            <div class="metric-value" style="color: #38bdf8;">Random Forest</div>
            <div class="metric-desc">100 Trees Ensemble (Depth 16)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Test Set Accuracy</div>
            <div class="metric-value" style="color: #34d399;">85.39%</div>
            <div class="metric-desc">Tested on 40,000 Unseen Flows</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Macro F1-Score</div>
            <div class="metric-value" style="color: #a78bfa;">85.53%</div>
            <div class="metric-desc">Balanced Across All 3 Tiers</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Predictor Features</div>
            <div class="metric-value" style="color: #fbbf24;">58 Features</div>
            <div class="metric-desc">Zero Target Leakage Quarantine</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Overview Layout: Two Columns
    c_left, c_right = st.columns([3, 2])
    
    with c_left:
        st.subheader("🎯 Project Purpose & Methodology")
        st.markdown("""
        This machine learning platform predicts network traffic congestion states from statistical flow telemetry 
        derived from the benchmark **CIC-IDS2017** dataset.
        
        Because CIC-IDS2017 is an intrusion dataset without native router drop counters, the ground-truth target is formulated as a 
        mathematically rigorous **Traffic Intensity Score (TIS)** combining normalized logarithmic bandwidth demand and packet processing load:
        """)
        st.latex(r"\text{TIS} = 0.5 \cdot \text{norm}(\log_{10}(1 + \text{Flow Bytes/s})) + 0.5 \cdot \text{norm}(\log_{10}(1 + \text{Flow Packets/s}))")
        
        st.markdown("""
        #### 🛡️ Strict Target Leakage Quarantine
        To guarantee scientific integrity and prevent the ML model from merely inverting an arithmetic formula:
        - **Quarantined (Excluded):** `Flow Bytes/s`, `Flow Packets/s`, `Flow Duration`, and all 14 `Flow/Fwd/Bwd IAT` features.
        - **Allowed Predictors (58 features):** Packet morphology distributions, TCP control flags, window buffer headroom, and active/idle duty cycles.
        """)
        
    with c_right:
        st.subheader("🚦 Congestion Level Taxonomy")
        st.markdown("""
        <div style="background-color: #1e293b; border-left: 4px solid #10b981; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">
            <b style="color: #34d399; font-size: 15px;">🟢 LOW Congestion (TIS &le; 0.2894)</b><br>
            <span style="color: #cbd5e1; font-size: 13px;">Quiescent baseline traffic; prolonged idle pauses (median ~5.86s), high client window buffer headroom, and large MTU payloads.</span>
        </div>
        
        <div style="background-color: #1e293b; border-left: 4px solid #f59e0b; padding: 12px 16px; border-radius: 4px; margin-bottom: 12px;">
            <b style="color: #fbbf24; font-size: 15px;">🟡 MEDIUM Congestion (0.2894 &lt; TIS &le; 0.5255)</b><br>
            <span style="color: #cbd5e1; font-size: 13px;">Moderate transmission load; intermediate burst activity, emerging queuing strain, transitional buffer occupancy.</span>
        </div>
        
        <div style="background-color: #1e293b; border-left: 4px solid #ef4444; padding: 12px 16px; border-radius: 4px;">
            <b style="color: #f87171; font-size: 15px;">🔴 HIGH Congestion (TIS &gt; 0.5255)</b><br>
            <span style="color: #cbd5e1; font-size: 13px;">Severe network saturation; continuous zero-idle transmission, small-packet flood dynamics (median ~16 bytes), reduced TCP window capacity.</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.subheader("🚀 How to Obtain Predictions")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        **1. Single-Flow Manual / Preset Telemetry**
        - Navigate to **🔮 Congestion Prediction**.
        - Load a real-world demonstration preset or input custom telemetry values across the 58 network fields.
        - Click **Predict Congestion Level** to receive an instant classification and confidence probabilities.
        """)
    with p2:
        st.markdown("""
        **2. Batch Flow CSV Upload**
        - Upload any CSV export containing the 58 standard predictor columns.
        - The system validates missing columns, infers congestion tiers in batches, and lets you download an annotated CSV.
        """)

# -----------------------------------------------------------------------------
# PAGE 2: CONGESTION PREDICTION
# -----------------------------------------------------------------------------
elif page == "🔮 Congestion Prediction":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">Network Congestion Inference Engine</h1>
        <p class="header-subtitle">Real-time classification using the production Random Forest Classifier</p>
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
        st.subheader("📁 Upload Flow Telemetry CSV")
        st.markdown("""
        Upload a CSV file containing network flow records. The file must contain the **58 required predictor columns**. 
        *(Note: Target column `Congestion_Level` and quarantined rate columns must NOT be present).*
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read sample / head
                df_upload = pd.read_csv(uploaded_file, low_memory=False)
                st.success(f"Successfully loaded CSV with **{len(df_upload):,} rows** and **{len(df_upload.columns)} columns**.")
                
                # Check for target column in input
                if "Congestion_Level" in df_upload.columns:
                    st.warning("⚠️ Column 'Congestion_Level' detected in uploaded file. It will be ignored during inference.")
                    df_upload = df_upload.drop(columns=["Congestion_Level"])
                    
                # Preview uploaded data
                st.markdown("##### Uploaded Telemetry Preview (First 5 Rows):")
                st.dataframe(df_upload.head(5), height=180)
                
                if st.button("🚀 Run Batch Congestion Prediction", type="primary"):
                    with st.spinner(f"Evaluating {len(df_upload):,} flow records..."):
                        t_start = time.perf_counter()
                        
                        # Run batch prediction via predict_congestion
                        batch_results = predictor.predict(df_upload, return_probabilities=True)
                        t_elapsed = time.perf_counter() - t_start
                        
                        # Extract predictions and confidences
                        preds = [r["predicted_congestion_level"] for r in batch_results]
                        confs = [r["confidence"] for r in batch_results]
                        
                        # Add results to dataframe
                        df_result = df_upload.copy()
                        df_result.insert(0, "Predicted_Congestion_Level", preds)
                        df_result.insert(1, "Prediction_Confidence", confs)
                        
                        st.success(f"Completed batch prediction in **{t_elapsed:.3f} seconds** ({t_elapsed/len(df_upload)*1e6:.1f} \u03bcs/sample)!")
                        
                        # Summary metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Total Flows Evaluated", f"{len(df_result):,}")
                        c2.metric("LOW Congestion Flows", f"{(df_result['Predicted_Congestion_Level']=='LOW').sum():,}")
                        c3.metric("MEDIUM Congestion Flows", f"{(df_result['Predicted_Congestion_Level']=='MEDIUM').sum():,}")
                        c4.metric("HIGH Congestion Flows", f"{(df_result['Predicted_Congestion_Level']=='HIGH').sum():,}")
                        
                        # Preview results
                        st.markdown("##### Annotated Results Preview:")
                        st.dataframe(df_result.head(10), height=250)
                        
                        # Download button
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
        st.subheader("⚡ Single-Flow Telemetry Prediction")
        st.markdown("Select a real test demonstration preset or input custom flow parameters:")
        
        presets = load_demonstration_presets()
        preset_names = ["Custom / Manual Input"] + list(presets.keys())
        
        selected_preset = st.selectbox(
            "Load Demonstration Telemetry Profile:",
            preset_names,
            index=1 if len(preset_names) > 1 else 0
        )
        
        # Base values from selected preset
        base_vals = presets.get(selected_preset, {}) if selected_preset != "Custom / Manual Input" else {}
        
        if selected_preset != "Custom / Manual Input":
            st.info(f"Loaded telemetry profile: **{selected_preset}** (extracted from unseen test partition).")
            
        # Form for entering the 58 features in structured expandable categories
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
                with st.spinner("Analyzing network flow dynamics..."):
                    pred_res = predictor.predict(input_dict, return_probabilities=True)
                    
                p_class = pred_res["predicted_congestion_level"]
                p_conf = pred_res["confidence"]
                p_probs = pred_res["probabilities"]
                
                # Card Styling based on class
                card_class = f"result-card-{p_class.lower()}"
                badge_class = "badge-green" if p_class == "LOW" else ("badge-amber" if p_class == "MEDIUM" else "badge-red")
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                        <div>
                            <span class="badge {badge_class}">{p_class} CONGESTION</span>
                            <span style="color: #94a3b8; font-size: 13px; margin-left: 10px;">Model Confidence: <b>{p_conf*100:.1f}%</b></span>
                        </div>
                        <small style="color: #64748b;">Random Forest Model (58 Features)</small>
                    </div>
                    <h2 style="margin: 0 0 8px 0; color: #f8fafc; font-size: 24px;">Predicted Network State: <b>{p_class}</b></h2>
                    <p style="margin: 0; color: #cbd5e1; font-size: 14px;">
                        {'The flow displays large payloads and long inter-burst idle pauses indicative of non-congested traffic.' if p_class=='LOW' else ('The flow displays moderate packet activity with transitional buffer pressure.' if p_class=='MEDIUM' else 'The flow displays continuous zero-idle transmission and small packet flood dynamics consistent with high congestion.')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📊 Model Class Probabilities (Confidence Breakdown)")
                st.caption("*Note: Probabilities represent Random Forest tree voting confidence, not direct physical queue overflow guarantees.*")
                
                prob_df = pd.DataFrame({
                    "Congestion Level": ["LOW", "MEDIUM", "HIGH"],
                    "Probability": [p_probs["LOW"], p_probs["MEDIUM"], p_probs["HIGH"]]
                })
                
                chart = alt.Chart(prob_df).mark_bar(cornerRadius=4).encode(
                    x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]), title="Prediction Probability"),
                    y=alt.Y("Congestion Level:N", sort=["LOW", "MEDIUM", "HIGH"], title=""),
                    color=alt.Color("Congestion Level:N", scale=alt.Scale(
                        domain=["LOW", "MEDIUM", "HIGH"],
                        range=["#10b981", "#f59e0b", "#ef4444"]
                    ), legend=None),
                    tooltip=["Congestion Level", alt.Tooltip("Probability:Q", format=".2%")]
                ).properties(height=140)
                
                st.altair_chart(chart, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

# -----------------------------------------------------------------------------
# PAGE 3: MODEL PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "📈 Model Performance":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">Multi-Model Performance Evaluation</h1>
        <p class="header-subtitle">Empirical validation results on 40,000 unseen holdout test flows</p>
    </div>
    """, unsafe_allow_html=True)
    
    comp_df, metrics_dict = load_evaluation_data()
    
    if comp_df is None:
        st.warning("Model comparison data not found at `./results/model_comparison.csv`.")
    else:
        # Comparison Table
        st.subheader("📊 Comparative Test Performance Table")
        st.markdown("All metrics were computed strictly on the **40,000 unseen test records** from `processed_test.csv`:")
        
        # Display formatted table
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
            use_container_width=True
        )
        
        # Performance comparison chart
        comp_chart_path = "./results/model_performance_comparison.png"
        if os.path.exists(comp_chart_path):
            st.image(comp_chart_path, caption="Comparative Performance Across All Evaluated Metrics", use_container_width=True)
            
        st.markdown("---")
        st.subheader("🔍 Confusion Matrix Analysis")
        st.markdown("Detailed breakdown of predictions across actual test ground-truth tiers:")
        
        cm_tabs = st.tabs(["Random Forest (Best Model)", "Decision Tree", "Logistic Regression"])
        
        with cm_tabs[0]:
            cm_rf_path = "./results/confusion_matrices/random_forest_confusion_matrix.png"
            if os.path.exists(cm_rf_path):
                st.image(cm_rf_path, width=650)
            st.markdown("""
            **Key Findings for Random Forest:**
            - **Accuracy:** 85.39% | **Macro F1:** 85.53%
            - **Non-Adjacent Error Rate:** Only **21 out of 13,176 true HIGH flows (0.16%)** were misclassified as LOW.
            - **LOW Precision:** Reached **96.05%**, preventing false alarms during normal operations.
            """)
            
        with cm_tabs[1]:
            cm_dt_path = "./results/confusion_matrices/decision_tree_confusion_matrix.png"
            if os.path.exists(cm_dt_path):
                st.image(cm_dt_path, width=650)
            st.markdown("""
            **Key Findings for Decision Tree:**
            - **Accuracy:** 84.85% | **Macro F1:** 84.91%
            - Fast inference (0.094s for 40,000 flows). Strong non-linear splits, but slightly higher boundary variance on transitional MEDIUM flows.
            """)
            
        with cm_tabs[2]:
            cm_lr_path = "./results/confusion_matrices/logistic_regression_confusion_matrix.png"
            if os.path.exists(cm_lr_path):
                st.image(cm_lr_path, width=650)
            st.markdown("""
            **Key Findings for Logistic Regression:**
            - **Accuracy:** 70.38% | **Macro F1:** 70.59%
            - Serves as a solid linear baseline, but linear hyperplanes struggle with complex non-linear duty cycle and packet sizing thresholds.
            """)

# -----------------------------------------------------------------------------
# PAGE 4: FEATURE IMPORTANCE
# -----------------------------------------------------------------------------
elif page == "🔍 Feature Importance":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">Random Forest Feature Importance Analysis</h1>
        <p class="header-subtitle">Gini impurity reduction across 100 decision tree estimators</p>
    </div>
    """, unsafe_allow_html=True)
    
    fi_df = load_feature_importance()
    
    if fi_df is None:
        st.warning("Feature importance table not found at `./results/feature_importance.csv`.")
    else:
        st.subheader("🏆 Top 10 Most Important Network Features")
        st.markdown("""
        > **Scientific Disclaimer:** Feature importance reflects the **statistical reliance** of the Random Forest model during recursive tree splitting. It does **not** assert physical causation.
        """)
        
        # Interactive Horizontal Bar Chart of Top 10 Features
        top10_df = fi_df.head(10).copy()
        top10_chart = alt.Chart(top10_df).mark_bar(cornerRadiusEnd=4, color="#38bdf8").encode(
            x=alt.X("Importance:Q", title="Importance Score (Mean Decrease in Impurity)", axis=alt.Axis(format=".3%")),
            y=alt.Y("Feature:N", sort="-x", title="Network Feature"),
            tooltip=["Rank:Q", "Feature:N", alt.Tooltip("Importance:Q", format=".4%")]
        ).properties(height=340)
        st.altair_chart(top10_chart, use_container_width=True)

        # Reusable existing high-resolution chart
        fi_chart_path = "./results/graphs/random_forest_feature_importance.png"
        if os.path.exists(fi_chart_path):
            with st.expander("🖼️ View High-Resolution Publication Chart", expanded=False):
                st.image(fi_chart_path, caption="Top 10 Feature Importance Ranking (Gini Metric)", use_container_width=True)
            
        # Top 10 Summary
        top10 = fi_df.head(10).copy()
        
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("##### 📌 Dominant Feature Categories:")
            st.markdown("""
            1. **Packet Morphology (13.17% of Top 10):**
               `Average Packet Size`, `Packet Length Mean`, `Max Packet Length` distinguish small control floods from bulk transfers.
            2. **Duty Cycle & Channel Silence (17.98% of Top 10):**
               `Idle Max`, `Idle Min`, `Active Max`, `Idle Mean` separate prolonged quiet periods from saturated links.
            3. **Transmission Volume (12.87% of Top 10):**
               `Fwd Header Length`, `Subflow Fwd Packets`, `Total Fwd Packets` capture cumulative transmission push.
            """)
            
        with col_t2:
            st.markdown("##### 🛡️ Leakage Verification Status:")
            st.markdown("""
            - `Flow Bytes/s`: **Quarantined (Excluded)**
            - `Flow Packets/s`: **Quarantined (Excluded)**
            - `Flow Duration`: **Quarantined (Excluded)**
            - All `Flow/Fwd/Bwd IAT`: **Quarantined (Excluded)**
            - Original `Label`: **Quarantined (Excluded)**
            
            *Verification Passed: Zero target-defining rate metrics contaminate the feature space.*
            """)
            
        st.markdown("---")
        st.subheader("📋 Complete 58-Feature Importance Ranking")
        st.dataframe(
            fi_df.style.format({"Importance": "{:.6f}"}),
            use_container_width=True,
            height=350
        )

# -----------------------------------------------------------------------------
# PAGE 5: ABOUT THE PROJECT
# -----------------------------------------------------------------------------
elif page == "ℹ️ About the Project":
    st.markdown("""
    <div class="header-box">
        <h1 class="header-title">About the Project</h1>
        <p class="header-subtitle">College Machine Learning Capstone Project Architecture & Documentation</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 Project Details
    - **Title:** Internet Traffic Congestion Prediction Using Machine Learning
    - **Dataset:** CIC-IDS2017 (Canadian Institute for Cybersecurity)
    - **Target Scope:** 3-Tier Congestion State Estimation (`LOW`, `MEDIUM`, `HIGH`)
    - **Hardware Constraint:** 8 GB RAM Optimized Architecture
    
    ---
    
    ### 🔬 Scientific Methodology & Target Derivation
    CIC-IDS2017 is an intrusion benchmark with attack categories (`BENIGN`, `DDoS`, `PortScan`, `Bot`, etc.) rather than native congestion labels.
    
    To formulate a legitimate machine learning task:
    1. **Traffic Intensity Score (TIS):** Defined mathematically from normalized log-rate telemetry.
    2. **Strict Train-First Fitting:** Normalization bounds and tercile thresholds ($\tau_{33.3}=0.2894, \tau_{66.7}=0.5255$) were fitted **exclusively on the 80% training split (160,000 flows)** and applied out-of-sample to the 20% test split (40,000 flows).
    3. **Target Leakage Prevention:** All direct components of TIS (`Flow Bytes/s`, `Flow Packets/s`) and all algebraically coupled duration/IAT features were placed in a strict **Feature Quarantine** and deleted from the ML feature matrix.
    4. **Model Evaluation:** The production Random Forest model achieved **85.39% Accuracy** and **85.53% Macro F1** using only physical packet morphology, TCP flags, buffering, and duty cycle indicators.
    
    ---
    
    ### ⚠️ Academic Disclaimer
    This software is a capstone demonstration system evaluating flow-level traffic intensity proxies. It does not interface directly with physical router queue counters or execute active network packet modification.
    """)
