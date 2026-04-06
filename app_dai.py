"""
💎 Diamond Dynamics — Streamlit Web Application
================================================
Modules:
  1️⃣  Price Prediction  — predicts price in USD & INR
  2️⃣  Market Segment   — predicts K-Means cluster label

Run:
    streamlit run app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────
APP_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(APP_DIR)
MODEL_DIR  = os.path.join(BASE_DIR, "models")
PLOT_DIR   = os.path.join(BASE_DIR, "plots")

# Add base dir so pipeline.ANNRegressor can be imported
sys.path.insert(0, BASE_DIR)

USD_TO_INR = 83.5

# ── Ordinal mappings ─────────────────────────
CUT_ORDER     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER   = list("DEFGHIJ")
CLARITY_ORDER = ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"]


# ── Load artefacts ───────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_artefacts():
    def _load(fname):
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            return None
        return joblib.load(path)

    return {
        "reg_model":        _load("best_regression_model.pkl"),
        "encoder":          _load("ordinal_encoder.pkl"),
        "reg_scaler":       _load("regression_scaler.pkl"),
        "km_model":         _load("kmeans_model.pkl"),
        "clus_scaler":      _load("clustering_scaler.pkl"),
        "pca_model":        _load("pca_model.pkl"),
        "cluster_name_map": _load("cluster_name_map.pkl"),
        "reg_features":     _load("regression_feature_cols.pkl"),
        "clus_features":    _load("clustering_feature_cols.pkl"),
        "log_cols":         _load("log_transformed_cols.pkl"),
    }


def models_ready(artefacts: dict) -> bool:
    required = ["reg_model", "encoder", "reg_scaler",
                "km_model", "clus_scaler", "cluster_name_map",
                "reg_features", "clus_features", "log_cols"]
    return all(artefacts.get(k) is not None for k in required)


# ── Feature preparation ───────────────────────
def prepare_features(carat, cut, color, clarity, depth,
                     table, x, y, z, artefacts: dict) -> tuple:
    """
    Build the exact feature vector used during training.
    Returns (reg_vector_scaled, clus_vector_scaled)
    """
    log_cols  = set(artefacts["log_cols"] or [])
    encoder   = artefacts["encoder"]
    reg_feats = artefacts["reg_features"]
    cl_feats  = artefacts["clus_features"]

    # Base row dict
    row = {
        "carat":   carat,
        "cut":     cut,
        "color":   color,
        "clarity": clarity,
        "depth":   depth,
        "table":   table,
        "x":       x,
        "y":       y,
        "z":       z,
    }

    # Log-transform matching training
    for col in ["carat", "x", "y", "z"]:
        if col in log_cols:
            row[col] = np.log1p(row[col])

    # Derived features
    row["volume"]          = row["x"] * row["y"] * row["z"]
    row["dimension_ratio"] = (row["x"] + row["y"]) / (2 * row["z"]) if row["z"] != 0 else 0

    # Ordinal encode
    cat_arr = encoder.transform([[cut, color, clarity]])
    row["cut"], row["color"], row["clarity"] = cat_arr[0]

    df_row = pd.DataFrame([row])

    def _vec(features):
        cols_present = [c for c in features if c in df_row.columns]
        arr = df_row[cols_present].values.astype(float)
        # Pad missing columns with 0
        if len(cols_present) < len(features):
            missing = len(features) - len(cols_present)
            arr = np.hstack([arr, np.zeros((1, missing))])
        return arr

    reg_vec  = _vec(reg_feats)
    clus_vec = _vec(cl_feats)

    reg_vec_sc   = artefacts["reg_scaler"].transform(reg_vec)
    clus_vec_sc  = artefacts["clus_scaler"].transform(clus_vec)
    return reg_vec_sc, clus_vec_sc


# ── UI helpers ────────────────────────────────
def metric_card(label: str, value: str, color: str = "#6c5ce7"):
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border-left: 4px solid {color};
            border-radius: 10px;
            padding: 16px 20px;
            margin: 8px 0;
        ">
            <div style="font-size:0.8rem; color:#888; margin-bottom:4px;">{label}</div>
            <div style="font-size:1.6rem; font-weight:700; color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── PAGE CONFIG ───────────────────────────────
st.set_page_config(
    page_title="💎 Diamond Dynamics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton > button {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        color: white; border: none; border-radius: 8px;
        padding: 0.5rem 2rem; font-weight: 600; font-size: 1rem;
        transition: transform 0.15s;
    }
    .stButton > button:hover { transform: translateY(-2px); }
    h1 { color: #a29bfe !important; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #a29bfe; margin: 16px 0 8px;
        border-bottom: 1px solid #333; padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────
with st.sidebar:
    st.markdown("## 💎 Diamond Dynamics")
    st.markdown("**Price Prediction & Market Segmentation**")
    st.divider()
    page = st.radio("Navigation", ["🏠 Home", "💰 Price Prediction",
                                    "🔵 Market Segmentation", "📊 EDA Insights"])
    st.divider()
    st.caption("Powered by RandomForest / ANN + K-Means")
    st.caption("Domain: Luxury Goods Analytics")

# ── LOAD MODELS ───────────────────────────────
artefacts = load_artefacts()
trained   = models_ready(artefacts)

# ════════════════════════════════════════
# INPUT FORM (shared)
# ════════════════════════════════════════
def diamond_input_form(key_prefix: str = ""):
    st.markdown('<div class="section-header">💎 Diamond Physical Attributes</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        carat   = st.number_input("Carat Weight", 0.20, 5.0, 0.90, 0.01,
                                   key=f"{key_prefix}_carat",
                                   help="Weight of the diamond in carats")
        depth   = st.number_input("Depth %", 43.0, 80.0, 61.7, 0.1,
                                   key=f"{key_prefix}_depth")
        x       = st.number_input("Length x (mm)", 0.1, 12.0, 4.5, 0.01,
                                   key=f"{key_prefix}_x")
        z       = st.number_input("Height z (mm)", 0.1, 8.0, 2.9, 0.01,
                                   key=f"{key_prefix}_z")
    with c2:
        table   = st.number_input("Table %", 43.0, 100.0, 57.5, 0.1,
                                   key=f"{key_prefix}_table")
        y       = st.number_input("Width y (mm)", 0.1, 12.0, 4.5, 0.01,
                                   key=f"{key_prefix}_y")

    st.markdown('<div class="section-header">🏷️ Quality Grades</div>',
                unsafe_allow_html=True)
    c3, c4, c5 = st.columns(3)
    with c3:
        cut     = st.selectbox("Cut Quality", CUT_ORDER,
                               index=CUT_ORDER.index("Ideal"),
                               key=f"{key_prefix}_cut")
    with c4:
        color   = st.selectbox("Color Grade", COLOR_ORDER,
                               index=COLOR_ORDER.index("E"),
                               key=f"{key_prefix}_color")
    with c5:
        clarity = st.selectbox("Clarity Grade", CLARITY_ORDER,
                               index=CLARITY_ORDER.index("VS1"),
                               key=f"{key_prefix}_clarity")
    return carat, cut, color, clarity, depth, table, x, y, z

# ════════════════════════════════════════
# HOME
# ════════════════════════════════════════
if page == "🏠 Home":
    st.title("💎 Diamond Dynamics")
    st.subheader("Price Prediction & Market Segmentation Platform")
    st.markdown("""
    Welcome to **Diamond Dynamics** — a production-ready ML platform that:
    - 🔮 **Predicts diamond prices** (USD & INR) using trained regression models
    - 🎯 **Segments diamonds** into market categories using K-Means clustering
    - 📊 **Visualises insights** from ~54,000 real diamond records

    ---
    ### How to Use
    1. Navigate to **💰 Price Prediction** and enter diamond attributes
    2. Click **Predict Price** to get the estimated market value
    3. Navigate to **🔵 Market Segmentation** to find the diamond's market cluster
    4. Explore **📊 EDA Insights** for data visualisations
    """)

    # if not trained:
    #     st.warning("""
    #     ⚠️ **Models not yet trained.**
    #     Run the training pipeline first:
    #     ```bash
    #     cd diamond_dynamics
    #     python pipeline.py
    #     ```
    #     Then refresh this app.
    #     """)
    # else:
    #     st.success("✅ All models loaded and ready!")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", "~54,000 rows")
    with col2:
        st.metric("Features", "10 original + engineered")
    with col3:
        st.metric("Models", "5 ML + ANN + K-Means")

# ════════════════════════════════════════
# PRICE PREDICTION
# ════════════════════════════════════════
elif page == "💰 Price Prediction":
    st.title("💰 Diamond Price Prediction")
    st.markdown("Enter diamond attributes to predict its market price.")

    if not trained:
        st.error("⚠️ Models not found. Please run `pipeline.py` first.")
        st.stop()

    with st.form("price_form"):
        carat, cut, color, clarity, depth, table, x, y, z = diamond_input_form("reg")
        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)

    if submitted:
        with st.spinner("Calculating …"):
            try:
                reg_vec, _ = prepare_features(
                    carat, cut, color, clarity, depth, table, x, y, z, artefacts
                )
                reg_model = artefacts["reg_model"]
                log_pred  = reg_model.predict(reg_vec)[0]
                # Reverse log1p if price was log-transformed
                if "price" in (artefacts["log_cols"] or []):
                    pred_usd = np.expm1(log_pred)
                else:
                    pred_usd = log_pred
                pred_usd = max(0, pred_usd)
                pred_inr = pred_usd * USD_TO_INR

                st.markdown("---")
                st.subheader("💎 Prediction Results")
                c1, c2 = st.columns(2)
                with c1:
                    metric_card("Predicted Price (USD)", f"${pred_usd:,.0f}", "#6c5ce7")
                with c2:
                    metric_card("Predicted Price (INR)", f"₹{pred_inr:,.0f}", "#00b894")

                # Summary
                st.markdown("---")
                st.markdown("**📋 Input Summary**")
                summary = pd.DataFrame([{
                    "Carat": carat, "Cut": cut, "Color": color,
                    "Clarity": clarity, "Depth": depth, "Table": table,
                    "x": x, "y": y, "z": z
                }])
                st.dataframe(summary, use_container_width=True, hide_index=True)

                # Price range indicator
                st.markdown("---")
                tier = ("💎 Premium" if pred_usd > 8000
                        else "🟡 Mid-range" if pred_usd > 2000
                        else "🟢 Budget-friendly")
                st.info(f"**Price Tier:** {tier}")

            except Exception as e:
                st.error(f"Prediction error: {e}")


# ════════════════════════════════════════
# MARKET SEGMENTATION
# ════════════════════════════════════════
elif page == "🔵 Market Segmentation":
    st.title("🔵 Market Segment Prediction")
    st.markdown("Identify which market category your diamond belongs to.")

    if not trained:
        st.error("⚠️ Models not found. Please run `pipeline.py` first.")
        st.stop()

    with st.form("clus_form"):
        carat, cut, color, clarity, depth, table, x, y, z = diamond_input_form("cl")
        submitted = st.form_submit_button("🎯 Predict Segment", use_container_width=True)

    if submitted:
        with st.spinner("Segmenting …"):
            try:
                _, clus_vec = prepare_features(
                    carat, cut, color, clarity, depth, table, x, y, z, artefacts
                )
                km_model   = artefacts["km_model"]
                name_map   = artefacts["cluster_name_map"]
                cluster_id = int(km_model.predict(clus_vec)[0])
                seg_name   = name_map.get(cluster_id, f"Cluster {cluster_id}")

                st.markdown("---")
                st.subheader("🎯 Segmentation Result")
                col_a, col_b = st.columns(2)
                with col_a:
                    metric_card("Cluster ID", f"#{cluster_id}", "#e17055")
                with col_b:
                    metric_card("Market Segment", seg_name, "#fdcb6e")

                # Segment description
                descriptions = {
                    "Affordable Small Diamonds": (
                        "Light-weight, budget-friendly stones. Ideal for everyday jewellery "
                        "and entry-level buyers. Typically below 0.5 carats."
                    ),
                    "Mid-range Balanced Diamonds": (
                        "Well-balanced diamonds offering a good quality-to-price ratio. "
                        "Popular for engagement rings and gifting. 0.5–1.5 carats."
                    ),
                    "Premium Heavy Diamonds": (
                        "Large, high-quality diamonds commanding top market prices. "
                        "Sought by collectors and high-end retailers. Above 1.5 carats."
                    ),
                }
                desc = descriptions.get(seg_name, "A distinct market segment based on diamond characteristics.")
                st.info(f"**ℹ️ About this segment:** {desc}")

                # PCA visualisation from saved plot
                pca_plot = os.path.join(PLOT_DIR, "10_cluster_pca_2d.png")
                if os.path.exists(pca_plot):
                    st.markdown("---")
                    st.markdown("**📊 Cluster Distribution (PCA 2D)**")
                    st.image(pca_plot, use_container_width=True)

            except Exception as e:
                st.error(f"Segmentation error: {e}")


# ════════════════════════════════════════
# EDA INSIGHTS
# ════════════════════════════════════════
elif page == "📊 EDA Insights":
    st.title("📊 EDA Insights")
    st.markdown("Exploratory data analysis visualisations from the training dataset.")

    plot_files = {
        "Price Distribution":          "01_price_distribution.png",
        "Carat Distribution":          "02_carat_distribution.png",
        "Categorical Counts":          "03_categorical_counts.png",
        "Price by Cut & Color":        "04_price_by_cut_color.png",
        "Correlation Heatmap":         "05_correlation_heatmap.png",
        "Carat vs Price":              "06_carat_vs_price.png",
        "Average Price by Clarity":    "07_avg_price_by_clarity.png",
        "Model Comparison":            "08_model_comparison.png",
        "Elbow & Silhouette":          "09_clustering_elbow_silhouette.png",
        "Cluster PCA Projection":      "10_cluster_pca_2d.png",
    }

    # Grid layout
    items = list(plot_files.items())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, (title, fname) in enumerate(items[i:i+2]):
            path = os.path.join(PLOT_DIR, fname)
            with cols[j]:
                st.markdown(f"**{title}**")
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.caption("Run `pipeline.py` to generate this plot.")

    # Model metrics table
    metrics_csv = os.path.join(PLOT_DIR, "model_metrics.csv")
    if os.path.exists(metrics_csv):
        st.markdown("---")
        st.subheader("📋 Model Evaluation Metrics")
        st.dataframe(
            pd.read_csv(metrics_csv).style.highlight_max(
                subset=["R²"], color="#6c5ce7AA"
            ).highlight_min(
                subset=["MAE", "RMSE"], color="#00b89444"
            ),
            use_container_width=True,
        )
