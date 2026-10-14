"""
UHPSFRC Compressive Strength Predictor
--------------------------------------
Streamlit GUI for the paper:
"Interpretable Machine Learning Framework for Compressive Strength Prediction of
 Ultra-High Performance Steel Fiber Reinforced Concrete (UHPSFRC) with Practical GUI Deployment"

Repo layout expected:
    app.py
    best_model.pkl        <- trained Gradient Boosting pipeline
    best_model.json       <- model metadata (optional, used for the R2 badge)
    requirements.txt
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="UHPSFRC Strength Predictor",
    page_icon="🏗️",
    layout="wide",
)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
META_PATH = BASE_DIR / "best_model.json"

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 1250px;}
        .hero {
            background: linear-gradient(120deg, #0f2a43 0%, #1d5a8a 100%);
            padding: 1.6rem 2rem; border-radius: 14px; color: white; margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0; font-size: 1.9rem; color: white;}
        .hero p  {margin: .4rem 0 0 0; opacity: .88; font-size: .95rem;}
        .result-card {
            background: linear-gradient(135deg, #0f2a43, #1d5a8a);
            border-radius: 14px; padding: 1.6rem; text-align: center; color: white;
        }
        .result-card .val {font-size: 3.2rem; font-weight: 700; line-height: 1.1;}
        .result-card .unit {font-size: 1.1rem; opacity: .85;}
        .result-card .cls {margin-top: .6rem; font-size: 1rem; font-weight: 600;}
        div[data-testid="stMetric"] {
            background: rgba(120,140,160,.10); border-radius: 10px; padding: .6rem .9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta():
    if META_PATH.exists():
        with open(META_PATH) as f:
            return json.load(f)
    return {}


model = load_model()
meta = load_meta()

# Exact column names / order the pipeline was trained on.
# NOTE: the training sheet contains Cement, Blast furnace slag and Silica fume
# twice (".1" copies, identical values). The GUI asks once and feeds both.
FEATURES = list(model.feature_names_in_)

FIBER_TYPES = [
    "Straight Steel Fiber",
    "Hooked-End Steel Fiber",
    "Crimped Steel Fiber",
    "Twisted (Torex) Steel Fiber",
    "Mix (Straight + Hooked-End)",
    "Mix (Straight + Twisted)",
]
# Valid ranges come from the training dataset (min, max)
CURING_PERIODS = [1, 3, 7, 14, 28, 56, 90, 180, 365]

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏗️ UHPSFRC Compressive Strength Predictor</h1>
        <p>Interpretable machine-learning framework for ultra-high performance steel fiber
        reinforced concrete. Enter the mix design, fiber details and curing regime,
        then click <b>Predict</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Inputs  (every field starts with a sensible default value)
# ----------------------------------------------------------------------------
left, right = st.columns([1.55, 1], gap="large")

with left:
    tab_bind, tab_agg, tab_fiber, tab_cure = st.tabs(
        ["🧱 Binders", "💧 Aggregate, Water & Admixture", "🧵 Steel Fiber", "🌡️ Curing"]
    )

    with tab_bind:
        st.caption("Cementitious materials and powders (kg/m³)")
        c1, c2 = st.columns(2)
        cement = c1.number_input("Cement", 270.0, 1260.0, 720.0, 10.0, help="Range in data: 270 – 1251 kg/m³")
        slag = c2.number_input("Blast furnace slag", 0.0, 400.0, 0.0, 5.0, help="Range in data: 0 – 375 kg/m³")
        silica = c1.number_input("Silica fume", 0.0, 450.0, 120.0, 5.0, help="Range in data: 0 – 434 kg/m³")
        fly = c2.number_input("Fly ash", 0.0, 360.0, 0.0, 5.0, help="Range in data: 0 – 356 kg/m³")
        lime = c1.number_input("Limestone powder", 0.0, 1100.0, 0.0, 10.0, help="Range in data: 0 – 1058 kg/m³")
        quartz = c2.number_input("Quartz powder", 0.0, 400.0, 0.0, 5.0, help="Range in data: 0 – 397 kg/m³")
        nano = c1.number_input("Nano silica", 0.0, 50.0, 0.0, 0.5, help="Range in data: 0 – 47.5 kg/m³")

    with tab_agg:
        st.caption("Aggregate, mixing water and superplasticizer (kg/m³)")
        c1, c2, c3 = st.columns(3)
        agg = c1.number_input("Aggregate (total)", 400.0, 2000.0, 1100.0, 10.0, help="Range in data: 408 – 1992 kg/m³")
        water = c2.number_input("Water", 90.0, 275.0, 160.0, 1.0, help="Range in data: 90 – 272.6 kg/m³")
        sp = c3.number_input("Superplasticizer", 1.0, 60.0, 25.0, 0.5, help="Range in data: 1.1 – 57 kg/m³")

    with tab_fiber:
        st.caption("Steel fiber characteristics")
        fiber_type = st.selectbox("Steel fiber type", FIBER_TYPES, index=0)
        c1, c2 = st.columns(2)
        aspect = c1.number_input("Aspect ratio (L/d)", 30, 90, 65, 1, help="Range in data: 30 – 90")
        fiber_amt = c2.number_input("Fiber amount (kg/m³)", 25, 160, 80, 5, help="Range in data: 25 – 160 kg/m³")

    with tab_cure:
        st.caption("Curing regime")
        c1, c2 = st.columns(2)
        cure_temp = c1.number_input("Curing temperature (°C)", 20, 210, 20, 1, help="Range in data: 20 – 210 °C")
        cure_days = c2.selectbox("Curing period (days)", CURING_PERIODS, index=CURING_PERIODS.index(28))
        custom_days = c2.checkbox("Enter a custom curing period")
        if custom_days:
            cure_days = c2.number_input("Custom curing period (days)", 1, 365, int(cure_days), 1)


def build_input() -> pd.DataFrame:
    """Map GUI values to the exact training columns (duplicated columns filled identically)."""
    values = {
        "Cement (kg/m³)": cement,
        "Blast furnace slag (kg/m³)": slag,
        "Silica fume (kg/m³)": silica,
        "Cement (kg/m³).1": cement,
        "Blast furnace slag (kg/m³).1": slag,
        "Silica fume (kg/m³).1": silica,
        "Limestone powder (kg/m³)": lime,
        "Quartz powder (kg/m³)": quartz,
        "Fly ash (kg/m³)": fly,
        "Nano Silica (kg/m³)": nano,
        "Aggregate (kg/m³)": agg,
        "Water (kg/m³)": water,
        "Steel Fiber Type": fiber_type,
        "Steel Fiber Aspect Ratio": aspect,
        "Fiber amount (kg/m³)": fiber_amt,
        "Superplasticizer (kg/m³)": sp,
        "Curing temperature  (°C)": cure_temp,
        "Curing period (Days)": cure_days,
    }
    return pd.DataFrame([values])[FEATURES]


def classify(f: float) -> str:
    if f < 100:
        return "Below typical UHPC range (< 100 MPa)"
    if f < 150:
        return "UHPC range (100 – 150 MPa)"
    return "High-end UHPC (> 150 MPa)"


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
with right:
    st.subheader("Prediction")
    X_in = build_input()
    predict = st.button("🔮 Predict compressive strength", type="primary", use_container_width=True)

    # Predict automatically on first load so the page is never empty
    if predict or "first_run" not in st.session_state:
        st.session_state["first_run"] = True
        st.session_state["pred"] = float(model.predict(X_in)[0])
        st.session_state["inputs"] = X_in

    pred = st.session_state["pred"]
    st.markdown(
        f"""
        <div class="result-card">
            <div class="unit">Predicted compressive strength</div>
            <div class="val">{pred:,.1f}</div>
            <div class="unit">MPa</div>
            <div class="cls">{classify(pred)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state["inputs"].iloc[0].tolist() != X_in.iloc[0].tolist():
        st.info("Inputs changed — click **Predict** to refresh the result.")

    # Quick mix summary
    binder_total = cement + slag + silica + fly + lime + quartz + nano
    m1, m2 = st.columns(2)
    m1.metric("Total binder (kg/m³)", f"{binder_total:,.0f}")
    m2.metric("Water / binder", f"{water / binder_total:.3f}" if binder_total else "–")
    m3, m4 = st.columns(2)
    m3.metric("Fiber volume (≈ %)", f"{fiber_amt / 78.5:.2f}", help="Assuming steel density 7850 kg/m³")
    m4.metric("SP / binder (%)", f"{100 * sp / binder_total:.2f}" if binder_total else "–")

    if "test_r2" in meta:
        st.caption(
            f"Model: **{meta.get('model_name', 'Gradient Boosting')}** · test R² = {meta['test_r2']:.3f}"
        )

    # Download result
    out = st.session_state["inputs"].copy()
    out["Predicted compressive strength (MPa)"] = round(pred, 2)
    st.download_button(
        "⬇️ Download this prediction (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        "uhpsfrc_prediction.csv",
        "text/csv",
        use_container_width=True,
    )

# ----------------------------------------------------------------------------
# Interpretation section
# ----------------------------------------------------------------------------
st.divider()
st.subheader("🔍 Model interpretation")
t1, t2, t3 = st.tabs(["Strength vs. curing age", "Sensitivity analysis", "Global feature importance"])

with t1:
    st.caption("Predicted strength for the current mix at different curing periods.")
    days = np.array(CURING_PERIODS)
    grid = pd.concat([X_in] * len(days), ignore_index=True)
    grid["Curing period (Days)"] = days
    curve = pd.DataFrame({"Curing period (days)": days, "Strength (MPa)": model.predict(grid)})
    st.line_chart(curve, x="Curing period (days)", y="Strength (MPa)")

with t2:
    st.caption("Change one variable at a time (others fixed at current values) and see the effect on strength.")
    sweep_options = {
        "Cement (kg/m³)": (270, 1250, 40),
        "Silica fume (kg/m³)": (0, 430, 20),
        "Water (kg/m³)": (90, 270, 10),
        "Superplasticizer (kg/m³)": (1, 57, 3),
        "Fiber amount (kg/m³)": (25, 160, 5),
        "Steel Fiber Aspect Ratio": (30, 90, 5),
        "Curing temperature  (°C)": (20, 210, 10),
        "Aggregate (kg/m³)": (410, 1990, 80),
    }
    var = st.selectbox("Variable to vary", list(sweep_options))
    lo, hi, step = sweep_options[var]
    xs = np.arange(lo, hi + step, step)
    g = pd.concat([X_in] * len(xs), ignore_index=True)
    g[var] = xs
    if var in ("Cement (kg/m³)", "Silica fume (kg/m³)"):  # keep duplicated training columns in sync
        g[var + ".1"] = xs
    sens = pd.DataFrame({var: xs, "Strength (MPa)": model.predict(g)})
    st.line_chart(sens, x=var, y="Strength (MPa)")

with t3:
    st.caption("Impurity-based importance from the Gradient Boosting model (one-hot fiber columns are summed).")
    try:
        gb = model.named_steps["model"]
        names = model.named_steps["preprocessor"].get_feature_names_out()
        imp = pd.Series(gb.feature_importances_, index=names)
        clean = {}
        for n, v in imp.items():
            n = n.split("__", 1)[-1]
            if n.startswith("Steel Fiber Type_"):
                n = "Steel Fiber Type"
            n = n.replace(".1", "")           # merge the duplicated binder columns
            clean[n] = clean.get(n, 0) + v
        imp_df = (
            pd.Series(clean, name="Importance").sort_values(ascending=True).rename_axis("Feature").reset_index()
        )
        st.bar_chart(imp_df, x="Feature", y="Importance", horizontal=True)
    except Exception as e:  # pragma: no cover
        st.warning(f"Feature importance unavailable: {e}")

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
with st.expander("ℹ️ About & limitations"):
    st.markdown(
        """
        - Model: Gradient Boosting Regressor trained on a literature-compiled UHPSFRC database (662 mixes).
        - Predictions are only reliable **inside the ranges of the training data** (shown in each input's tooltip).
        - Results are for research and preliminary mix design; always confirm with laboratory tests.
        """
    )
st.caption("© UHPSFRC Strength Predictor · built with Streamlit")
