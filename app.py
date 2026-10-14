"""
UHPSFRC Compressive Strength Predictor
======================================
Streamlit GUI for the paper:
"Interpretable Machine Learning Framework for Compressive Strength Prediction of
 Ultra-High Performance Steel Fiber Reinforced Concrete (UHPSFRC) with Practical GUI Deployment"

Files expected in the same GitHub folder:
    app.py
    best_model.pkl     <- trained Gradient Boosting pipeline
    best_model.json    <- model metadata (optional, used for the R² line)
    requirements.txt
"""
import json
import pickle
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Page setup & styling
# ----------------------------------------------------------------------------
st.set_page_config(page_title="UHPSFRC Strength Predictor", page_icon="🏗️", layout="wide")

PRIMARY = "#1d5a8a"
ACCENT = "#e4572e"

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 1250px;}
        /* hide the Fork and GitHub buttons in the top-right toolbar */
        [data-testid="stToolbarActions"],
        [data-testid="stAppDeployButton"],
        .stAppDeployButton {display: none !important;}
        .hero {
            background: linear-gradient(120deg, #0f2a43 0%, #1d5a8a 100%);
            padding: 1.6rem 2rem; border-radius: 14px; color: #fff; margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0; font-size: 1.9rem; color: #fff;}
        .hero p  {margin: .4rem 0 0 0; opacity: .9; font-size: .95rem;}
        .result-card {
            background: linear-gradient(135deg, #0f2a43, #1d5a8a);
            border-radius: 14px; padding: 1.6rem; text-align: center; color: #fff;
            margin-bottom: .8rem;
        }
        .result-card .val  {font-size: 3.2rem; font-weight: 700; line-height: 1.1;}
        .result-card .unit {font-size: 1.1rem; opacity: .85;}
        .result-card .cls  {margin-top: .6rem; font-size: 1rem; font-weight: 600;}
        div[data-testid="stMetric"] {
            background: rgba(120,140,160,.10); border-radius: 10px; padding: .6rem .9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Locate & load the model (works even if the file was renamed or put in a subfolder)
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent


def find_file(name: str):
    exact = BASE_DIR / name
    if exact.exists():
        return exact
    ext = Path(name).suffix
    found = sorted(BASE_DIR.rglob(f"*{ext}"))
    found = [f for f in found if "best_model" in f.name.lower()] or found
    return found[0] if found else None


MODEL_PATH = find_file("best_model.pkl")
META_PATH = find_file("best_model.json")


@st.cache_resource(show_spinner="Loading model…")
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta(path):
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


if MODEL_PATH is None:
    st.error(
        "Model file not found. Upload **best_model.pkl** to the same GitHub folder as app.py "
        "(files found here: " + ", ".join(sorted(p.name for p in BASE_DIR.iterdir())) + ")."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:  # e.g. scikit-learn version mismatch
    st.error(
        "The model could not be loaded. It was saved with **scikit-learn 1.6.1**, so make sure "
        f"`requirements.txt` pins that version.\n\nDetails: `{e}`"
    )
    st.stop()

meta = load_meta(META_PATH)

# Exact column names / order used in training.
# The training sheet has Cement, Blast furnace slag and Silica fume twice (".1" copies with
# identical values), so the GUI asks for each once and feeds both columns.
FEATURES = list(model.feature_names_in_)

FIBER_TYPES = [
    "Straight Steel Fiber",
    "Hooked-End Steel Fiber",
    "Crimped Steel Fiber",
    "Twisted (Torex) Steel Fiber",
    "Mix (Straight + Hooked-End)",
    "Mix (Straight + Twisted)",
]
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
# Inputs (every field starts with a default value)
# ----------------------------------------------------------------------------
left, right = st.columns([1.55, 1], gap="large")

with left:
    tab_bind, tab_agg, tab_fiber, tab_cure = st.tabs(
        ["🧱 Binders", "💧 Aggregate, Water & Admixture", "🧵 Steel Fiber", "🌡️ Curing"]
    )

    with tab_bind:
        st.caption("Binders: cementitious materials and powders (kg/m³)")
        c1, c2 = st.columns(2)
        cement = c1.number_input("Cement (kg/m³)", 270.0, 1260.0, 720.0, 10.0, help="Training range: 270 – 1251")
        slag = c2.number_input("Blast furnace slag (kg/m³)", 0.0, 400.0, 0.0, 5.0, help="Training range: 0 – 375")
        silica = c1.number_input("Silica fume (kg/m³)", 0.0, 450.0, 120.0, 5.0, help="Training range: 0 – 434")
        fly = c2.number_input("Fly ash (kg/m³)", 0.0, 360.0, 0.0, 5.0, help="Training range: 0 – 356")
        lime = c1.number_input("Limestone powder (kg/m³)", 0.0, 1100.0, 0.0, 10.0, help="Training range: 0 – 1058")
        quartz = c2.number_input("Quartz powder (kg/m³)", 0.0, 400.0, 0.0, 5.0, help="Training range: 0 – 397")
        nano = c1.number_input("Nano silica (kg/m³)", 0.0, 50.0, 0.0, 0.5, help="Training range: 0 – 47.5")

    with tab_agg:
        st.caption("Aggregate, mixing water and superplasticizer (kg/m³)")
        c1, c2, c3 = st.columns(3)
        agg = c1.number_input("Aggregate (kg/m³)", 400.0, 2000.0, 1100.0, 10.0, help="Training range: 408 – 1992")
        water = c2.number_input("Water (kg/m³)", 90.0, 275.0, 160.0, 1.0, help="Training range: 90 – 272.6")
        sp = c3.number_input("Superplasticizer (kg/m³)", 1.0, 60.0, 25.0, 0.5, help="Training range: 1.1 – 57")

    with tab_fiber:
        st.caption("Steel fiber characteristics")
        fiber_type = st.selectbox("Steel fiber type", FIBER_TYPES, index=0)
        c1, c2 = st.columns(2)
        aspect = c1.number_input("Aspect ratio, L/d (–)", 30, 90, 65, 1, help="Training range: 30 – 90")
        fiber_amt = c2.number_input("Fiber amount (kg/m³)", 25, 160, 80, 5, help="Training range: 25 – 160")

    with tab_cure:
        st.caption("Curing regime")
        c1, c2 = st.columns(2)
        cure_temp = c1.number_input("Curing temperature (°C)", 20, 210, 20, 1, help="Training range: 20 – 210")
        cure_days = c2.selectbox("Curing period (days)", CURING_PERIODS, index=CURING_PERIODS.index(28))
        if c2.checkbox("Enter a custom curing period"):
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
# Prediction panel
# ----------------------------------------------------------------------------
X_in = build_input()

with right:
    st.subheader("Prediction")
    clicked = st.button("🔮 Predict compressive strength", type="primary", width="stretch")

    # Predict on first load (page never empty) and whenever the button is clicked
    if clicked or "pred" not in st.session_state:
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
    if not st.session_state["inputs"].equals(X_in):
        st.info("Inputs changed — click **Predict** to refresh the result.")

    binder_total = cement + slag + silica + fly + lime + quartz + nano
    a, b = st.columns(2)
    a.metric("Total binder (kg/m³)", f"{binder_total:,.0f}")
    b.metric("Water / binder", f"{water / binder_total:.3f}")
    c, d = st.columns(2)
    c.metric("Fiber volume (%)", f"{fiber_amt / 78.5:.2f}", help="Fiber amount ÷ steel density (7850 kg/m³)")
    d.metric("SP / binder (%)", f"{100 * sp / binder_total:.2f}")



    out = st.session_state["inputs"].copy()
    out["Predicted compressive strength (MPa)"] = round(pred, 2)
    st.download_button(
        "⬇️ Download this prediction (CSV)",
        out.to_csv(index=False).encode("utf-8"),
        "uhpsfrc_prediction.csv",
        "text/csv",
        width="stretch",
    )

# ----------------------------------------------------------------------------
# Interpretation
# ----------------------------------------------------------------------------
def line_plot(df: pd.DataFrame, x: str, current_x=None, y: str = "Strength (MPa)"):
    """Line chart with a zoomed y-axis and a marker for the currently entered value."""
    base = alt.Chart(df).encode(
        x=alt.X(x, title=x),
        y=alt.Y(y, scale=alt.Scale(zero=False), title=y),
        tooltip=[alt.Tooltip(x, format=".4~g"), alt.Tooltip(y, format=".1f")],
    )
    chart = base.mark_line(point=True, color=PRIMARY)
    if current_x is not None:
        marker = (
            alt.Chart(pd.DataFrame({x: [current_x]}))
            .mark_rule(color=ACCENT, strokeDash=[5, 4], size=2)
            .encode(x=alt.X(x, title=x))
        )
        chart = chart + marker
    st.altair_chart(chart.properties(height=360), width="stretch")


st.divider()
st.subheader("🔍 Model interpretation")
tab_age, tab_sens = st.tabs(["Strength vs. curing age", "Sensitivity analysis"])

with tab_age:
    st.caption("Predicted strength of the current mix at different curing periods.")
    days = np.array(CURING_PERIODS)
    grid = pd.concat([X_in] * len(days), ignore_index=True)
    grid["Curing period (Days)"] = days
    curve = pd.DataFrame({"Curing period (days)": days, "Strength (MPa)": model.predict(grid)})
    line_plot(curve, "Curing period (days)", current_x=cure_days)
    st.caption("Dashed line = your current curing period.")

with tab_sens:
    st.caption(
        "Pick a variable. Every other input stays fixed at the value you entered, so the curve shows "
        "the effect of that one variable on strength."
    )
    # label with unit -> (training column(s), min, max, step, current GUI value)
    SWEEP = {
        "Cement (kg/m³)": (["Cement (kg/m³)", "Cement (kg/m³).1"], 270, 1250, 20, cement),
        "Blast furnace slag (kg/m³)": (["Blast furnace slag (kg/m³)", "Blast furnace slag (kg/m³).1"], 0, 375, 15, slag),
        "Silica fume (kg/m³)": (["Silica fume (kg/m³)", "Silica fume (kg/m³).1"], 0, 430, 15, silica),
        "Fly ash (kg/m³)": (["Fly ash (kg/m³)"], 0, 356, 15, fly),
        "Limestone powder (kg/m³)": (["Limestone powder (kg/m³)"], 0, 1060, 40, lime),
        "Quartz powder (kg/m³)": (["Quartz powder (kg/m³)"], 0, 400, 15, quartz),
        "Nano silica (kg/m³)": (["Nano Silica (kg/m³)"], 0, 47.5, 2.5, nano),
        "Aggregate (kg/m³)": (["Aggregate (kg/m³)"], 410, 1990, 60, agg),
        "Water (kg/m³)": (["Water (kg/m³)"], 90, 270, 6, water),
        "Superplasticizer (kg/m³)": (["Superplasticizer (kg/m³)"], 1, 57, 2, sp),
        "Steel fiber amount (kg/m³)": (["Fiber amount (kg/m³)"], 25, 160, 5, fiber_amt),
        "Steel fiber aspect ratio (–)": (["Steel Fiber Aspect Ratio"], 30, 90, 2, aspect),
        "Curing temperature (°C)": (["Curing temperature  (°C)"], 20, 210, 10, cure_temp),
        "Curing period (days)": (["Curing period (Days)"], None, None, None, cure_days),
        "Steel fiber type (category)": (["Steel Fiber Type"], None, None, None, fiber_type),
    }
    st.markdown("**Variable to vary**")
    var = (
        st.pills(
            "Variable to vary",
            list(SWEEP),
            default="Cement (kg/m³)",
            selection_mode="single",
            key="sweep_var",
            label_visibility="collapsed",
        )
        or "Cement (kg/m³)"  # pills can be un-clicked; fall back to the default
    )
    cols, lo, hi, step, current = SWEEP[var]

    if var == "Steel fiber type (category)":
        g = pd.concat([X_in] * len(FIBER_TYPES), ignore_index=True)
        g["Steel Fiber Type"] = FIBER_TYPES
        sens = pd.DataFrame({"Steel fiber type": FIBER_TYPES, "Strength (MPa)": model.predict(g)})
        sens["Current"] = np.where(sens["Steel fiber type"] == current, "Current selection", "Other")
        bars = (
            alt.Chart(sens)
            .mark_bar()
            .encode(
                y=alt.Y("Steel fiber type", sort="-x", title="Steel fiber type (category)",
                        axis=alt.Axis(labelLimit=300)),
                x=alt.X("Strength (MPa)", title="Compressive strength (MPa)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Current",
                    scale=alt.Scale(domain=["Current selection", "Other"], range=[ACCENT, PRIMARY]),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                tooltip=["Steel fiber type", alt.Tooltip("Strength (MPa)", format=".1f")],
            )
            .properties(height=300)
        )
        st.altair_chart(bars, width="stretch")
    else:
        xs = np.array(CURING_PERIODS) if var == "Curing period (days)" else np.arange(lo, hi + step / 2, step)
        g = pd.concat([X_in] * len(xs), ignore_index=True)
        for col in cols:  # duplicated training columns stay in sync
            g[col] = xs
        sens = pd.DataFrame({var: xs, "Strength (MPa)": model.predict(g)})
        line_plot(sens, var, current_x=current)
        st.caption(f"Dashed line = your current value ({current:g}). Range shown follows the training data.")

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
with st.expander("ℹ️ About & limitations"):
    st.markdown(
        """
        - Model: Gradient Boosting Regressor trained on a literature-compiled UHPSFRC database.
        - Predictions are reliable only **inside the ranges of the training data** (shown in each input's tooltip).
        - Results are intended for research and preliminary mix design; confirm with laboratory tests.
        """
    )
st.markdown(
    "<div style='text-align:center; opacity:.75; font-size:.9rem; padding-top:.5rem;'>"
    "Created by <b>Naresh Bhatta</b><br>Kathmandu University, Nepal"
    "</div>",
    unsafe_allow_html=True,
)
