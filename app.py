import os, json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH    = "models/model.joblib"
META_PATH     = "models/metadata.json"
STATS_PATH    = "models/train_stats.json"
PRED_LOG_PATH = "logs/predictions.csv"
FEATURES = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
CLASSES  = ["setosa", "versicolor", "virginica"]
LOG_COLS = ["timestamp"] + FEATURES + ["prediction", "confidence", "model_version"]

st.set_page_config(page_title="Iris MLOps Demo", page_icon="🌸", layout="wide")


@st.cache_resource
def load_model(mtime):
    """mtime is part of the cache key, so a redeployed file reloads automatically."""
    return joblib.load(MODEL_PATH)


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}


def log_prediction(row):
    os.makedirs("logs", exist_ok=True)
    pd.DataFrame([row], columns=LOG_COLS).to_csv(
        PRED_LOG_PATH, mode="a", header=not os.path.exists(PRED_LOG_PATH), index=False
    )


if not os.path.exists(MODEL_PATH):
    st.error("No model deployed yet. Run the deployment cell in the notebook.")
    st.stop()

model = load_model(os.path.getmtime(MODEL_PATH))
meta  = load_json(META_PATH)
stats = load_json(STATS_PATH)

st.title("🌸 Iris Classifier")
st.caption(f"Serving model **v{meta.get('version', '?')}** ({meta.get('run_name', '')}) "
           f"| deployed {meta.get('deployed_at', '?')}")

tab_predict, tab_monitor, tab_info = st.tabs(["🔮 Predict", "📈 Monitoring", "ℹ️ Model info"])

# ------------------------------------------------------------------ Predict
with tab_predict:
    left, right = st.columns(2)
    with left:
        st.subheader("Flower measurements")
        sl = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
        sw = st.slider("Sepal width (cm)",  2.0, 4.5, 3.0, 0.1)
        pl = st.slider("Petal length (cm)", 1.0, 7.0, 4.3, 0.1)
        pw = st.slider("Petal width (cm)",  0.1, 2.5, 1.3, 0.1)

    x = pd.DataFrame([[sl, sw, pl, pw]], columns=FEATURES)
    proba = model.predict_proba(x)[0]
    pred = int(np.argmax(proba))

    with right:
        st.subheader("Prediction")
        st.success(f"**{CLASSES[pred]}**  ({proba[pred]:.1%} confident)")
        st.bar_chart(pd.DataFrame({"probability": proba}, index=CLASSES))
        if st.button("📝 Log this prediction"):
            log_prediction([datetime.now().isoformat(timespec="seconds"), sl, sw, pl, pw,
                            CLASSES[pred], round(float(proba[pred]), 4), meta.get("version", 0)])
            st.toast("Logged!")

# --------------------------------------------------------------- Monitoring
with tab_monitor:
    st.subheader("Production monitoring")
    if not os.path.exists(PRED_LOG_PATH):
        st.info("No predictions logged yet. Use the Predict tab and click 'Log this prediction'.")
    else:
        logs = pd.read_csv(PRED_LOG_PATH)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total predictions", len(logs))
        c2.metric("Avg confidence", f"{logs['confidence'].mean():.1%}")
        c3.metric("Latest model version", int(logs["model_version"].iloc[-1]))

        st.markdown("**Predictions by class**")
        st.bar_chart(logs["prediction"].value_counts())

        st.markdown("**Confidence over time** (rolling mean of 10)")
        st.line_chart(logs["confidence"].rolling(10, min_periods=1).mean())

        st.markdown("**Feature drift** (z-score of recent mean vs. training mean)")
        window = st.slider("Use last N predictions", 10, max(len(logs), 11), min(len(logs), 50))
        recent = logs.tail(window)
        z = pd.Series({f: (recent[f].mean() - stats[f]["mean"]) / stats[f]["std"] for f in FEATURES})
        st.bar_chart(z.rename("z-score"))
        drifted = z[z.abs() > 1.0]
        if len(drifted):
            st.warning("Possible data drift in: " + ", ".join(drifted.index) +
                       ". Consider investigating and retraining.")
        else:
            st.success("No significant drift detected.")

        with st.expander("Raw log"):
            st.dataframe(logs.tail(50), use_container_width=True)

# --------------------------------------------------------------- Model info
with tab_info:
    st.subheader("Deployed model")
    st.json(meta)
