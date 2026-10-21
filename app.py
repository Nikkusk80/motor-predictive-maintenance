"""Streamlit dashboard (bonus). Run with:  streamlit run app.py"""
import streamlit as st
import pandas as pd
import pipeline

st.set_page_config(page_title="Motor Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance for Factory Motor")
st.caption("AI for IoT Capstone: simulated sensors, Linear Regression, Isolation Forest, Z-score")

with st.sidebar:
    st.header("Simulation settings")
    n = st.slider("Number of readings", 3000, 8000, 5000, 500)
    fault_start = st.slider("Fault starts at (minute)", 3200, n - 500, 4000, 100)
    contamination = st.slider("Isolation Forest contamination", 0.005, 0.10, 0.02, 0.005)
    z_thresh = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.5)

res = pipeline.run(n=n, fault_start=fault_start, train_end=3000,
                   contamination=contamination, z_thresh=z_thresh)

st.subheader("Model metrics")
c1, c2, c3 = st.columns(3)
c1.metric("Prediction RMSE", f"{res['pred_metrics']['RMSE']:.3f}")
c2.metric("Prediction R²", f"{res['pred_metrics']['R2']:.3f}")
c3.metric("Z-score F1", f"{res['anom_metrics']['Z-score']['f1']:.3f}")
st.dataframe(pd.DataFrame(res["anom_metrics"]).T.round(3))

st.subheader("Sensors and detected anomalies")
st.pyplot(res["figs"]["sensors"])
col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Predicted vs actual")
    st.pyplot(res["figs"]["prediction"])
with col_b:
    st.subheader("Z-score view")
    st.pyplot(res["figs"]["zscore"])

df = res["data"]
alarm = df[(df["time"] >= 3000) & (df["z_flag"] == 1)]
if len(alarm):
    st.error(f"Maintenance alert: first anomaly flagged at minute {int(alarm['time'].iloc[0])}")
else:
    st.success("Motor healthy: no anomalies flagged")
