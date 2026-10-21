"""
Predictive Maintenance for a Factory Motor
AI for IoT - Capstone (Experiment 8)

Pipeline:
Simulated sensors -> Preprocessing -> Prediction (Linear Regression)
-> Anomaly detection (Isolation Forest + Z-score) -> Plots / Dashboard
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import MinMaxScaler

SENSORS = ["vibration", "temperature", "current"]


# ---------------------------------------------------------------- 1. SIMULATION
def simulate_sensors(n=5000, fault_start=4000, seed=42):
    """Simulate 3 motor sensors (1 reading per minute).

    Healthy motor: stable values + periodic load cycle + Gaussian noise.
    After `fault_start`: bearing wear -> vibration, temperature and current
    drift upward (impending failure). A few random spikes are added as well.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    load = 0.5 + 0.5 * np.sin(2 * np.pi * t / 480)           # 8-hour load cycle

    vibration = 2.0 + 0.4 * load + rng.normal(0, 0.10, n)     # mm/s
    temperature = 60 + 8 * load + rng.normal(0, 0.60, n)      # deg C
    current = 10 + 3 * load + rng.normal(0, 0.25, n)          # Amps

    # Gradual degradation (grows faster as failure gets closer)
    wear = np.zeros(n)
    k = np.arange(n - fault_start)
    wear[fault_start:] = (k / k.max()) ** 1.5
    vibration += 3.0 * wear
    temperature += 18 * wear
    current += 2.5 * wear

    # Random sensor spikes in the healthy period (short transient faults)
    spike_idx = rng.choice(np.arange(500, fault_start - 100), size=12, replace=False)
    vibration[spike_idx] += rng.uniform(1.2, 2.0, size=12)
    temperature[spike_idx] += rng.uniform(5, 9, size=12)

    # Ground truth label: spikes, or the point where wear is clearly visible
    label = np.zeros(n, dtype=int)
    label[spike_idx] = 1
    label[wear > 0.20] = 1

    return pd.DataFrame({"time": t, "vibration": vibration,
                         "temperature": temperature, "current": current,
                         "is_anomaly": label})


# -------------------------------------------------------------- 2. PREPROCESSING
def preprocess(df, window=30):
    """Fill missing values, add rolling features, scale to [0, 1]."""
    df = df.copy()
    df[SENSORS] = df[SENSORS].interpolate().bfill()
    for s in SENSORS:
        df[f"{s}_rmean"] = df[s].rolling(window, min_periods=1).mean()
        df[f"{s}_rstd"] = df[s].rolling(window, min_periods=1).std().fillna(0)
    return df


# ------------------------------------------------------------------ 3. PREDICTION
def build_lag_features(df, lags=5, horizon=1):
    """Use the last `lags` readings of all sensors to predict the next vibration."""
    X = pd.DataFrame(index=df.index)
    for s in SENSORS:
        for L in range(lags):
            X[f"{s}_lag{L}"] = df[s].shift(L)
    X["target"] = df["vibration"].shift(-horizon)
    X["time"] = df["time"]
    X["is_anomaly"] = df["is_anomaly"]
    return X.dropna().reset_index(drop=True)


def train_predictor(df, train_end=3000, lags=5):
    data = build_lag_features(df, lags)
    # Autoregressive model on the vibration history only. Temperature/current
    # mostly track the load cycle, so a model trained on healthy data would use
    # them as load proxies and fail once the motor degrades.
    feats = [c for c in data.columns if c.startswith("vibration_lag")]
    train = data[data["time"] < train_end]
    test = data[data["time"] >= train_end]

    scaler = MinMaxScaler().fit(train[feats])
    model = LinearRegression().fit(scaler.transform(train[feats]), train["target"])
    pred = model.predict(scaler.transform(test[feats]))

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(test["target"], pred))),
        "MAE": float(mean_absolute_error(test["target"], pred)),
        "R2": float(r2_score(test["target"], pred)),
    }
    out = test[["time", "target", "is_anomaly"]].copy()
    out["predicted"] = pred
    return model, out, metrics


# ------------------------------------------------------------- 4. ANOMALY DETECTION
def detect_anomalies(df, train_end=3000, contamination=0.02, z_thresh=3.0, seed=42):
    feats = SENSORS + [f"{s}_rmean" for s in SENSORS] + [f"{s}_rstd" for s in SENSORS]
    train = df[df["time"] < train_end]

    # Isolation Forest (trained on healthy data only)
    scaler = MinMaxScaler().fit(train[feats])
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=seed).fit(scaler.transform(train[feats]))
    df = df.copy()
    df["iso_flag"] = (iso.predict(scaler.transform(df[feats])) == -1).astype(int)

    # Z-score on vibration (mean/std from healthy data)
    mu, sd = train["vibration"].mean(), train["vibration"].std()
    df["z_vibration"] = (df["vibration"] - mu) / sd
    df["z_flag"] = (df["z_vibration"].abs() > z_thresh).astype(int)

    test = df[df["time"] >= train_end]
    metrics = {}
    for name, col in [("IsolationForest", "iso_flag"), ("Z-score", "z_flag")]:
        metrics[name] = {
            "precision": float(precision_score(test["is_anomaly"], test[col], zero_division=0)),
            "recall": float(recall_score(test["is_anomaly"], test[col], zero_division=0)),
            "f1": float(f1_score(test["is_anomaly"], test[col], zero_division=0)),
        }
    return df, metrics


# ------------------------------------------------------------------- 5. PLOTTING
def plot_sensors(df, path=None):
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    units = {"vibration": "mm/s", "temperature": "°C", "current": "A"}
    for ax, s in zip(axes, SENSORS):
        ax.plot(df["time"], df[s], lw=0.6, color="tab:blue", label=s)
        flagged = df[df["iso_flag"] == 1]
        ax.scatter(flagged["time"], flagged[s], s=6, color="red", label="anomaly")
        ax.set_ylabel(f"{s} ({units[s]})")
        ax.legend(loc="upper left")
    axes[0].set_title("Simulated motor sensors with detected anomalies (Isolation Forest)")
    axes[-1].set_xlabel("Time (minutes)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_prediction(pred_df, path=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(pred_df["time"], pred_df["target"], lw=0.8, label="Actual vibration")
    ax.plot(pred_df["time"], pred_df["predicted"], lw=0.8, ls="--", label="Predicted vibration")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Vibration (mm/s)")
    ax.set_title("Linear Regression: predicted vs actual vibration (test set)")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_zscore(df, z_thresh=3.0, path=None):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["time"], df["z_vibration"], lw=0.6, color="tab:purple")
    ax.axhline(z_thresh, color="red", ls="--", label=f"±{z_thresh} threshold")
    ax.axhline(-z_thresh, color="red", ls="--")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Z-score (vibration)")
    ax.set_title("Z-score anomaly detection on vibration")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


# --------------------------------------------------------------------- RUN ALL
def run(n=5000, fault_start=4000, train_end=3000, contamination=0.02,
        z_thresh=3.0, save_dir=None):
    raw = simulate_sensors(n, fault_start)
    df = preprocess(raw)
    model, pred_df, pred_metrics = train_predictor(df, train_end)
    df, anom_metrics = detect_anomalies(df, train_end, contamination, z_thresh)

    figs = {
        "sensors": plot_sensors(df, f"{save_dir}/sensors_anomalies.png" if save_dir else None),
        "prediction": plot_prediction(pred_df, f"{save_dir}/prediction.png" if save_dir else None),
        "zscore": plot_zscore(df, z_thresh, f"{save_dir}/zscore.png" if save_dir else None),
    }
    return {"data": df, "pred": pred_df, "pred_metrics": pred_metrics,
            "anom_metrics": anom_metrics, "figs": figs}


if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)
    res = run(save_dir="outputs")
    res["data"].to_csv("outputs/motor_data_with_flags.csv", index=False)

    print("\n=== Prediction (Linear Regression) ===")
    for k, v in res["pred_metrics"].items():
        print(f"{k}: {v:.4f}")
    print("\n=== Anomaly detection ===")
    for name, m in res["anom_metrics"].items():
        print(f"{name}: precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}")
    print("\nPlots saved in ./outputs")
