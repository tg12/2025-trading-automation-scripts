"""Copyright (C) 2025 James Sawyer
All rights reserved.

This script and the associated files are private
and confidential property. Unauthorized copying of
this file, via any medium, and the divulgence of any
contained information without express written consent
is strictly prohibited.

This script is intended for personal use only and should
not be distributed or used in any commercial or public
setting unless otherwise authorized by the copyright holder.
By using this script, you agree to abide by these terms.

DISCLAIMER: This script is provided 'as is' without warranty
of any kind, either express or implied, including, but not
limited to, the implied warranties of merchantability,
fitness for a particular purpose, or non-infringement. In no
event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action
of contract, tort or otherwise, arising from, out of, or in
connection with the script or the use or other dealings in
the script.
"""

# -*- coding: utf-8 -*-
# pylint: disable=C0116, W0621, W1203, C0103, C0301, W1201, W0511, E0401, E1101, E0606

# C0116: Missing function or method docstring
# W0621: Redefining name %r from outer scope (line %s)
# W1203: Use % formatting in logging functions and pass the % parameters as arguments
# C0103: Constant name "%s" doesn't conform to UPPER_CASE naming style
# C0301: Line too long (%s/%s)
# W1201: Specify string format arguments as logging function parameters
# W0511: TODOs
# E0606: possibly-used-before-assignment, ignore this

# Author : James Sawyer
# Maintainer : James Sawyer
# Version : 0.9a
# Status : BETA
# Copyright (C) 2025 James Sawyer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

csv_path = "./backtest_prices.csv"


def load_ohlc_csv(filepath: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(filepath)
        required_cols = {"open", "high", "low", "close"}
        if not required_cols.issubset({col.lower() for col in df.columns}):
            return None
        df.columns = [col.lower() for col in df.columns]
        return df
    except Exception:
        return None


def regime_hmm(ohlc: pd.DataFrame, n_states=3):
    ret = np.log(ohlc["close"]).diff().dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=100)
    model.fit(ret)
    regime = model.predict(ret)
    return pd.Series(regime, index=ohlc.index[-len(regime) :]), model


def detect_dist_shift(ohlc: pd.DataFrame, window=60):
    ret = np.log(ohlc["close"]).diff().dropna()
    shifts = []
    for i in range(window, len(ret)):
        w1 = ret.iloc[i - window : i]
        w2 = ret.iloc[i - window + 1 : i + 1]
        ks = ks_2samp(w1, w2).statistic
        wd = wasserstein_distance(w1, w2)
        shifts.append((ret.index[i], ks, wd))
    return pd.DataFrame(shifts, columns=["time", "ks", "wd"]).set_index("time")


# Load and validate data
ohlc_data = load_ohlc_csv(csv_path)
if ohlc_data is None or len(ohlc_data) < 100:
    raise ValueError("Data not sufficient or failed to load")

# Prepare return series
returns = np.log(ohlc_data["close"]).diff().dropna()
ohlc_data = ohlc_data.iloc[1:].copy()
ohlc_data["returns"] = returns

# Apply HMM to identify regimes
regime_labels, hmm_model = regime_hmm(ohlc_data, n_states=2)
ohlc_data["regime"] = regime_labels.astype(int)

# Calculate distributional shifts
shift_signals = detect_dist_shift(ohlc_data)
ohlc_data = ohlc_data.join(shift_signals, how="left")

# Calculate forward return
ohlc_data["forward_return"] = ohlc_data["close"].shift(-5) / ohlc_data["close"] - 1
ohlc_data["target"] = (ohlc_data["forward_return"] > 0).astype(int)

# Train classifier for predictive value
train_data = ohlc_data.dropna(subset=["ks", "wd", "regime", "target"])
X = train_data[["regime", "ks", "wd"]]
y = train_data["target"]
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)
y_pred = clf.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_pred)
print(f"Forward Return Prediction AUC: {auc:.3f}")

# Signal: high probability of positive forward return
ohlc_data["signal"] = 0
ohlc_data.loc[train_data.index, "signal"] = (y_pred > 0.6).astype(int)

# Visualize
plt.figure(figsize=(18, 6))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
for regime in sorted(ohlc_data["regime"].dropna().unique()):
    subset = ohlc_data[ohlc_data["regime"] == regime]
    plt.plot(subset.index, subset["close"], color=colors[int(regime)], label=f"Regime {int(regime)}")

ks_threshold = ohlc_data["ks"].quantile(0.95)
spike_points = ohlc_data[ohlc_data["ks"] > ks_threshold]
plt.scatter(spike_points.index, spike_points["close"], color="black", s=20, label="KS Spike")

# Overlay forward return signal
signal_points = ohlc_data[ohlc_data["signal"] == 1]
plt.scatter(signal_points.index, signal_points["close"], color="green", s=30, marker="^", label="Forward Return Signal")

plt.title("Market Regimes, Dist. Shift & Forward Return Signal")
plt.ylabel("Close Price")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()
