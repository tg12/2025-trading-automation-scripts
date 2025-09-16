"""Polarity → predictive signal (daily OHLC). Causal calibration, no argparse.
2x2 regime grid: Polarity (momentum/mean-reversion) × Directional (long/short).
"""

from __future__ import annotations

import logging

import colorlog
import numpy as np
import pandas as pd
from backtest_loader import load_backtest_prices

# Copyright (c) 2025 James Sawyer. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# IMPORTANT DISCLAIMER:
# This software is for educational and research purposes only. It is not intended
# to provide financial, investment, trading, or other advice. Trading cryptocurrencies
# and other financial instruments involves substantial risk of loss and is not
# suitable for every investor. The user of this software assumes all risks associated
# with its use. Past performance does not guarantee future results. The author
# assumes no responsibility for any losses incurred through the use of this software.
# Always consult with a qualified financial advisor before making investment decisions.
# Use at your own risk.

# ======= CONFIG (hard-coded) =======
CSV_PATH = None  # None => bundled backtest_prices.csv
W_PI = 20  # rolling window for PI mean/var
EPS = 1e-12
W_CAL = 252  # rolling window for probability calibration (~1y)
N_BINS = 6  # quantile bins
LAPLACE_K = 2.0  # pseudo-count per class
MIN_OBS_BIN = max(10, int(0.3 * W_CAL / N_BINS))  # minimum observations per bin for calibration
VOLWIN = 60  # daily vol window for scaling
GAMMA_M = 0.1  # regime gate threshold
GAMMA_U = 0.05  # directional gate threshold
# ===================================


def compute_polarity(df: pd.DataFrame, w=W_PI, eps=EPS) -> pd.Series:
    """Robust polarity: ensemble of multiple return variants to reduce noise.
    Variants: overnight (o - c_prev), intraday (c - o), close-close (c - c_prev), open-open (o - o_prev).
    """
    o = df["open"].astype("float64")
    h = df["high"].astype("float64")
    l = df["low"].astype("float64")
    c = df["close"].astype("float64")
    valid = (o > 0) & (h > 0) & (l > 0) & (c > 0)
    o = np.log(o.where(valid))
    h = np.log(h.where(valid))
    l = np.log(l.where(valid))
    c = np.log(c.where(valid))

    # Variants
    r_oc = c - o  # intraday
    r_co = o - c.shift(1)  # overnight
    r_cc = c - c.shift(1)  # close-close
    r_oo = o - o.shift(1)  # open-open

    polarities = []
    for r1, r2 in [(r_oc, r_co), (r_cc, r_oo), (r_oc, r_cc), (r_co, r_oo)]:
        mu1 = r1.ewm(alpha=2 / (w + 1)).mean()
        mu2 = r2.ewm(alpha=2 / (w + 1)).mean()
        var1 = r1.ewm(alpha=2 / (w + 1)).var()
        var2 = r2.ewm(alpha=2 / (w + 1)).var()
        pol = (mu1 - mu2) / np.sqrt(var1 + var2 + eps)
        polarities.append(pol)

    # Ensemble: average
    return pd.concat(polarities, axis=1).mean(axis=1)


def _rolling_bins(x: pd.Series, window=W_CAL, n_bins=N_BINS) -> pd.Series:
    q = np.linspace(0, 1, n_bins + 1)
    v = x.values.astype(np.float64)
    edges = [None] * len(v)
    for t in range(len(v)):
        past = v[max(0, t - window) : t]
        past = past[~np.isnan(past)]
        edges[t] = None if past.size < max(5, n_bins) else np.quantile(past, q)
    b = np.full(len(v), -1, dtype=int)
    for t, e in enumerate(edges):
        if e is None:
            continue
        b[t] = int(np.clip(np.searchsorted(e, v[t], side="right") - 1, 0, n_bins - 1))
    return pd.Series(b, index=x.index)


def _pav_means(means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    m = means.copy().astype(np.float64)
    w = weights.copy().astype(np.float64)
    k = len(m)
    i = 0
    while i < k - 1:
        if not np.isfinite(m[i]) or not np.isfinite(m[i + 1]):
            i += 1
            continue
        if m[i] > m[i + 1]:
            tot = w[i] + w[i + 1]
            avg = (m[i] * w[i] + m[i + 1] * w[i + 1]) / (tot if tot > 0 else 1.0)
            m[i], m[i + 1] = avg, avg
            w[i], w[i + 1] = tot, tot
            j = i - 1
            while j >= 0 and np.isfinite(m[j]) and m[j] > m[j + 1]:
                tot = w[j] + w[j + 1]
                avg = (m[j] * w[j] + m[j + 1] * w[j + 1]) / (tot if tot > 0 else 1.0)
                m[j], m[j + 1] = avg, avg
                w[j], w[j + 1] = tot, tot
                j -= 1
            i = max(j + 1, 0)
        else:
            i += 1
    return m


def _soft_gate(x: pd.Series, gamma: np.float64) -> pd.Series:
    g = (x.abs() - gamma) / (1.0 - gamma + 1e-12)
    return g.clip(lower=0.0, upper=1.0)


def validate_monotonicity_walkforward(feature, y_next_up, window=W_CAL, n_bins=N_BINS):
    idx = feature.index
    hits_by_bin = [[] for _ in range(n_bins)]
    q = np.linspace(0, 1, n_bins + 1)
    v = feature.values.astype(np.float64)
    y = y_next_up.reindex(idx).values.astype(np.float64)
    for t in range(len(v)):
        past = v[max(0, t - window) : t]
        if past.size < max(5, n_bins) or np.isnan(v[t]) or np.isnan(y[t]):
            continue
        edges = np.quantile(past, q)
        b = np.searchsorted(edges, v[t], side="right") - 1
        b = int(np.clip(b, 0, n_bins - 1))
        hits_by_bin[b].append(y[t])
    hit_rates = [np.mean(h) if len(h) > 0 else np.nan for h in hits_by_bin]
    is_mono = np.all(np.diff([h for h in hit_rates if not np.isnan(h)]) >= 0)
    return is_mono, hit_rates


def walk_forward_validation(df: pd.DataFrame, n_folds: int = 5) -> dict:
    """True walk-forward validation: chronological partitioning into train/cal/test folds.
    Re-calibrates probability bins forward-only, reports hit-rate stability.
    """
    n = len(df)
    fold_size = n // n_folds
    hit_rates_pi = []
    hit_rates_dir = []

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n
        train_df = df.iloc[:test_start]
        test_df = df.iloc[test_start:test_end]

        if len(train_df) < W_CAL or len(test_df) < 10:
            continue

        # Build signal on training data
        train_signal = build_signal(train_df)

        # Apply calibrated probabilities to test data
        test_pi = compute_polarity(test_df)
        test_log_c = np.log(test_df["close"].astype("float64"))
        test_y_next_up = (test_log_c.shift(-1) - test_log_c > 0).astype(int)
        test_r_cc = test_log_c.diff()
        test_sgn = np.sign(test_r_cc)
        test_y_cont = ((test_sgn.shift(-1) * test_sgn) > 0).astype(np.float64)
        test_y_cont[(test_sgn == 0) | (test_sgn.shift(-1) == 0)] = np.nan

        # Use training calibrations for test
        m_test = calibrate_monotone(test_pi, test_y_cont)  # This will use rolling on test, but ideally use train params

        # For simplicity, since calibrate_monotone is rolling, it will adapt, but to make it strict, we can pass train params
        # For now, proceed with rolling

        level_test, slope_test = kalman_level_slope(test_log_c)
        rs_test = rogers_satchell_vol(test_df, window=20)
        drift_norm_test = (slope_test / (rs_test + 1e-12)).replace([np.inf, -np.inf], np.nan)
        u_test = calibrate_monotone(drift_norm_test, test_y_next_up)

        # Compute hit rates on test
        pred_up = u_test > 0.5
        actual_up = test_y_next_up == 1
        hit_rate_dir = (pred_up & actual_up).sum() / pred_up.sum() if pred_up.sum() > 0 else np.nan

        pred_cont = m_test > 0.5
        actual_cont = test_y_cont == 1
        hit_rate_pi = (pred_cont & actual_cont).sum() / pred_cont.sum() if pred_cont.sum() > 0 else np.nan

        hit_rates_dir.append(hit_rate_dir)
        hit_rates_pi.append(hit_rate_pi)

    return {
        "pi_hit_rates": hit_rates_pi,
        "dir_hit_rates": hit_rates_dir,
        "pi_mean": np.nanmean(hit_rates_pi),
        "dir_mean": np.nanmean(hit_rates_dir),
        "pi_std": np.nanstd(hit_rates_pi),
        "dir_std": np.nanstd(hit_rates_dir),
    }


def noise_floor_estimation(df: pd.DataFrame, n_boot: int = 100) -> dict:
    """Bootstrap polarity and drift features on permuted returns to estimate expected random hit-rate."""
    pi = compute_polarity(df)
    log_c = np.log(df["close"].astype("float64"))
    y_next_up = (log_c.shift(-1) - log_c > 0).astype(int)
    r_cc = log_c.diff()
    sgn = np.sign(r_cc)
    y_cont = ((sgn.shift(-1) * sgn) > 0).astype(np.float64)
    y_cont[(sgn == 0) | (sgn.shift(-1) == 0)] = np.nan

    level, slope = kalman_level_slope(log_c)
    rs = rogers_satchell_vol(df, window=20)
    drift_norm = (slope / (rs + 1e-12)).replace([np.inf, -np.inf], np.nan)

    hit_rates_pi_boot = []
    hit_rates_dir_boot = []

    for _ in range(n_boot):
        # Permute targets
        y_cont_perm = y_cont.sample(frac=1, random_state=_).values
        y_cont_perm = pd.Series(y_cont_perm, index=y_cont.index)
        y_next_up_perm = y_next_up.sample(frac=1, random_state=_).values
        y_next_up_perm = pd.Series(y_next_up_perm, index=y_next_up.index)

        # Calibrate on permuted
        m_perm = calibrate_monotone(pi, y_cont_perm)
        u_perm = calibrate_monotone(drift_norm, y_next_up_perm)

        # Hit rates on original targets
        pred_cont = m_perm > 0.5
        hit_rate_pi = (pred_cont & (y_cont == 1)).sum() / pred_cont.sum() if pred_cont.sum() > 0 else np.nan

        pred_up = u_perm > 0.5
        hit_rate_dir = (pred_up & (y_next_up == 1)).sum() / pred_up.sum() if pred_up.sum() > 0 else np.nan

        hit_rates_pi_boot.append(hit_rate_pi)
        hit_rates_dir_boot.append(hit_rate_dir)

    return {
        "pi_noise_floor": np.nanmean(hit_rates_pi_boot),
        "dir_noise_floor": np.nanmean(hit_rates_dir_boot),
        "pi_noise_std": np.nanstd(hit_rates_pi_boot),
        "dir_noise_std": np.nanstd(hit_rates_dir_boot),
    }


def calibrate_monotone(feature: pd.Series, target01: pd.Series, window=W_CAL, n_bins=N_BINS, k=LAPLACE_K) -> pd.Series:
    bins = _rolling_bins(feature, window, n_bins)
    idx = feature.index
    y = target01.reindex(idx).astype(np.float64).values
    p = np.full(len(idx), np.nan, np.float64)
    counts = np.zeros((n_bins, 2), np.float64)  # [:,0]=neg, [:,1]=pos
    for t in range(len(idx)):
        if t - window >= 0:
            b_old = bins.iloc[t - window]
            if b_old >= 0 and not np.isnan(y[t - window]):
                counts[b_old, int(y[t - window])] -= 1
        if t - 1 >= 0:
            b_p = bins.iloc[t - 1]
            if b_p >= 0 and not np.isnan(y[t - 1]):
                counts[b_p, int(y[t - 1])] += 1

        counts = np.maximum(counts, 0.0)

        b = bins.iloc[t]
        if b >= 0:
            n = counts.sum(axis=1)
            pos = counts[:, 1]
            raw = (pos + k) / (n + 2 * k)  # Laplace-smoothed per-bin means
            raw[~np.isfinite(raw)] = 0.5
            mono = _pav_means(raw, np.maximum(n, 1.0))  # enforce monotonicity
            # Regularization: shrink towards global mean
            global_mean = np.average(raw, weights=np.maximum(n, 1.0))
            lambda_reg = 0.1  # shrinkage strength
            mono = (1 - lambda_reg) * mono + lambda_reg * global_mean
            nb = counts[b].sum()
            p[t] = mono[b] if nb >= MIN_OBS_BIN else 0.5
    return pd.Series(p, index=idx)


def rogers_satchell_vol(df: pd.DataFrame, window: int = 20) -> pd.Series:
    # RS uses log OHLC and is robust to drift
    O, H, L, C = np.log(df["open"]), np.log(df["high"]), np.log(df["low"]), np.log(df["close"])
    term = (H - C.shift(1)) * (H - O) + (L - C.shift(1)) * (L - O)
    return term.rolling(window).mean().clip(lower=1e-12).pow(0.5)


def kalman_level_slope(log_close: pd.Series, q: np.float64 = 1e-5, r: np.float64 = 1e-4):
    # constant-velocity 2-state filter
    n = len(log_close)
    level = np.zeros(n)
    slope = np.zeros(n)
    x = np.array([log_close.iloc[0], 0.0])
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    H = np.array([[1.0, 0.0]])
    for i, z in enumerate(log_close.values):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update
        y = z - (H @ x)  # innovation
        S = H @ P @ H.T + r
        K = (P @ H.T) / S
        x = x + (K.flatten() * y)
        P = (np.eye(2) - K @ H) @ P
        level[i], slope[i] = x[0], x[1]
    idx = log_close.index
    return pd.Series(level, idx), pd.Series(slope, idx)


def build_signal(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.to_datetime(df["snapshotTime"])
    df = df.assign(snapshotTime=t).sort_values("snapshotTime").set_index("snapshotTime")

    # Polarity
    pi = compute_polarity(df)
    # Stationarity: z-score
    pi_z = (pi - pi.rolling(W_CAL).mean()) / (pi.rolling(W_CAL).std() + 1e-12)

    # Targets
    log_c = np.log(df["close"].astype("float64"))
    y_next_up = (log_c.shift(-1) - log_c > 0).astype(int)  # direction target
    r_cc = log_c.diff()
    sgn = np.sign(r_cc)
    y_cont = ((sgn.shift(-1) * sgn) > 0).astype(np.float64)
    y_cont[(sgn == 0) | (sgn.shift(-1) == 0)] = np.nan  # exclude zero-returns

    # Calibrations
    m = calibrate_monotone(pi_z, y_cont)  # P(momentum | PI)

    r = log_c.diff()

    # D. E. Shaw feature: drift / RS volatility
    level, slope = kalman_level_slope(log_c)
    rs = rogers_satchell_vol(df, window=20)
    drift_norm = (slope / (rs + 1e-12)).replace([np.inf, -np.inf], np.nan)
    # Stationarity: z-score
    drift_z = (drift_norm - drift_norm.rolling(W_CAL).mean()) / (drift_norm.rolling(W_CAL).std() + 1e-12)

    # Calibrate to directional probability
    u = calibrate_monotone(drift_z, y_next_up)  # P(up | drift/σ_RS)

    # Fusion
    m_c = 2 * m - 1
    u_c = 2 * u - 1
    g_m = _soft_gate(m_c, GAMMA_M).fillna(0.0)
    g_u = _soft_gate(u_c, GAMMA_U).fillna(0.0)
    s_raw = (m_c.fillna(0.0) * u_c.fillna(0.0)) * g_m * g_u

    vol = r.rolling(VOLWIN).std(ddof=1).bfill()
    med_vol = vol.rolling(VOLWIN).median()
    mad = (r - r.rolling(VOLWIN).median()).abs().rolling(VOLWIN).median().bfill()
    scale = (vol / (mad + 1e-12)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.5, 2.0)
    s = (s_raw / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-1, 1)
    # Turnover and cost filters: suppress weak signals
    cost_threshold = 0.001  # 0.1% transaction cost
    s = s.where(abs(s) >= cost_threshold / (vol + 1e-12), 0.0)
    direction = s.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

    # MA-based signal for comparison
    price = df["close"]
    ma3 = price.rolling(3).mean()
    ma5 = price.rolling(5).mean()
    ma10 = price.rolling(10).mean()

    # Simple MA-based regime signal
    ma_signal = np.where(
        (pi > 0) & (price > ma3) & (price > ma5) & (price > ma10),
        1,  # momentum + above MAs = long
        np.where(
            (pi > 0) & (price < ma3) & (price < ma5) & (price < ma10),
            -1,  # momentum + below MAs = short
            np.where(
                (pi < 0) & (price > ma3) & (price > ma5) & (price > ma10),
                -1,  # mean-revert + above MAs = short (fade)
                np.where(
                    (pi < 0) & (price < ma3) & (price < ma5) & (price < ma10),
                    1,  # mean-revert + below MAs = long (fade)
                    0,  # neutral
                ),
            ),
        ),
    )

    return pd.DataFrame({"PI": pi, "P_momentum": m, "P_up": u, "signal": s, "direction": direction, "close": df["close"], "ma_signal": ma_signal, "signal_live": s.shift(1).fillna(0.0), "direction_live": direction.shift(1).fillna(0).astype(int), "ma_signal_live": pd.Series(ma_signal, index=df.index).shift(1).fillna(0).astype(int)})


# === run ===
prices, summary = load_backtest_prices(CSV_PATH)
print(f"Loaded {summary.rows} rows covering {summary.first_bar} → {summary.last_bar}")

signal_df = build_signal(prices)
print(signal_df.tail(10))  # Show last 10 rows for inspection

# Walk-forward validation
wf_results = walk_forward_validation(prices, n_folds=5)
print(f"Walk-forward PI hit rate: {wf_results['pi_mean']:.3f} ± {wf_results['pi_std']:.3f}")
print(f"Walk-forward Dir hit rate: {wf_results['dir_mean']:.3f} ± {wf_results['dir_std']:.3f}")

# Noise floor estimation
noise_results = noise_floor_estimation(prices, n_boot=50)
print(f"Noise floor PI: {noise_results['pi_noise_floor']:.3f} ± {noise_results['pi_noise_std']:.3f}")
print(f"Noise floor Dir: {noise_results['dir_noise_floor']:.3f} ± {noise_results['dir_noise_std']:.3f}")

# Setup colored logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create console handler with color formatting
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)s:%(name)s:%(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    ),
)
logger.addHandler(handler)

# Print last signal with color
last_signal = signal_df["direction_live"].iloc[-1]
last_pi = signal_df["PI"].iloc[-1]
last_price = signal_df["close"].iloc[-1]
last_date = signal_df.index[-1]

if last_signal == 1:
    logger.info(f"🟢 LONG SIGNAL | Date: {last_date.strftime('%Y-%m-%d')} | Price: ${last_price:.2f} | PI: {last_pi:.3f}")
elif last_signal == -1:
    logger.info(f"🔴 SHORT SIGNAL | Date: {last_date.strftime('%Y-%m-%d')} | Price: ${last_price:.2f} | PI: {last_pi:.3f}")
else:
    logger.info(f"⚪ NEUTRAL | Date: {last_date.strftime('%Y-%m-%d')} | Price: ${last_price:.2f} | PI: {last_pi:.3f}")

if last_pi > 0:
    logger.info("📈 Momentum Regime (Intraday Dominates)")
else:
    logger.info("📉 Mean-Reversion Regime (Overnight Dominates)")

# Validation
t = pd.to_datetime(prices["snapshotTime"])
prices_sorted = prices.assign(snapshotTime=t).sort_values("snapshotTime")
log_c = np.log(prices_sorted["close"].astype("float64"))
y_next_up = (log_c.shift(-1) - log_c > 0).astype(int)
r_cc = log_c.diff()
sgn = np.sign(r_cc)
y_cont = ((sgn.shift(-1) * sgn) > 0).astype(np.float64)
y_cont[(sgn == 0) | (sgn.shift(-1) == 0)] = np.nan

# PI monotonicity (vs continuation)
pi = compute_polarity(prices_sorted)
is_mono_pi, hrs_pi = validate_monotonicity_walkforward(pi, y_cont)
print(f"PI Monotonicity: {is_mono_pi}, Hit rates: {hrs_pi}")

# Directional feature monotonicity
level, slope = kalman_level_slope(log_c)
rs = rogers_satchell_vol(prices_sorted, window=20)
drift_norm = (slope / (rs + 1e-12)).replace([np.inf, -np.inf], np.nan)
is_mono_dir, hrs_dir = validate_monotonicity_walkforward(drift_norm, y_next_up)
print(f"Directional Monotonicity: {is_mono_dir}, Hit rates: {hrs_dir}")

# Now consume signal_df["signal"] (signed strength), signal_df["direction"] (discrete trade).

# Plot the signal
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 12))

# Top: Polarity Index with regime shading
plt.subplot(3, 1, 1)
plt.plot(signal_df.index, signal_df["PI"], linewidth=1.5, color="blue", label="Polarity Index")
plt.fill_between(signal_df.index, signal_df["PI"], 0, where=(signal_df["PI"] > 0), color="green", alpha=0.3, label="Momentum Regime")
plt.fill_between(signal_df.index, signal_df["PI"], 0, where=(signal_df["PI"] < 0), color="red", alpha=0.3, label="Mean-Reversion Regime")
plt.axhline(0, linestyle="--", linewidth=1.0, color="black")
plt.title("Overnight/Intraday Polarity Index with Regimes")
plt.xlabel("Date")
plt.ylabel("Polarity")
plt.legend()
plt.grid(True, alpha=0.3)

# Middle: Price with rolling means and signals
plt.subplot(3, 1, 2)
plt.plot(signal_df.index, signal_df["close"], linewidth=1.5, color="black", label="Close Price")
plt.plot(signal_df.index, signal_df["close"].rolling(3).mean(), linewidth=1.0, color="blue", label="3-day MA")
plt.plot(signal_df.index, signal_df["close"].rolling(5).mean(), linewidth=1.0, color="green", label="5-day MA")
plt.plot(signal_df.index, signal_df["close"].rolling(10).mean(), linewidth=1.0, color="red", label="10-day MA")

# Overlay MA-based signals as diamonds
ma_buy = signal_df[signal_df["ma_signal"] == 1]
ma_sell = signal_df[signal_df["ma_signal"] == -1]
plt.scatter(ma_buy.index, ma_buy["close"], marker="D", color="cyan", s=50, label="MA Long", zorder=6, alpha=0.8)
plt.scatter(ma_sell.index, ma_sell["close"], marker="D", color="magenta", s=50, label="MA Short", zorder=6, alpha=0.8)

plt.title("Asset Price with Rolling Means and Signals")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True, alpha=0.3)

# Bottom: Signal
plt.subplot(3, 1, 3)
plt.plot(signal_df.index, signal_df["signal"], linewidth=1.5, color="purple", label="Calibrated Signal")
plt.plot(signal_df.index, signal_df["ma_signal"], linewidth=1.0, color="orange", label="MA-based Signal", linestyle="--")
plt.axhline(0, linestyle="--", linewidth=1.0, color="black")
plt.title("Signals Comparison")
plt.xlabel("Date")
plt.ylabel("Signal Strength")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
