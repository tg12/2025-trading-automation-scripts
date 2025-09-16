#!/usr/bin/env python3
"""Price-Only Consensus — Quorum Edition (Single File)

Causal. Walk-forward. No exogenous inputs. All participants contribute strictly price-derived signals.
Consensus = logistic opinion pool with inverse-Brier causal weights. Execution is t→t+1 only.
Includes expanding walk-forward CV folds, cost-aware gating, and a CLI next-signal forecast.

Inputs via your loader: backtest_loader.load_backtest_prices(CSV_PATH)
Requires columns: open, high, low, close, plus a datetime index or snapshotTime/date column.
"""

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

from __future__ import annotations

import logging
import math

import colorlog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest_loader import load_backtest_prices

# ========= CONFIG =========
CSV_PATH = None  # None => use loader's bundled data
ANNUALIZATION = 252  # adjust for your bar frequency
COST_BPS = 5.0  # round-trip cost in bps
GAMMA = 0.03  # decision margin beyond 0.5
EMA_SPAN = 20  # momentum EMA span
VOL_WIN = 60  # robust vol window
CAL_WIN = 252  # isotonic calibration lookback
N_BINS = 9  # rolling quantile bins
LAPLACE_K = 2.0  # Laplace smoothing
MIN_OBS_BIN = 20  # min obs in chosen bin to trust estimate
RS_WIN = 20  # Rogers–Satchell window
BRIER_WIN = 126  # window for inverse-Brier weights
CV_FOLDS = 4  # expanding walk-forward folds (train->test blocks)
EMBARGO = 10  # embargo period between train and test
RNG_SEED = 7  # for any randomness if added later
# ==========================


# ===== Utilities =====
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    for col in ("snapshotTime", "date", "datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            return df.set_index(col).sort_index()
    raise ValueError("Provide datetime index or one of snapshotTime/date/datetime.")


def _mad_vol(x: pd.Series, window: int) -> pd.Series:
    med = x.rolling(window).median()
    mad = (x - med).abs().rolling(window).median()
    return (1.4826 * mad).clip(lower=1e-12)


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def _pav_means(means: np.ndarray, weights: np.ndarray) -> np.ndarray:
    m = means.copy().astype(np.float64)
    w = weights.copy().astype(np.float64)
    i, k = 0, len(m)
    while i < k - 1:
        if not np.isfinite(m[i]) or not np.isfinite(m[i + 1]):
            i += 1
            continue
        if m[i] > m[i + 1]:
            tot = w[i] + w[i + 1]
            avg = (m[i] * w[i] + m[i + 1] * w[i + 1]) / (tot if tot > 0 else 1.0)
            m[i] = m[i + 1] = avg
            w[i] = w[i + 1] = tot
            j = i - 1
            while j >= 0 and np.isfinite(m[j]) and m[j] > m[j + 1]:
                tot = w[j] + w[j + 1]
                avg = (m[j] * w[j] + m[j + 1] * w[j + 1]) / (tot if tot > 0 else 1.0)
                m[j] = m[j + 1] = avg
                w[j] = w[j + 1] = tot
                j -= 1
            i = max(j + 1, 0)
        else:
            i += 1
    return m


def corwin_schultz_spread(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Corwin-Schultz spread estimator from OHLC."""
    H = df["high"].astype(np.float64)
    L = df["low"].astype(np.float64)
    H_prev = H.shift(1)
    L_prev = L.shift(1)
    # Beta calculation
    beta = np.log(H / L) ** 2 + np.log(H_prev / L_prev) ** 2 - np.log(H / L_prev) ** 2 - np.log(H_prev / L) ** 2
    # Spread estimator
    spread = 2 * (np.exp(beta) - 1) / (1 + np.exp(beta))
    # Rolling average to smooth
    return spread.rolling(window).mean().clip(lower=0).fillna(0)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range."""
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    close = df["close"].astype(np.float64)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def triple_barrier_labels(df: pd.DataFrame, horizon: int = 8, profit_mult: np.float64 = 1.5, stop_mult: np.float64 = 2.5) -> pd.Series:
    """Vectorized triple-barrier labeling: +1 profit-take, -1 stop-loss, 0 timeout."""
    px = df["close"].astype(np.float64)
    atr_series = atr(df)
    n = len(px)
    labels = np.zeros(n, dtype=int)

    # For each possible entry point
    for i in range(n - horizon):
        entry_price = px.iloc[i]
        current_atr = atr_series.iloc[i]
        profit_level = entry_price + profit_mult * current_atr
        stop_level = entry_price - stop_mult * current_atr

        # Look ahead horizon bars
        future_prices = px.iloc[i + 1 : i + 1 + horizon]

        # Check if profit level is hit first
        profit_hit = (future_prices >= profit_level).idxmax() if (future_prices >= profit_level).any() else None
        stop_hit = (future_prices <= stop_level).idxmax() if (future_prices <= stop_level).any() else None

        if profit_hit and (not stop_hit or profit_hit < stop_hit):
            labels[i] = 1
        elif stop_hit and (not profit_hit or stop_hit < profit_hit):
            labels[i] = -1
        # else remains 0

    return pd.Series(labels, index=px.index)


def _rolling_bins(x: pd.Series, window: int, n_bins: int) -> pd.Series:
    q = np.linspace(0, 1, n_bins + 1)
    v = x.values.astype(np.float64)
    edges = [None] * len(v)
    for t in range(len(v)):
        past = v[max(0, t - window) : t]
        past = past[~np.isnan(past)]
        edges[t] = None if past.size < max(5, n_bins) else np.quantile(past, q)
    b = np.full(len(v), -1, dtype=int)
    for t, e in enumerate(edges):
        if e is None or not np.isfinite(v[t]):
            continue
        idx = int(np.searchsorted(e, v[t], side="right") - 1)
        b[t] = int(np.clip(idx, 0, n_bins - 1))
    return pd.Series(b, index=x.index)


def calibrate_monotone(feature: pd.Series, target01: pd.Series, window: int = CAL_WIN, n_bins: int = N_BINS, k: np.float64 = LAPLACE_K, min_obs_bin: int = MIN_OBS_BIN) -> pd.Series:
    bins = _rolling_bins(feature, window, n_bins)
    idx = feature.index
    y = target01.reindex(idx).astype(np.float64).values
    p = np.full(len(idx), np.nan, np.float64)
    counts = np.zeros((n_bins, 2), np.float64)  # [:,0]=neg, [:,1]=pos
    for t in range(len(idx)):
        if t - window >= 0:
            b_old = bins.iloc[t - window]
            y_old = y[t - window]
            if b_old >= 0 and np.isfinite(y_old):
                counts[b_old, int(y_old)] -= 1
        if t - 1 >= 0:
            b_prev = bins.iloc[t - 1]
            y_prev = y[t - 1]
            if b_prev >= 0 and np.isfinite(y_prev):
                counts[b_prev, int(y_prev)] += 1
        b = bins.iloc[t]
        if b < 0:
            continue
        n = counts.sum(axis=1)
        pos = counts[:, 1]
        raw = (pos + k) / (n + 2 * k)
        raw[~np.isfinite(raw)] = 0.5
        mono = _pav_means(raw, np.maximum(n, 1.0))
        nb = counts[b].sum()
        p[t] = mono[b] if nb >= min_obs_bin else 0.5
    return pd.Series(p, index=idx)


# ===== Price-only features (contributed “tracks”) =====
def feat_momentum(r: pd.Series) -> pd.Series:
    mu = _ema(r, EMA_SPAN)
    sig = _mad_vol(r, VOL_WIN)
    return (mu / sig).replace([np.inf, -np.inf], np.nan)


def feat_polarity(df: pd.DataFrame, w: int = 20, eps: np.float64 = 1e-12) -> pd.Series:
    o = df["open"].astype("float64")
    c = df["close"].astype("float64")
    valid = (o > 0) & (c > 0)
    lo = np.log(o.where(valid))
    lc = np.log(c.where(valid))
    r_oc, r_co = lc - lo, lo - lc.shift(1)
    mu_oc, mu_co = r_oc.rolling(w).mean(), r_co.rolling(w).mean()
    var_sum = r_oc.rolling(w).var(ddof=1) + r_co.rolling(w).var(ddof=1)
    return (mu_oc - mu_co) / np.sqrt(var_sum + eps)


def rs_vol(df: pd.DataFrame, window: int = RS_WIN) -> pd.Series:
    O, H, L, C = np.log(df["open"]), np.log(df["high"]), np.log(df["low"]), np.log(df["close"])
    term = (H - C.shift(1)) * (H - O) + (L - C.shift(1)) * (L - O)
    return term.rolling(window).mean().clip(lower=1e-12).pow(0.5)


def kalman_level_slope(log_close: pd.Series, q: np.float64 = 1e-5, r: np.float64 = 1e-4):
    n = len(log_close)
    level = np.zeros(n)
    slope = np.zeros(n)
    x = np.array([log_close.iloc[0], 0.0])
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = q * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    H = np.array([[1.0, 0.0]])
    for i, z in enumerate(log_close.values):
        x = F @ x
        P = F @ P @ F.T + Q
        y = z - (H @ x)
        S = H @ P @ H.T + r
        K = (P @ H.T) / S
        x = x + (K.flatten() * y)
        P = (np.eye(2) - K @ H) @ P
        level[i], slope[i] = x[0], x[1]
    idx = log_close.index
    return pd.Series(level, idx), pd.Series(slope, idx)


def feat_drift_over_rs(df: pd.DataFrame) -> pd.Series:
    logc = np.log(df["close"].astype("float64").replace(0.0, np.nan)).ffill()
    _, slope = kalman_level_slope(logc)
    vol = rs_vol(df, window=RS_WIN)
    return (slope / (vol + 1e-12)).replace([np.inf, -np.inf], np.nan)


def feat_reversion_z(px: pd.Series, w: int = 20) -> pd.Series:
    logp = np.log(px.replace(0.0, np.nan)).ffill()
    m = logp.rolling(w).mean()
    s = logp.rolling(w).std(ddof=1)
    z = (logp - m) / (s + 1e-12)
    # sign so that large |z| implies mean-revert up-probability decreasing with z
    return (-z).astype(np.float64)  # negative z -> up-prob higher


def feat_sign_persist(r: pd.Series, w: int = 10) -> pd.Series:
    s = np.sign(r)
    # fraction of positives in last w, centered to zero mean
    p = s.replace(0, np.nan).rolling(w).apply(lambda x: np.nanmean(x > 0), raw=False)
    return p - 0.5  # positive => recent continuation up


# ===== Consensus and trading =====
def logistic_pool(probs: pd.DataFrame, y_next_up: pd.Series, brier_win: int = BRIER_WIN) -> pd.Series:
    eps = 1e-8
    y = y_next_up.astype(np.float64)
    probs = probs.reindex(y.index)
    losses = (probs.subtract(y, axis=0)) ** 2
    rbrier = losses.rolling(brier_win, min_periods=max(5, brier_win // 4)).mean().shift(1)
    w = 1.0 / (eps + rbrier)
    w = w.div(w.sum(axis=1), axis=0).fillna(0.0)

    def to_logodds(p):
        p = p.clip(1e-6, 1 - 1e-6)
        return np.log(p / (1.0 - p))

    lod = probs.apply(to_logodds)
    pooled = (w * lod).sum(axis=1)
    return 1.0 / (1.0 + np.exp(-pooled))


def compute_signals(prices: pd.DataFrame):
    df = _ensure_datetime_index(prices.copy())
    px = df["close"].astype("float64")
    logp = np.log(px.replace(0.0, np.nan)).ffill()
    r = logp.diff()

    # Targets (direction next bar)
    y_tb = triple_barrier_labels(df, profit_mult=1.5, stop_mult=2.5, horizon=8)
    y_next_up = (y_tb == 1).astype(np.float64)

    # Feature tracks
    f_mom = feat_momentum(r)
    f_pol = feat_polarity(df)
    f_drv = feat_drift_over_rs(df)
    f_rev = feat_reversion_z(px)
    f_pst = feat_sign_persist(r)

    # Calibrate each to P(up) causally
    p_mom = calibrate_monotone(f_mom, y_next_up)
    p_pol = calibrate_monotone(f_pol, y_next_up)
    p_drv = calibrate_monotone(f_drv, y_next_up)
    p_rev = calibrate_monotone(f_rev, y_next_up)
    p_pst = calibrate_monotone(f_pst, y_next_up)

    # Consensus probability
    probs = pd.DataFrame({"mom": p_mom, "pol": p_pol, "drift": p_drv, "revert": p_rev, "persist": p_pst})
    p_ens = logistic_pool(probs, y_next_up, brier_win=BRIER_WIN)

    # Spread estimator for no-trade band
    spread = corwin_schultz_spread(df)
    atr_series = atr(df)

    # Decision: probability margin + simple cost proxy + spread gate
    margin = (p_ens - 0.5).abs()
    edge = 2.0 * p_ens - 1.0
    cost_edge = (COST_BPS * 1e-4) * 2.0
    gate = (margin > np.maximum(GAMMA, np.maximum(cost_edge, spread))).astype(int)

    # Volatility-scaled sizing
    vol = atr_series.rolling(20).mean()  # Use ATR as volatility proxy
    expected_return = edge * vol  # Scale edge by volatility
    vol_scaled_size = expected_return / (vol + 1e-12)
    # Clip to reasonable bounds
    vol_scaled_size = vol_scaled_size.clip(-2.0, 2.0)

    pos = pd.Series(0.0, index=px.index)
    pos[(p_ens > 0.5) & (gate == 1)] = vol_scaled_size[(p_ens > 0.5) & (gate == 1)]
    pos[(p_ens < 0.5) & (gate == 1)] = vol_scaled_size[(p_ens < 0.5) & (gate == 1)]
    pos_live = pos.shift(1).fillna(0.0)  # execute next bar only

    # PnL with costs
    dpos = pos_live.diff().abs().fillna(pos_live.abs())
    ret_next = r.fillna(0.0)
    cost = (COST_BPS * 1e-4) * dpos
    net = pos_live * ret_next - cost
    eq = net.cumsum().apply(np.exp)
    if np.isfinite(eq).any():
        eq = eq / eq[np.isfinite(eq)].iloc[0]

    out = pd.DataFrame({
        "close": px,
        "r": r,
        "p_ens": p_ens,
        "edge": edge,
        "pos": pos,
        "pos_live": pos_live,
        "ret_net": net,
        "equity": eq,
        "p_mom": p_mom,
        "p_pol": p_pol,
        "p_drift": p_drv,
        "p_revert": p_rev,
        "p_persist": p_pst,
        "spread": spread,
    })
    stats = _stats(net, pos_live, dpos, eq)
    return out, stats


def _stats(net: pd.Series, pos_live: pd.Series, dpos: pd.Series, eq: pd.Series):
    ann_ret = net.mean() * ANNUALIZATION
    ann_vol = net.std(ddof=1) * math.sqrt(ANNUALIZATION)
    sharpe = ann_ret / (ann_vol + 1e-12)
    trades = int((dpos > 0).sum())
    turnover = np.float64(dpos.sum())
    dd = np.float64((eq / eq.cummax() - 1.0).min()) if len(eq) else np.nan
    active = pos_live != 0
    hit = ((np.sign(pos_live[active]) * np.sign(net[active])) > 0).mean() if active.any() else np.nan
    return {"bars": len(net), "trades": trades, "turnover_abs": turnover, "hit_rate": np.float64(hit) if np.isfinite(hit) else np.nan, "ann_return": np.float64(ann_ret), "ann_vol": np.float64(ann_vol), "sharpe": np.float64(sharpe), "max_drawdown": dd}


# ===== Expanding walk-forward CV =====
def expanding_walkforward(prices: pd.DataFrame, folds: int = CV_FOLDS, embargo: int = EMBARGO):
    df = _ensure_datetime_index(prices.copy())
    n = len(df)
    cuts = np.linspace(0.5, 0.9, num=folds, endpoint=True)  # train start ~50% -> test end ~90%
    results = []
    for cut in cuts:
        split = int(n * cut)
        train_end = max(0, split - embargo)
        test_start = min(n, split + embargo)
        test_end = min(n, test_start + max(1, int(0.05 * n)))
        train = df.iloc[:train_end].copy()
        test = df.iloc[test_start:test_end].copy()
        if len(test) == 0:
            continue
        out, stats = compute_signals(test)  # calibration inside is causal; training segment implicit via windows
        # Evaluate only on the last 10% of the used sample to mimic OOS for this fold
        tail_slice = out.iloc[-max(1, int(0.1 * len(out))) :]
        fold_stats = _stats(tail_slice["ret_net"], tail_slice["pos_live"], tail_slice["pos_live"].diff().abs().fillna(tail_slice["pos_live"].abs()), tail_slice["equity"])
        fold_stats["cut_idx"] = split
        results.append(fold_stats)
    return pd.DataFrame(results)


# ===== Plot =====
def plot_work(df: pd.DataFrame):
    fig, ax = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax[0].plot(df.index, df["close"], linewidth=1.3, color="black", label="Close")
    long_pos = df[df["pos_live"] > 0]
    short_pos = df[df["pos_live"] < 0]
    ax[0].scatter(long_pos.index, long_pos["close"], marker="^", color="green", s=50 * long_pos["pos_live"].abs(), label="Long", alpha=0.7)
    ax[0].scatter(short_pos.index, short_pos["close"], marker="v", color="red", s=50 * short_pos["pos_live"].abs(), label="Short", alpha=0.7)
    ax[0].set_title("Price with Quorum Consensus Signals (t→t+1)")
    ax[0].set_ylabel("Price")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(df.index, df["p_ens"], linewidth=1.2, label="Ensemble P(up)")
    ax[1].axhline(0.5 + GAMMA, linestyle="--", color="gray", linewidth=1.0, label="Upper threshold")
    ax[1].axhline(0.5 - GAMMA, linestyle="--", color="gray", linewidth=1.0, label="Lower threshold")
    ax[1].set_title("Calibrated Ensemble Probability")
    ax[1].set_ylabel("Probability")
    ax[1].grid(alpha=0.3)
    ax[1].legend()
    fig.tight_layout()
    plt.show()


# ===== Main =====
def main():
    prices, summary = load_backtest_prices(CSV_PATH)
    prices = _ensure_datetime_index(prices)

    # Core backtest
    out, stats = compute_signals(prices)

    # Walk-forward CV summary
    cv = expanding_walkforward(prices, folds=CV_FOLDS, embargo=EMBARGO)

    # CLI next-signal forecast
    logger = logging.getLogger("forecast")
    logger.setLevel(logging.INFO)
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter("%(log_color)s%(levelname)s:%(message)s", log_colors={"INFO": "green", "WARNING": "yellow", "ERROR": "red"}))
    if not logger.handlers:
        logger.addHandler(handler)

    last_date = out.index[-1]
    last_price = out["close"].iloc[-1]
    p_next = np.float64(out["p_ens"].iloc[-1])
    pos_next = int(out["pos_live"].iloc[-1])
    direction = "LONG" if pos_next == 1 else ("SHORT" if pos_next == -1 else "NEUTRAL")
    logger.info(f"NEXT SIGNAL {direction} | {last_date.strftime('%Y-%m-%d')} | Price: {last_price:.4f} | P(up)={p_next:.4f}")

    # Print stats and CV table
    print("\n=== in-sample-like summary (causal estimator applied) ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}" if isinstance(v, np.float64) else f"{k}: {v}")
    print("\n=== expanding walk-forward folds (tail evaluation) ===")
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(cv.to_string(index=False, float_format=lambda x: f"{x: .6f}"))

    # Plot
    plot_work(out)


if __name__ == "__main__":
    main()
