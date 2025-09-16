#!/usr/bin/env python3
"""Retail-Ready Price-Only Trading System (daily/weekly modes)
Causal t->t+1 open execution. Robust stats. Audit trail.

Inputs:
    DataFrame with DateTimeIndex and columns: open, high, low, close

Outputs:
  universal_signal_next_open.csv   # forward tradable signals
  universal_panel.csv              # features + audit trail
  Plot with: price & M_ref + markers, close-M_ref, P(up)
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

import json
from pathlib import Path
import logging
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest_loader import (
    load_backtest_prices,  # externalized loader (returns (df, meta))
)

# ======================= MODE / PRESETS =======================
MODE: str = "daily"  # "daily" or "weekly"

PRESETS: Dict[str, Dict] = {
    "daily": {"WIN_PI": 5, "RMA": 5, "HMA": 9, "DEMA": 10, "MED": 7, "Z_MAD": 50, "TAU_IN": 0.12, "TAU_OUT": 0.08, "STOP_MULT": 1.75, "TIME_EXIT": 4, "COOLDOWN": 1},
    "weekly": {"WIN_PI": 8, "RMA": 14, "HMA": 21, "DEMA": 12, "MED": 15, "Z_MAD": 70, "TAU_IN": 0.12, "TAU_OUT": 0.08, "STOP_MULT": 2.50, "TIME_EXIT": 9, "COOLDOWN": 2},
}
P = PRESETS[MODE]
R_LEN, H_LEN, D_SPAN, M_LEN = P["RMA"], P["HMA"], P["DEMA"], P["MED"]

# ======================= EXEC / FILTER CFG ====================
CSV_PATH: str | None = None  # set path or replace loader
G_THR = 0.0075  # |open-close[-1]| threshold (0.75%)
V_LO = 0.004  # low vol bound (0.4%)
V_HI = 0.02  # high vol bound (2.0%)
Z_SOFT, Z_HARD = 1.5, 2.5


# ========================= UTILITIES ==========================
def rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False).mean().rename(f"RMA{n}")


def wma(x: pd.Series, n: int) -> pd.Series:
    n = max(int(n), 2)
    w = np.arange(1, n + 1, dtype=np.float64)
    w /= w.sum()
    return x.rolling(n).apply(lambda y: (y * w).sum(), raw=True).rename(f"WMA{n}")


def hma(x: pd.Series, n: int) -> pd.Series:
    n = max(int(n), 2)
    n2 = max(n // 2, 2)
    ns = max(int(np.sqrt(n)), 2)
    return wma(2 * wma(x, n2) - wma(x, n), ns).rename(f"HMA{n}")


def dema(x: pd.Series, span: int) -> pd.Series:
    ema1 = x.ewm(span=span, adjust=False).mean()
    ema2 = ema1.ewm(span=span, adjust=False).mean()
    return (2 * ema1 - ema2).rename(f"DEMA{span}")


def rolling_mad(s: pd.Series, w: int, minp: int) -> pd.Series:
    return s.rolling(w, min_periods=minp).apply(lambda a: np.median(np.abs(a - np.median(a))), raw=True)


def robust_cap(s: pd.Series, w: int = 100, k: np.float64 = 6.0) -> pd.Series:
    med = s.rolling(w, min_periods=w // 2).median()
    mad = rolling_mad(s, w, w // 2)
    scale = 1.4826 * mad
    return s.clip(lower=med - k * scale, upper=med + k * scale)


# =============== PILLAR 1: TREND ENSEMBLE (M_ref) ==============
def build_means(price: pd.Series) -> pd.DataFrame:
    m_rma = rma(price, R_LEN)
    m_hma = hma(price, H_LEN)
    m_dema = dema(price, D_SPAN)
    m_median = price.rolling(M_LEN, min_periods=M_LEN // 2).median().rename(f"Median{M_LEN}")
    m_ref = 0.4 * m_rma + 0.3 * m_hma + 0.2 * m_dema + 0.1 * m_median
    return pd.DataFrame({
        f"RMA{R_LEN}": m_rma,
        f"HMA{H_LEN}": m_hma,
        f"DEMA{D_SPAN}": m_dema,
        f"Median{M_LEN}": m_median,
        "M_ref": m_ref,
    })


# =============== PILLAR 2: POLARITY (REGIME) ===================
def compute_polarity(df: pd.DataFrame, win: int) -> pd.DataFrame:
    roc = df["open"] / df["close"].shift(1) - 1.0
    rco = df["close"] / df["open"] - 1.0
    roc = robust_cap(roc, w=max(5 * win, 100))
    rco = robust_cap(rco, w=max(5 * win, 100))
    m_oc, m_co = roc.rolling(win).mean(), rco.rolling(win).mean()
    v_oc, v_co = roc.rolling(win).var(), rco.rolling(win).var()
    diff = m_co - m_oc  # intraday minus overnight
    denom = (v_oc + v_co).pow(0.5).replace(0, np.nan)
    pol_raw = (np.sign(diff) * np.abs(diff) / denom).rename("polarity_raw")
    pol = rma(pol_raw, win).rename("polarity")
    return pd.concat([pol_raw, pol], axis=1)


def regime_series(pol_s: pd.Series) -> pd.Series:
    TAU_IN, TAU_OUT = P["TAU_IN"], P["TAU_OUT"]
    state = 0
    out = []
    for x in pol_s.fillna(0.0).values:
        if x >= TAU_IN:
            state = +1  # mean-reversion pressure
        elif x <= -TAU_IN:
            state = -1  # momentum pressure
        elif -TAU_OUT < x < TAU_OUT:
            state = 0
        out.append(state)
    return pd.Series(out, index=pol_s.index, name="regime")


# ===== PILLAR 3: PATH GEOMETRY (DEPTH-2 OHLC SUMMARY) ==========
def path_sig2(df: pd.DataFrame, w: int) -> pd.DataFrame:
    o, h, l, c = [df[k].pct_change().fillna(0.0) for k in ("open", "high", "low", "close")]
    X = pd.concat([o.rename("dO"), h.rename("dH"), l.rename("dL"), c.rename("dC")], axis=1)
    S1 = X.rolling(w).sum().add_suffix("_S1")
    cols = ["dO", "dH", "dL", "dC"]
    feats = {(f"{a}x{b}_S2"): (X[a] * X[b]).rolling(w).sum() for i, a in enumerate(cols) for b in cols[i:]}
    S2 = pd.DataFrame(feats)
    Z = pd.concat([S1, S2], axis=1)
    mu = Z.rolling(max(3, w)).mean()
    sd = Z.rolling(max(3, w)).std().replace(0, np.nan)
    Zz = (Z - mu) / sd
    sig_mean = Zz.mean(axis=1).rename("sig2_meanz")
    sig_energy = np.sign(S1["dC_S1"]).fillna(0.0) * Zz.abs().mean(axis=1).rename("sig2_energy")
    return pd.concat([sig_mean, sig_energy], axis=1)


# ===== PILLAR 4: VOL SQUEEZE / DRIFT TILT (YZ PROXY) ===========
def yz_vol_proxy(df: pd.DataFrame, w: int) -> pd.Series:
    ro = np.log(df["open"] / df["close"].shift(1))
    rc = np.log(df["close"] / df["open"])
    rh = np.log(df["high"] / df["open"])
    rl = np.log(df["low"] / df["open"])
    rs = (rh * (rh - rc) + rl * (rl - rc)).rolling(w).mean()
    vo = ro.rolling(w).var()
    sig = (rs + vo).clip(lower=0).pow(0.5)
    return sig.rename(f"sigmaYZp_{w}")


def drift_tilt(df: pd.DataFrame, w: int, sig: pd.Series) -> pd.Series:
    mu = np.log(df["close"]).diff().rolling(w).mean()
    tilt = (mu / (sig.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
    return tilt.rename(f"tiltYZp_{w}")


# ===== PILLAR 5: RESIDUAL MOMENTUM / REVERSAL ==================
def robust_residual_z(r: pd.Series, w: int) -> pd.Series:
    med = r.rolling(w, min_periods=w // 2).median()
    mad = rolling_mad(r, w, w // 2)
    return ((r - med) / (1.4826 * mad.replace(0, np.nan))).rename(f"residZ_{w}")


# ==================== EXHAUSTION / FILTERS ======================
def compute_exhaustion(close: pd.Series, m_ref: pd.Series) -> pd.DataFrame:
    W = P["Z_MAD"]
    dist = close - m_ref
    mad = rolling_mad(dist, W, max(20, W // 2))
    scale = (1.4826 * mad).replace(0, np.nan)
    return pd.DataFrame({"dist": dist, "dist_z": dist / scale})


def exec_filters(df: pd.DataFrame) -> pd.DataFrame:
    gap = df["open"] / df["close"].shift(1) - 1.0
    gap_ok = gap.abs() < G_THR
    vol = df["close"].pct_change().rolling(14).std()
    vol_mult = np.where((vol < V_LO) | (vol > V_HI), 0.5, 1.0)
    return pd.DataFrame({"gap_ok": gap_ok, "vol_mult": vol_mult})


# =================== META-SCORE / PROBABILITY ===================
def meta_score(df: pd.DataFrame) -> pd.DataFrame:
    slope = (df["M_ref"] - df["M_ref"].shift(1)) / (df["M_ref"].shift(1))
    slope_z = (slope - slope.rolling(50).mean()) / slope.rolling(50).std().replace(0, np.nan)
    slope_z = slope_z.clip(-5, 5).rename("trend_slope_z")

    reg_pressure = df["polarity"].clip(-3, 3).rename("reg_pressure")

    sig_mean = df["sig2_meanz"].clip(-4, 4) if "sig2_meanz" in df else pd.Series(0, index=df.index)
    sig_energy = df["sig2_energy"].clip(-4, 4) if "sig2_energy" in df else pd.Series(0, index=df.index)

    tilt_cols = df.filter(like="tiltYZp_")
    tilt = (tilt_cols.iloc[:, 0] if tilt_cols.shape[1] > 0 else pd.Series(0, index=df.index)).clip(-4, 4).rename("tilt")

    r = df["close"].pct_change()
    rz = robust_residual_z(r, 20).clip(-5, 5)

    z = 0.35 * slope_z.fillna(0) + 0.30 * reg_pressure.fillna(0) + 0.15 * sig_mean.fillna(0) + 0.10 * tilt.fillna(0) + 0.10 * rz.fillna(0)
    meta = np.tanh(z).rename("meta_score")
    p_up = (meta + 1.0) / 2.0
    return pd.DataFrame({"trend_slope_z": slope_z, "meta_score": meta, "p_up": p_up})


# =================== DIRECTION / SIZING / SIGNALS ===============
def compute_direction(df: pd.DataFrame) -> pd.Series:
    above = df["close"] > df["M_ref"]
    f = pd.Series(0, index=df.index, dtype=int, name="direction_raw")
    f[(df["regime"] == +1) & above] = -1
    f[(df["regime"] == +1) & ~above] = +1
    f[(df["regime"] == -1) & above] = +1
    f[(df["regime"] == -1) & ~above] = -1
    return f


def size_from_meta(df: pd.DataFrame) -> pd.Series:
    TAU_IN, TAU_OUT = P["TAU_IN"], P["TAU_OUT"]
    strength = ((df["polarity"].abs() - TAU_OUT) / max(TAU_IN - TAU_OUT, 1e-6)).clip(0, 1)
    z = df["dist_z"].abs()
    ex_mult = np.where(z >= Z_HARD, 0.0, np.where(z >= Z_SOFT, 0.5, 1.0))
    conv = (2 * df["p_up"] - 1.0).abs()
    s = (0.7 * strength + 0.3 * conv) * ex_mult
    return pd.Series(s, index=df.index, name="size_mult")


def apply_cooldown(direction: pd.Series, intent: pd.Series, n_bars: int) -> Tuple[pd.Series, pd.Series]:
    flips = direction.diff().ne(0) & direction.ne(0)
    cool_left = 0
    out = intent.copy()
    cd_flag = pd.Series(0, index=direction.index, dtype=int)
    for i, fl in enumerate(flips.values):
        if fl:
            cool_left = n_bars
        if cool_left > 0:
            out.iloc[i] = 0.0
            cd_flag.iloc[i] = 1
            cool_left -= 1
    return out, cd_flag


def forward_signal(df: pd.DataFrame) -> pd.DataFrame:
    direction = compute_direction(df)
    size_mult = size_from_meta(df)
    intent = direction * size_mult
    # Only gate on gap; spread is implicit in UK spreadbetting pricing
    filt = df["gap_ok"]
    intent[~filt] = 0.0
    intent = intent * df["vol_mult"]
    intent, cd_flag = apply_cooldown(direction, intent, P["COOLDOWN"])

    out = pd.DataFrame(index=df.index)
    out["signal_next_open"] = np.sign(intent.fillna(0)).astype(int)
    out["size_mult_next_open"] = intent.abs().clip(0, 1).fillna(0.0)
    out["stop_multiple_next"] = P["STOP_MULT"]
    out["time_exit_bars_next"] = P["TIME_EXIT"]
    out["p_up_next"] = df["p_up"].clip(0, 1)
    # Audit trail
    out["above"] = (df["close"] > df["M_ref"]).astype(int)
    out["regime_state"] = df["regime"].astype(int)
    out["direction_raw"] = direction.astype(int)
    out["gate_gap"] = (~(df["gap_ok"])).astype(int)
    out["gate_exhaust"] = (df["dist_z"].abs() >= Z_HARD).astype(int)
    out["gate_vol_half"] = (df["vol_mult"] < 1.0).astype(int)
    out["cooldown_flag"] = cd_flag.astype(int)
    return out


# ============================= MAIN ============================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    df, _ = load_backtest_prices(CSV_PATH)
    df = df.copy()

    pr = compute_polarity(df, P["WIN_PI"])
    means = build_means(df["close"])
    ex = compute_exhaustion(df["close"], means["M_ref"])
    reg = regime_series(pr["polarity"]).to_frame()
    sig2 = path_sig2(df[["open", "high", "low", "close"]], w=max(5, R_LEN))
    sigYZ = yz_vol_proxy(df, w=max(10, H_LEN))
    tilt = drift_tilt(df, w=max(10, H_LEN), sig=sigYZ).to_frame()
    filters = exec_filters(df)

    panel = df[["open", "high", "low", "close"]].join([pr, means, ex, reg, sig2, sigYZ, tilt, filters], how="outer")
    meta = meta_score(panel)
    panel = panel.join(meta)
    forward = forward_signal(panel)

    warmup = max(P["WIN_PI"], R_LEN, H_LEN, D_SPAN, M_LEN, H_LEN // 2, int(np.sqrt(H_LEN)), max(5, R_LEN), max(10, H_LEN)) + 10
    panel = panel.iloc[warmup:].copy()
    forward = forward.iloc[warmup:].copy()

    # Plot (simplified for retail spreadbetting clarity)
    fig, (axp, axu, axm) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    axp.plot(panel.index, panel["close"], color="black", lw=1.2, label="Close")
    axp.plot(panel.index, panel["M_ref"], color="orange", lw=1.1, label="M_ref")
    sig = forward["signal_next_open"].reindex(panel.index).fillna(0).astype(int)
    longs, shorts = panel[sig == 1], panel[sig == -1]
    axp.scatter(longs.index, longs["close"], marker="^", s=40, color="green", label="Long")
    axp.scatter(shorts.index, shorts["close"], marker="v", s=40, color="red", label="Short")
    if len(forward):
        last_sig = int(forward["signal_next_open"].iloc[-1])
    else:
        last_sig = 0
    axp.set_title(f"Retail Spreadbetting System — Last Signal: {last_sig}")
    axp.legend(loc="upper left")
    axp.set_ylabel("Price")

    # Deviation panel
    axu.plot(panel.index, panel["close"] - panel["M_ref"], lw=1.0, label="Close−M_ref")
    axu.axhline(0, ls="--", color="grey")
    axu.legend(loc="upper left")

    # Probability panel
    axm.plot(panel.index, forward["p_up_next"].reindex(panel.index), lw=1.0, label="P(up)")
    axm.axhline(0.5, ls="--", color="grey")
    axm.set_ylim(0, 1)
    axm.legend(loc="upper left")
    axm.set_ylabel("Probability")

    plt.tight_layout()
    plt.show()

    # Exports
    forward.to_csv("universal_signal_next_open.csv")
    keep = [
        "close",
        "polarity_raw",
        "polarity",
        "regime",
        f"RMA{R_LEN}",
        f"HMA{H_LEN}",
        f"DEMA{D_SPAN}",
        f"Median{M_LEN}",
        "M_ref",
        "dist",
        "dist_z",
        "sig2_meanz",
        "sig2_energy",
        f"sigmaYZp_{max(10, H_LEN)}",
        f"tiltYZp_{max(10, H_LEN)}",
        "trend_slope_z",
        "meta_score",
        "p_up",
    ]
    panel[[c for c in keep if c in panel.columns]].to_csv("universal_panel.csv")

    # Final next-bar signal snapshot with probability-derived bias band (LONG / SHORT / NEUTRAL)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    if len(forward):
        snap = forward.iloc[-1]
        p_up = float(snap["p_up_next"])  # convert to native types for JSON
        p_down = 1.0 - p_up
        if p_up >= 0.55:
            bias = "LONG"
        elif p_up <= 0.45:
            bias = "SHORT"
        else:
            bias = "NEUTRAL"
        result_payload = {
            "mode": MODE,
            "signal_next_open": int(snap["signal_next_open"]),
            "size_mult_next_open": float(snap["size_mult_next_open"]),
            "p_up_next": p_up,
            "p_down_next": p_down,
            "signal": bias,
            "stop_multiple_next": float(snap["stop_multiple_next"]),
            "time_exit_bars_next": int(snap["time_exit_bars_next"]),
        }
    else:
        # No forward signals produced (insufficient data / all filtered out)
        result_payload = {"mode": MODE, "status": "no_forward_rows"}

    # Print to stdout as before
    print(json.dumps(result_payload, indent=2))

    # Persist JSON to file
    out_file = results_dir / "trading_analysis.txt"
    with out_file.open("w") as f:
        json.dump(result_payload, f, indent=2, default=str)
    logging.info(f"Wrote trading analysis snapshot to {out_file.resolve()}")


if __name__ == "__main__":
    main()
