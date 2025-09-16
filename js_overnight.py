#!/usr/bin/env python3
"""Universal Price-Only Meta-Indicator (daily/weekly modes)
- Pillars fused:
  1) Low-lag robust trend filter ensemble: RMA + HMA + DEMA
  2) Regime pressure (polarity): overnight vs intraday balance
  3) Path geometry (depth-2 OHLC signature proxy)
  4) Volatility squeeze tilt (Yang–Zhang-style σ proxy)
  5) Residual momentum/reversion mixer (robust z of recent r)

Outputs:
- Graph: price with means and markers; predictive P(up) time-series
- CSVs: forward signal, and full panel for analysis
- Next-bar signal printed at end

Notes:
- Pure OHLC. Next-bar execution (t -> t+1 open). No look-ahead.

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

import logging
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# You must provide this loader or replace with your own data ingest.
# Expected df columns: ["open","high","low","close"] with DateTimeIndex.
from backtest_loader import load_backtest_prices

# ======================= MODE / PRESETS =======================
MODE: str = "daily"  # "daily" or "weekly"

PRESETS: Dict[str, Dict] = {
    "daily": {
        "WIN_PI": 5,  # polarity window
        "RMA": 5,  # RMA length
        "HMA": 9,  # HMA length
        "DEMA": 10,  # DEMA span
        "MED": 7,  # median window
        "Z_MAD": 50,  # MAD window for exhaustion
        "TAU_IN": 0.12,
        "TAU_OUT": 0.08,
        "STOP_MULT": 1.75,
        "TIME_EXIT": 4,
        "COOLDOWN": 1,
    },
    "weekly": {
        "WIN_PI": 8,
        "RMA": 14,
        "HMA": 21,
        "DEMA": 12,
        "MED": 15,
        "Z_MAD": 70,
        "TAU_IN": 0.12,
        "TAU_OUT": 0.08,
        "STOP_MULT": 2.5,
        "TIME_EXIT": 9,
        "COOLDOWN": 2,
    },
}
P = PRESETS[MODE]

# ======================= EXEC / FILTER CFG ====================
CSV_PATH: str | None = None
G_THR = 0.0075  # |open gap| threshold (0.75%)
V_LO = 0.004  # low vol clamp
V_HI = 0.02  # high vol clamp
S_THR = 0.0015  # spread threshold (15 bps), use df['spread_bps'] if present


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


# ===================== PILLAR 1: TREND ENSEMBLE =================
def build_means(price: pd.Series) -> pd.DataFrame:
    r_len, h_len, d_span, m_len = P["RMA"], P["HMA"], P["DEMA"], P["MED"]
    m_rma = rma(price, r_len)
    m_hma = hma(price, h_len)
    m_dema = dema(price, d_span)
    m_median = price.rolling(m_len, min_periods=m_len // 2).median().rename(f"Median{m_len}")
    m_ref = 0.4 * m_rma + 0.3 * m_hma + 0.2 * m_dema + 0.1 * m_median
    return pd.DataFrame({
        f"RMA{r_len}": m_rma,
        f"HMA{h_len}": m_hma,
        f"DEMA{d_span}": m_dema,
        f"Median{m_len}": m_median,
        "M_ref": m_ref,
    })


# ===================== PILLAR 2: POLARITY (REGIME) ==============
def compute_polarity(df: pd.DataFrame, win: int) -> pd.DataFrame:
    roc = df["open"] / df["close"].shift(1) - 1.0
    rco = df["close"] / df["open"] - 1.0
    roc = robust_cap(roc, w=max(5 * win, 100))
    rco = robust_cap(rco, w=max(5 * win, 100))
    m_oc, m_co = roc.rolling(win).mean(), rco.rolling(win).mean()
    v_oc, v_co = roc.rolling(win).var(), rco.rolling(win).var()
    diff = m_oc - m_co
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
            state = +1  # mean-revert pressure
        elif x <= -TAU_IN:
            state = -1  # momentum pressure
        elif -TAU_OUT < x < TAU_OUT:
            state = 0
        out.append(state)
    return pd.Series(out, index=pol_s.index, name="regime")


# ========== PILLAR 3: PATH GEOMETRY (DEPTH-2 SIGNATURE PROXY) ===
def path_sig2(df: pd.DataFrame, w: int) -> pd.DataFrame:
    # Normalize inputs to relative moves
    o, h, l, c = [df[k].pct_change().fillna(0.0) for k in ("open", "high", "low", "close")]
    X = pd.concat([o.rename("dO"), h.rename("dH"), l.rename("dL"), c.rename("dC")], axis=1)

    # First-order (S^1): rolling sums
    S1 = X.rolling(w).sum().add_suffix("_S1")

    # Second-order (S^2) proxy: rolling sums of pairwise products (order matters)
    cols = ["dO", "dH", "dL", "dC"]
    feats = {}
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            name = f"{cols[i]}x{cols[j]}_S2"
            feats[name] = (X[cols[i]] * X[cols[j]]).rolling(w).sum()
    S2 = pd.DataFrame(feats)

    # Collapse to compact scores (z-scored)
    Z = pd.concat([S1, S2], axis=1)
    mu = Z.rolling(max(3, w)).mean()
    sd = Z.rolling(max(3, w)).std().replace(0, np.nan)
    Zz = (Z - mu) / sd
    # Two principal summaries: mean z and signed concentration
    sig_mean = Zz.mean(axis=1).rename("sig2_meanz")
    sig_energy = np.sign(S1["dC_S1"]).fillna(0.0) * Zz.abs().mean(axis=1).rename("sig2_energy")
    return pd.concat([sig_mean, sig_energy], axis=1)


# ===== PILLAR 4: VOL SQUEEZE / DRIFT TILT (YZ-STYLE PROXY) ======
def yz_vol_proxy(df: pd.DataFrame, w: int) -> pd.Series:
    # Simple proxy: combine overnight and intraday ranges
    ro = np.log(df["open"] / df["close"].shift(1))
    rc = np.log(df["close"] / df["open"])
    rh = np.log(df["high"] / df["open"])
    rl = np.log(df["low"] / df["open"])
    # Rogers–Satchell-like component
    rs = (rh * (rh - rc) + rl * (rl - rc)).rolling(w).mean()
    # Overnight component variance
    vo = ro.rolling(w).var()
    sig = (rs + vo).clip(lower=0).pow(0.5)
    return sig.rename(f"sigmaYZp_{w}")


def drift_tilt(df: pd.DataFrame, w: int, sig: pd.Series) -> pd.Series:
    mu = np.log(df["close"]).diff().rolling(w).mean()
    tilt = (mu / (sig.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
    return tilt.rename(f"tiltYZp_{w}")


# ====== PILLAR 5: RESIDUAL MOM/REV (ROBUST Z OF RETURNS) ========
def robust_residual_z(r: pd.Series, w: int) -> pd.Series:
    med = r.rolling(w, min_periods=w // 2).median()
    mad = rolling_mad(r, w, w // 2)
    z = (r - med) / (1.4826 * mad.replace(0, np.nan))
    return z.rename(f"residZ_{w}")


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
    spread_bps = df.get("spread_bps", pd.Series(1.0, index=df.index))
    spread_ok = (spread_bps / 1e4) <= S_THR
    return pd.DataFrame({"gap_ok": gap_ok, "vol_mult": vol_mult, "spread_ok": spread_ok})


# =================== META-SCORE AND FORECAST ====================
def meta_score(df: pd.DataFrame) -> pd.DataFrame:
    # Components standardized to z and bounded
    # Trend slope via M_ref: difference today vs yesterday (low-lag slope)
    slope = (df["M_ref"] - df["M_ref"].shift(1)) / (df["M_ref"].shift(1))
    slope_z = (slope - slope.rolling(50).mean()) / slope.rolling(50).std().replace(0, np.nan)
    slope_z = slope_z.clip(-5, 5).rename("trend_slope_z")

    # Regime pressure: polarity with hysteresis already encoded in df["regime"]
    reg_pressure = df["polarity"].clip(-3, 3).rename("reg_pressure")

    # Path geometry summary
    sig_mean = df["sig2_meanz"].clip(-4, 4) if "sig2_meanz" in df else pd.Series(0, index=df.index)
    sig_energy = df["sig2_energy"].clip(-4, 4) if "sig2_energy" in df else pd.Series(0, index=df.index)

    # Volatility tilt
    tilt = df.filter(like="tiltYZp_").iloc[:, 0].clip(-4, 4).rename("tilt")

    # Residual momentum/reversion
    r = df["close"].pct_change()
    rz = robust_residual_z(r, 20).clip(-5, 5)

    # Weighted fusion -> meta score in [-1,1] via tanh
    # Weights tilted to trend/regime; path and tilt adjust conviction; rz catches micro-structure
    z = 0.35 * slope_z.fillna(0) + 0.30 * reg_pressure.fillna(0) + 0.15 * sig_mean.fillna(0) + 0.10 * tilt.fillna(0) + 0.10 * rz.fillna(0)
    meta = np.tanh(z).rename("meta_score")

    # Map to probability P(up) in [0,1]
    p_up = (meta + 1.0) / 2.0
    return pd.DataFrame({"trend_slope_z": slope_z, "meta_score": meta, "p_up": p_up})


def compute_direction(df: pd.DataFrame) -> pd.Series:
    # Direction from regime + close vs M_ref, as agreed
    above = df["close"] > df["M_ref"]
    f = pd.Series(0, index=df.index, dtype=int, name="direction_raw")
    f[(df["regime"] == +1) & above] = -1
    f[(df["regime"] == +1) & ~above] = +1
    f[(df["regime"] == -1) & above] = +1
    f[(df["regime"] == -1) & ~above] = -1
    return f


def size_from_meta(df: pd.DataFrame) -> pd.Series:
    # Base from polarity dead-zone; throttle by exhaustion and vol
    TAU_IN, TAU_OUT = P["TAU_IN"], P["TAU_OUT"]
    strength = ((df["polarity"].abs() - TAU_OUT) / max(TAU_IN - TAU_OUT, 1e-6)).clip(0, 1)
    z = df["dist_z"].abs()
    ex_mult = np.where(z >= 2.5, 0.0, np.where(z >= 1.5, 0.5, 1.0))
    # Blend with predictive probability
    conv = (2 * df["p_up"] - 1.0).abs()  # |2p-1|
    s = (0.7 * strength + 0.3 * conv) * ex_mult
    return pd.Series(s, index=df.index, name="size_mult")


def apply_cooldown(direction: pd.Series, intent: pd.Series, n_bars: int) -> tuple[pd.Series, pd.Series]:
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
    filt = df["gap_ok"] & df["spread_ok"]
    intent[~filt] = 0.0
    intent = intent * df["vol_mult"]
    intent, cd_flag = apply_cooldown(direction, intent, P["COOLDOWN"])

    out = pd.DataFrame(index=df.index)
    out["signal_next_open"] = np.sign(intent.fillna(0)).astype(int)
    out["size_mult_next_open"] = intent.abs().clip(0, 1).fillna(0.0)
    out["stop_multiple_next"] = P["STOP_MULT"]
    out["time_exit_bars_next"] = P["TIME_EXIT"]
    out["p_up_next"] = df["p_up"].clip(0, 1)  # predictive probability

    # Audit trail
    out["above"] = (df["close"] > df["M_ref"]).astype(int)
    out["regime_state"] = df["regime"].astype(int)
    out["direction_raw"] = direction.astype(int)
    out["gate_gap"] = (~(df["gap_ok"])).astype(int)
    out["gate_spread"] = (~(df["spread_ok"])).astype(int)
    out["gate_exhaust"] = (df["dist_z"].abs() >= 2.5).astype(int)
    out["gate_vol_half"] = (df["vol_mult"] < 1.0).astype(int)
    out["cooldown_flag"] = cd_flag.astype(int)

    return out


# ============================= MAIN ============================
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        df, _ = load_backtest_prices(CSV_PATH)
        df = df.copy()
    except Exception as e:
        logging.exception(f"Failed to load data: {e}")
        return

    # Core panels
    pr = compute_polarity(df, P["WIN_PI"])
    means = build_means(df["close"])
    ex = compute_exhaustion(df["close"], means["M_ref"])
    reg = regime_series(pr["polarity"]).to_frame()
    # Path features over a compact window
    sig2 = path_sig2(df[["open", "high", "low", "close"]], w=max(5, P["RMA"]))
    # Vol squeeze tilt
    sigYZ = yz_vol_proxy(df, w=max(10, P["HMA"]))
    tilt = drift_tilt(df, w=max(10, P["HMA"]), sig=sigYZ).to_frame()
    filters = exec_filters(df)

    panel = df[["open", "high", "low", "close"]].join([pr, means, ex, reg, sig2, sigYZ, tilt, filters], how="outer")

    # Meta score and probability
    meta = meta_score(panel)
    panel = panel.join(meta)

    # Forward-only signal
    forward = forward_signal(panel)

    # Warmup
    warmup = max(P["WIN_PI"], P["RMA"], P["HMA"], P["DEMA"], P["MED"], P["HMA"] // 2, int(np.sqrt(P["HMA"])), max(5, P["RMA"]), max(10, P["HMA"])) + 10
    panel = panel.iloc[warmup:].copy()
    forward = forward.iloc[warmup:].copy()

    # Plot
    try:
        sns.set_style("whitegrid")
        fig, (axp, axu, axm) = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

        # Price + means
        r_len, h_len, d_span, m_len = P["RMA"], P["HMA"], P["DEMA"], P["MED"]
        axp.plot(panel.index, panel["close"], color="black", lw=1.2, label="Close")
        axp.plot(panel.index, panel["M_ref"], color="tab:orange", lw=1.1, label="M_ref")
        axp.plot(panel.index, panel[f"RMA{r_len}"], lw=0.8, alpha=0.7, label=f"RMA{r_len}")
        axp.plot(panel.index, panel[f"HMA{h_len}"], lw=0.8, alpha=0.7, label=f"HMA{h_len}")
        axp.plot(panel.index, panel[f"DEMA{d_span}"], lw=0.8, alpha=0.7, label=f"DEMA{d_span}")
        axp.plot(panel.index, panel[f"Median{m_len}"], lw=0.8, alpha=0.7, label=f"Median{m_len}")
        # Markers: next-bar signal decided at t
        sig = forward["signal_next_open"].reindex(panel.index).fillna(0).astype(int)
        longs = panel[sig == 1]
        shorts = panel[sig == -1]
        axp.scatter(longs.index, longs["close"], marker="^", s=28, color="green", label="Long")
        axp.scatter(shorts.index, shorts["close"], marker="v", s=28, color="red", label="Short")
        axp.set_ylabel("Price")
        axp.legend(loc="upper left")

        # Show Close - M_ref explicitly (your complaint)
        axu.plot(panel.index, (panel["close"] - panel["M_ref"]), lw=1.0, label="Close - M_ref")
        axu.axhline(0, ls="--", color="grey", lw=1.0)
        axu.set_ylabel("Close−M_ref")
        axu.legend(loc="upper left")

        # Predictive probability of up move
        axm.plot(panel.index, forward["p_up_next"].reindex(panel.index), lw=1.0, label="P(up) next bar")
        axm.axhline(0.5, ls="--", color="grey", lw=1.0)
        axm.set_ylabel("P(up)")
        axm.set_ylim(0, 1)
        axm.legend(loc="upper left")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.exception(f"Plotting failed: {e}")

    # Exports
    try:
        forward.to_csv("universal_price_only_signal_next_open.csv")
        keep = [
            "close",
            "polarity_raw",
            "polarity",
            "regime",
            f"RMA{r_len}",
            f"HMA{h_len}",
            f"DEMA{d_span}",
            f"Median{m_len}",
            "M_ref",
            "dist",
            "dist_z",
            "sig2_meanz",
            "sig2_energy",
            sigYZ.name,
            "tiltYZp_" + str(max(10, P["HMA"])),
            "meta_score",
            "p_up",
        ]
        panel[[c for c in keep if c in panel.columns]].to_csv("universal_price_only_panel.csv")
    except Exception as e:
        logging.exception(f"Export failed: {e}")

    # Print final next-bar signal snapshot
    if len(forward):
        snap = forward.iloc[-1]
        print("\n=== NEXT-BAR SIGNAL ===")
        print({
            "mode": MODE,
            "signal_next_open": int(snap["signal_next_open"]),
            "size_mult_next_open": np.float64(snap["size_mult_next_open"]),
            "p_up_next": np.float64(snap["p_up_next"]),
            "stop_multiple_next": np.float64(snap["stop_multiple_next"]),
            "time_exit_bars_next": int(snap["time_exit_bars_next"]),
        })


if __name__ == "__main__":
    main()
