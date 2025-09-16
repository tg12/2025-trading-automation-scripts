"""Indicator lab for price-only experimental signals.

Loads ``backtest_prices.csv`` via ``backtest_loader`` and computes a suite of
prototype quantitative indicators inspired by leading systematic trading firms.
Each indicator produces a time series signal, the latest score, and short
context text describing interpretation.  This script focuses on clarity and
robustness instead of optimisation.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from backtest_loader import BacktestCSVLoader
from numpy.typing import ArrayLike
from scipy import linalg, signal


@dataclass
class IndicatorResult:
    name: str
    latest_signal: float
    series: pd.Series
    description: str
    suggested_plot: str
    direction: int  # -1 short, 0 neutral, 1 long


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def zscore(series: pd.Series, window: int = 50) -> pd.Series:
    r = series.rolling(window)
    return (series - r.mean()) / (r.std(ddof=1) + 1e-8)


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


def fractional_difference(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    x = series.ffill().to_numpy(dtype=float)
    w = [1.0]
    k = 1
    while True:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < thresh:
            break
        w.append(w_next)
        k += 1
    w = np.asarray(w, float)
    y = np.empty_like(x)
    for t in range(len(x)):
        i0 = max(0, t - (len(w) - 1))
        ws = w[: t - i0 + 1][::-1]
        y[t] = np.dot(x[i0 : t + 1], ws)
    return pd.Series(y, index=series.index)


def tv_l1_trend_filter(y: ArrayLike, lam: float, rho: float = 2.0, max_iter: int = 500, tol: float = 1e-5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= 2:
        return y.copy()

    x = y.copy()
    z = np.zeros(n - 1)
    u = np.zeros(n - 1)

    diag = np.ones(n) + rho * np.concatenate(([1], np.repeat(2.0, n - 2), [1]))
    off_diag = -rho * np.ones(n - 1)

    for _ in range(max_iter):
        z_minus_u = z - u
        rhs = y.copy()
        if n > 1:
            rhs[0] += rho * (z_minus_u[0])
            rhs[-1] -= rho * (z_minus_u[-1])
        if n > 2:
            for i in range(1, n - 1):
                rhs[i] += rho * (z_minus_u[i] - z_minus_u[i - 1])

        x_prev = x.copy()
        x = _solve_tridiagonal(diag, off_diag, rhs)

        dx = np.diff(x)
        z = soft_threshold(dx + u, lam / rho)
        u += dx - z

        if np.linalg.norm(x - x_prev) / (np.linalg.norm(x_prev) + 1e-8) < tol:
            break
    return x


def _solve_tridiagonal(diag: np.ndarray, off_diag: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    ab = np.zeros((3, diag.size))
    ab[0, 1:] = off_diag
    ab[1, :] = diag
    ab[2, :-1] = off_diag
    return linalg.solve_banded((1, 1), ab, rhs)


def huber_mean(y: pd.Series, delta: float = 1.5, max_iter: int = 50) -> pd.Series:
    mean = y.rolling(window=5, min_periods=1).median()
    scale = y.rolling(window=20, min_periods=1).std().fillna(y.std())
    scale = scale.replace(0, scale.median() or 1.0)
    estimate = mean.copy()
    for _ in range(max_iter):
        residuals = y - estimate
        weights = residuals.abs() <= delta * scale
        weights = weights.astype(float) * 1.0
        outliers = ~weights.astype(bool)
        weights[outliers] = (delta * scale[outliers]).div(residuals[outliers].abs() + 1e-8)
        estimate = (weights * y).rolling(window=5, min_periods=1).sum() / (weights.rolling(window=5, min_periods=1).sum() + 1e-8)
    return estimate


def median_of_means(series: pd.Series, window: int, groups: int) -> pd.Series:
    def _mom(x: np.ndarray) -> float:
        splits = np.array_split(x, groups)
        means = [np.mean(s) for s in splits if len(s) > 0]
        return float(np.median(means)) if means else np.nan

    return series.rolling(window=window, min_periods=window).apply(_mom, raw=True)


def rogers_satchell_vol(df: pd.DataFrame, window: int = 14) -> pd.Series:
    term = np.log(df["high"] / df["close"].shift(1)) * np.log(df["high"] / df["open"]) + np.log(df["low"] / df["close"].shift(1)) * np.log(df["low"] / df["open"])
    return term.rolling(window).mean().clip(lower=1e-8).pow(0.5)


def kalman_level_slope(log_price: pd.Series, process_var: float = 1e-4, obs_var: float = 1e-3) -> Tuple[pd.Series, pd.Series]:
    level = np.zeros_like(log_price.values)
    slope = np.zeros_like(log_price.values)
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = process_var * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])
    H = np.array([[1.0, 0.0]])

    state = np.array([log_price.iloc[0], 0.0])
    for i, obs in enumerate(log_price.values):
        state = F @ state
        P = F @ P @ F.T + Q
        y = obs - H @ state
        S = H @ P @ H.T + obs_var
        K = (P @ H.T) / S
        state = state + (K.flatten() * y)
        P = (np.eye(2) - K @ H) @ P
        level[i] = state[0]
        slope[i] = state[1]
    return pd.Series(level, index=log_price.index), pd.Series(slope, index=log_price.index)


def higuchi_fd(series: pd.Series, k_max: int = 6) -> pd.Series:
    out = np.full(len(series), np.nan, dtype=float)
    for t in range(k_max * 2, len(series)):
        subseq = series.iloc[: t + 1].to_numpy(dtype=float)
        Lk = []
        ks = []
        for k in range(1, k_max + 1):
            lengths = []
            for m in range(k):
                pts = subseq[m::k]
                if pts.size < 2:
                    continue
                length = np.abs(np.diff(pts)).sum() * (len(subseq) - 1) / (k * len(pts))
                lengths.append(length)
            if lengths:
                Lk.append(np.log(np.mean(lengths)))
                ks.append(np.log(k))
        if len(ks) >= 2:
            slope, _ = np.polyfit(ks, Lk, 1)
            out[t] = slope
    return pd.Series(out, index=series.index)


def logistic(x: pd.Series) -> pd.Series:
    return 1 / (1 + np.exp(-x))


# ---------------------------------------------------------------------------
# Indicator implementations
# ---------------------------------------------------------------------------


def renaissance_tv_l1(df: pd.DataFrame) -> IndicatorResult:
    log_close = np.log(df["close"])
    frac = fractional_difference(log_close, d=0.4)
    mu = pd.Series(tv_l1_trend_filter(frac.values, lam=5.0), index=df.index)
    delta_mu = mu.diff()
    signal_series = zscore(delta_mu, window=50)
    last = signal_series.iloc[-1]
    return IndicatorResult(
        name="Renaissance – TV-L1 Fractional Trend",
        latest_signal=float(last),
        series=signal_series,
        description="Trend pressure via fractional differencing followed by TV-L1 denoising.",
        suggested_plot="Dual-axis: log-price, TV-L1 trend, and z-scored delta trend histogram.",
        direction=1 if last > 0 else -1 if last < 0 else 0,
    )


def two_sigma_mom(df: pd.DataFrame) -> IndicatorResult:
    k = 10
    returns = df["close"].pct_change(k)
    huber_fit = huber_mean(returns.fillna(0.0))
    residual = returns - huber_fit
    mom = median_of_means(residual.fillna(0.0), window=60, groups=5)
    signal_series = zscore(mom.fillna(0.0), window=30)
    return IndicatorResult(
        name="Two Sigma – Median-of-Means Residual Momentum",
        latest_signal=float(signal_series.iloc[-1]),
        series=signal_series,
        description="Robust residual momentum using Huber residuals and median-of-means stabilisation.",
        suggested_plot="Rolling median-of-means residual vs price with shaded z-score bands.",
        direction=1 if signal_series.iloc[-1] > 0 else -1 if signal_series.iloc[-1] < 0 else 0,
    )


def de_shaw_range_normalised(df: pd.DataFrame) -> IndicatorResult:
    log_price = np.log(df["close"])
    level, slope = kalman_level_slope(log_price)
    rs_vol = rogers_satchell_vol(df)
    signal_series = (slope / rs_vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    last = signal_series.iloc[-1]
    return IndicatorResult(
        name="D. E. Shaw – RS-Normalised Kalman Drift",
        latest_signal=float(last),
        series=signal_series,
        description="Kalman slope scaled by Rogers–Satchell volatility to flag drift strength.",
        suggested_plot="Slope/volatility ratio heatmap overlaid beneath log-price line chart.",
        direction=1 if last > 0 else -1 if last < 0 else 0,
    )


def jane_street_polarity(df: pd.DataFrame) -> IndicatorResult:
    log_c, log_o = np.log(df["close"]), np.log(df["open"])
    r_oc = log_c - log_o  # intraday
    r_co = log_o - log_c.shift(1)  # gap
    window = 20
    mu_oc = r_oc.rolling(window).mean()
    mu_co = r_co.rolling(window).mean()
    var_sum = r_oc.rolling(window).var(ddof=1) + r_co.rolling(window).var(ddof=1)
    pi = (mu_oc - mu_co) / (np.sqrt(var_sum) + 1e-8)
    last = pi.iloc[-1]
    return IndicatorResult(
        name="Jane Street – Overnight/Intraday Polarity",
        latest_signal=float(last),
        series=pi,
        description="Positive ⇒ intraday dominates ⇒ momentum bias; negative ⇒ gap dominates ⇒ mean-revert bias.",
        suggested_plot="Rolling means of r_OC and r_CO with PI overlay.",
        direction=1 if last > 0 else -1 if last < 0 else 0,
    )


def jump_hayashi_yoshida(df: pd.DataFrame) -> IndicatorResult:
    log_c, log_o = np.log(df["close"]), np.log(df["open"])
    r_oc = log_o - log_c.shift(1)  # gap
    r_co = log_c - log_o  # intraday
    aligned = pd.DataFrame({"overnight": r_oc, "intraday": r_co}).dropna()
    cov = (aligned["overnight"] * aligned["intraday"]).rolling(30).mean()
    variance = aligned["intraday"].rolling(30).var(ddof=1)
    signal_series = (cov / (variance + 1e-8)).rename("overnight→intraday beta")
    last = signal_series.iloc[-1]
    return IndicatorResult(
        name="Jump Trading – Overnight/Intraday Cov Beta (proxy)",
        latest_signal=float(last),
        series=signal_series,
        description="Same-asset gap/intraday covariance divided by intraday variance; lead–lag proxy.",
        suggested_plot="Scatter of overnight vs intraday returns with rolling beta.",
        direction=1 if last > 0 else -1 if last < 0 else 0,
    )


def hrt_ssa(df: pd.DataFrame) -> IndicatorResult:
    r = df["close"].pct_change().fillna(0.0)
    L = 20
    if len(r) < L:
        raise ValueError("Not enough observations for SSA window")

    # Trajectory with past-only lags
    traj = pd.concat([r.shift(i) for i in range(L)], axis=1).dropna()  # columns: t, t-1, ..., t-(L-1)
    X = traj.to_numpy()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    comp1 = X @ Vt[0]
    comp2 = X @ Vt[1] if len(s) > 1 else np.zeros_like(comp1)

    idx = traj.index
    slope = (
        pd.Series(comp1, index=idx)
        .rolling(10)
        .apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0],
            raw=True,
        )
    )
    curvature = (
        pd.Series(comp2, index=idx)
        .rolling(10)
        .apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 2)[0],
            raw=True,
        )
    )

    score = logistic(((slope.fillna(0.0) - curvature.fillna(0.0)) * 10.0).clip(-10, 10))
    aligned = score.reindex(df.index)
    return IndicatorResult(
        name="HRT – SSA Turning-Point Probability",
        latest_signal=float(aligned.iloc[-1]),
        series=aligned,
        description="Causal SSA (past-only) with slope−curvature logistic map.",
        suggested_plot="Probability line with >0.6 and <0.4 shading.",
        direction=1 if aligned.iloc[-1] > 0.5 else -1 if aligned.iloc[-1] < 0.5 else 0,
    )


def xtx_signature(df: pd.DataFrame) -> IndicatorResult:
    window = 15
    norm = df[["open", "high", "low", "close"]].div(
        df["close"].rolling(window).mean(),
        axis=0,
    )
    increments = norm.diff().dropna()
    primary = increments.rolling(window).sum().sum(axis=1)

    secondary = pd.Series(0.0, index=increments.index)
    for lag in range(1, window):
        lag_prod = (increments * increments.shift(lag)).sum(axis=1)
        secondary += lag_prod.fillna(0.0)
    secondary /= max(window - 1, 1)

    logits = (primary + 0.5 * secondary).fillna(0.0)
    prob = logistic(logits)
    aligned = prob.reindex(df.index)
    return IndicatorResult(
        name="XTX – Depth-2 Path Signature Logit",
        latest_signal=float(aligned.iloc[-1]),
        series=aligned,
        description="Approximated signature features converted to direction probability.",
        suggested_plot="Signature probability heatmap over OHLC candlesticks.",
        direction=1 if aligned.iloc[-1] > 0.5 else -1 if aligned.iloc[-1] < 0.5 else 0,
    )


def optiver_yz_breakout(df: pd.DataFrame) -> IndicatorResult:
    o, h, l, c = [np.log(df[x]) for x in ("open", "high", "low", "close")]
    k = 20
    r_o = o - c.shift(1)  # open to prev close (overnight)
    r_c = c - o  # close to open (intraday)
    rs = np.log(h / o) * np.log(h / c) + np.log(l / o) * np.log(l / c.shift(1))  # RS with prev close
    sigma_o2 = r_o.rolling(k).var(ddof=1)
    sigma_c2 = r_c.rolling(k).var(ddof=1)
    sigma_rs = rs.rolling(k).mean()
    alpha = 0.34 / (1.34 + (k + 1) / max(k - 1, 1))
    sigma_yz = (sigma_o2 + alpha * sigma_c2 + (1 - alpha) * sigma_rs).clip(lower=1e-12).pow(0.5)
    drift = r_c.rolling(k).mean()
    signal_series = drift.abs() / (sigma_yz + 1e-8)
    last = signal_series.iloc[-1]
    return IndicatorResult(
        name="Optiver – Yang–Zhang Drift Tilt",
        latest_signal=float(last),
        series=signal_series,
        description="|E[intraday]| scaled by Yang–Zhang σ; high values tilt toward breakout.",
        suggested_plot="Histogram of |drift|/σYZ with threshold line and price overlay.",
        direction=1 if last > 0 else -1 if last < 0 else 0,
    )


def imc_keltner_toggle(df: pd.DataFrame) -> IndicatorResult:
    hl_range = (df["high"] - df["low"]).ewm(span=20, adjust=False).mean()
    ema = df["close"].ewm(span=20, adjust=False).mean()
    upper = ema + 1.5 * hl_range
    lower = ema - 1.5 * hl_range
    pct = (df["close"] - lower) / (upper - lower + 1e-8)
    expansion = (upper - lower).pct_change().fillna(0.0)
    signal_series = zscore(pct, window=30) + expansion
    return IndicatorResult(
        name="IMC – Keltner Percentile Toggle",
        latest_signal=float(signal_series.iloc[-1]),
        series=signal_series,
        description="Percentile of price within HL-based Keltner band plus expansion cue.",
        suggested_plot="Price with Keltner bands and percentile oscillator panel.",
        direction=1 if signal_series.iloc[-1] > 0 else -1 if signal_series.iloc[-1] < 0 else 0,
    )


def drw_fractal_drift(df: pd.DataFrame) -> IndicatorResult:
    log_price = np.log(df["close"])
    fd = higuchi_fd(log_price)
    slope = log_price.diff().rolling(20).mean()
    trend_pressure = (2 - fd).fillna(0.0) * np.sign(slope.fillna(0.0))
    return IndicatorResult(
        name="DRW – Fractal Dimension Drift",
        latest_signal=float(trend_pressure.iloc[-1]),
        series=trend_pressure,
        description="Trend pressure from Higuchi fractal dimension combined with slope sign.",
        suggested_plot="Fractal dimension vs price with zones under 1.5 and above 1.5 highlighted.",
        direction=1 if trend_pressure.iloc[-1] > 0 else -1 if trend_pressure.iloc[-1] < 0 else 0,
    )


def cubist_mixer(df: pd.DataFrame) -> IndicatorResult:
    returns = df["close"].pct_change().fillna(0.0)
    horizon = 5
    features = []
    for lag in range(1, horizon + 1):
        features.append(returns.shift(lag))
        features.append(returns.shift(lag).abs().rank(pct=True))
    hl_range_pct = (df["high"] - df["low"]) / df["close"]
    features.append(hl_range_pct)
    X = pd.concat(features, axis=1).fillna(0.0)
    y = np.sign(returns.shift(-1).fillna(0.0))
    ridge = np.linalg.pinv(X.T @ X + 0.1 * np.eye(X.shape[1])) @ X.T @ y
    signal_series = pd.Series(X.values @ ridge, index=df.index)
    return IndicatorResult(
        name="Cubist – Nonlinear Reversal/Momentum Mixer",
        latest_signal=float(signal_series.iloc[-1]),
        series=signal_series,
        description="Quadratic-like ridge combiner of returns, ranks, and range percentage.",
        suggested_plot="Panel showing mixer score vs realised next-day direction scatter.",
        direction=1 if signal_series.iloc[-1] > 0 else -1 if signal_series.iloc[-1] < 0 else 0,
    )


def millennium_bocpd(df: pd.DataFrame) -> IndicatorResult:
    r = df["close"].pct_change()
    short = r.rolling(5).mean()
    long = r.rolling(40).mean()
    vol = r.rolling(30).std(ddof=1).replace(0, np.nan)
    score = (short - long) / (vol + 1e-6)
    posterior = logistic(score.clip(-5, 5))
    expectation = posterior * short - (1 - posterior) * short
    last = expectation.iloc[-1]
    return IndicatorResult(
        name="Millennium – BOCPD Expected Return",
        latest_signal=float(last),
        series=expectation,
        description="Logistic-gated momentum proxy; causal, no forward fill.",
        suggested_plot="Posterior and expectation bands.",
        direction=1 if last > 0 else -1 if last < 0 else 0,
    )


def pdt_sparse_motif(df: pd.DataFrame) -> IndicatorResult:
    r = df["close"].pct_change().to_numpy(dtype=float)
    window = 20
    if len(r) <= window:
        raise ValueError("Not enough samples for motif window")
    patches = np.lib.stride_tricks.sliding_window_view(r, window_shape=window)  # shape: (N-window+1, window)
    y = r[window:]  # next-step after each patch start+window-1
    # SVD dictionary (top-K atoms)
    try:
        U, s, Vt = np.linalg.svd(patches, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD did not converge, use identity
        Vt = np.eye(window)
    K = 5
    D = Vt[:K].T  # (window, K)
    coeffs = patches @ D  # (N-window+1, K)
    coeffs = coeffs[: len(y)]  # trim to match y
    try:
        w, *_ = np.linalg.lstsq(coeffs, y, rcond=None)
    except np.linalg.LinAlgError:
        w = np.zeros(K)
    signal = coeffs @ w  # aligned to patch end
    idx = df.index[window : window + len(signal)]  # causal alignment
    series = pd.Series(signal, index=idx).reindex(df.index)
    return IndicatorResult(
        name="PDT – Sparse Motif Dictionary",
        latest_signal=float(series.dropna().iloc[-1]),
        series=series,
        description="Causal motif activations mapped to next-step forecast; no forward fill.",
        suggested_plot="Motif activation-derived forecast overlay.",
        direction=1 if series.dropna().iloc[-1] > 0 else -1 if series.dropna().iloc[-1] < 0 else 0,
    )


def aqr_bridge_momentum(df: pd.DataFrame) -> IndicatorResult:
    lp = np.log(df["close"])
    k = 40
    steps = range(5, k + 1, 5)  # 5,10,...,40
    bridges = [lp - lp.shift(h) for h in steps]  # past-only k-day bridges
    B = pd.concat(bridges, axis=1)
    w = np.linspace(1.0, 0.2, B.shape[1])  # heavier recent bridge
    num = B.fillna(0.0) @ w
    den = np.sqrt(B.var(ddof=1).sum() + 1e-8)
    s = (num / den).rename("bridge_mom")
    s = s.reindex(df.index).fillna(0.0)
    return IndicatorResult(
        name="AQR – Bridge-Overlap Momentum",
        latest_signal=float(s.iloc[-1]),
        series=s,
        description="Overlapping past bridges, variance-normalized; positive ⇒ long bias.",
        suggested_plot="Ribbon of bridges with aggregate momentum curve.",
        direction=1 if s.iloc[-1] > 0 else -1 if s.iloc[-1] < 0 else 0,
    )


def g_research_wavelet(df: pd.DataFrame) -> IndicatorResult:
    r = df["close"].pct_change().fillna(0.0).values
    widths = np.arange(1, 16)
    try:
        cwt = signal.cwt(r, signal.ricker, widths)  # (scales, time)
    except AttributeError:
        kernels = [np.exp(-(np.linspace(-2, 2, w) ** 2) / 2) for w in widths]
        cwt = np.array([np.convolve(r, k, mode="same") for k in kernels])
    energy = (np.abs(cwt) ** 2).mean(axis=0)  # time series
    series = pd.Series((energy - energy.mean()) / (energy.std() + 1e-8), index=df.index)
    prob = logistic(series.clip(-8, 8))
    last = prob.iloc[-1]
    return IndicatorResult(
        name="G-Research – Wavelet Scattering Classifier",
        latest_signal=float(last),
        series=prob,
        description="Scale-averaged CWT energy mapped to probability.",
        suggested_plot="Energy time series with probability overlay.",
        direction=1 if last > 0.5 else -1 if last < 0.5 else 0,
    )


INDICATOR_FUNCS: Iterable[Callable[[pd.DataFrame], IndicatorResult]] = (
    renaissance_tv_l1,
    two_sigma_mom,
    de_shaw_range_normalised,
    jane_street_polarity,
    jump_hayashi_yoshida,
    hrt_ssa,
    xtx_signature,
    optiver_yz_breakout,
    imc_keltner_toggle,
    drw_fractal_drift,
    cubist_mixer,
    millennium_bocpd,
    pdt_sparse_motif,
    aqr_bridge_momentum,
    g_research_wavelet,
)


def load_prices() -> pd.DataFrame:
    loader = BacktestCSVLoader()
    df, summary = loader.load()
    logger.info("Loaded %s", summary.as_dict())
    return df


def run_all(df: pd.DataFrame) -> List[IndicatorResult]:
    results: List[IndicatorResult] = []
    for func in INDICATOR_FUNCS:
        try:
            res = func(df)
            results.append(res)
        except Exception as exc:
            logger.error("Indicator %s failed: %s", func.__name__, exc)
    return results


def compile_report(results: Iterable[IndicatorResult]) -> pd.DataFrame:
    summary_rows = []
    for res in results:
        summary_rows.append({
            "Indicator": res.name,
            "LatestSignal": res.latest_signal,
            "Description": res.description,
            "SuggestedPlot": res.suggested_plot,
        })
    return pd.DataFrame(summary_rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    df = load_prices()
    results = run_all(df)
    report = compile_report(results)
    pd.set_option("display.max_colwidth", 120)
    logger.info("Indicator Report:\n%s", report)


if __name__ == "__main__":
    main()
