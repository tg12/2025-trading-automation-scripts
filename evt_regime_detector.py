#!/usr/bin/env python3
"""tails_regime_opt.py
Tail-event detection from price data using Extreme Value Theory (EVT) + regime classification
with Optuna hyperparameter optimization and diagnostic plots.

================================================================================
                              ATTRIBUTION
================================================================================
If you use this code in any project, research, or publication, please provide
attribution by citing:

Author: James Sawyer
GitHub: https://github.com/tg12

================================================================================
                            LEGAL DISCLAIMER
================================================================================
THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LOSSES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

THIS SOFTWARE IS FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY. IT IS NOT
INTENDED TO PROVIDE, AND SHOULD NOT BE RELIED UPON FOR, INVESTMENT ADVICE.

PAST PERFORMANCE IS NOT INDICATIVE OF FUTURE RESULTS. TRADING AND INVESTING
IN FINANCIAL INSTRUMENTS INVOLVES SUBSTANTIAL RISK OF LOSS. YOU MAY LOSE
SOME OR ALL OF YOUR CAPITAL.

NO REPRESENTATION IS BEING MADE THAT ANY ACCOUNT WILL OR IS LIKELY TO ACHIEVE
PROFITS OR LOSSES SIMILAR TO THOSE DISCUSSED IN THIS SOFTWARE.

THE USER ASSUMES FULL RESPONSIBILITY FOR ANY INVESTMENT DECISIONS MADE BASED
ON THE USE OF THIS SOFTWARE. ALWAYS CONDUCT YOUR OWN RESEARCH AND CONSULT
WITH A QUALIFIED FINANCIAL ADVISOR BEFORE MAKING ANY INVESTMENT DECISIONS.

BY USING THIS SOFTWARE, YOU ACKNOWLEDGE THAT YOU HAVE READ THIS DISCLAIMER
AND AGREE TO BE BOUND BY ITS TERMS.
================================================================================

METHODOLOGY NOTES:
- Walk-forward analysis: GPD fits use only historical data up to each point
- Realistic trading simulation: enters at next bar's price with slippage
- Regime classification: uses current bar stats, but trading uses lagged regime
- No lookahead bias in event detection or strategy execution
- Optuna optimization with train/test split for validation

Data source: from backtest_loader import load_backtest_prices  # returns (df, meta)
"""

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import genpareto

    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import optuna
except Exception as e:
    raise RuntimeError(
        "Optuna is required. Install with: pip install optuna",
    ) from e

from backtest_loader import load_backtest_prices  # returns (df, meta)

# =====================================================================
#                        UTILITY FUNCTIONS
# =====================================================================


def _infer_price_series(df: pd.DataFrame) -> pd.Series:
    """Extract price series from DataFrame.

    Prefers typical price column names (close, price, settle, last),
    otherwise uses first numeric column. Cleans infinities and NaNs.

    Args:
        df: Input DataFrame with price data

    Returns:
        Clean price series with name="price"

    """
    candidates = [c for c in df.columns if c.lower() in ("close", "price", "settle", "last")]
    if candidates:
        s = df[candidates[0]].astype(np.float64)
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        assert num_cols, "No numeric columns found in df"
        s = df[num_cols[0]].astype(np.float64)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s.name = "price"
    return s


def _as_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame to DatetimeIndex if possible.

    Attempts to parse date column if index is not already DatetimeIndex.
    Falls back to RangeIndex if no date column found.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with DatetimeIndex (if possible) sorted by time

    """
    if not isinstance(df.index, pd.DatetimeIndex):
        date_cols = [c for c in df.columns if c.lower() in ("date", "datetime", "timestamp")]
        if date_cols:
            dt = pd.to_datetime(df[date_cols[0]], errors="coerce", utc=True).tz_convert(None)
            out = df.drop(columns=[date_cols[0]]).copy()
            out.index = dt
            df = out
    return df.sort_index()


def log_returns(price: pd.Series) -> pd.Series:
    """Calculate log returns: ln(P_t / P_{t-1}).

    Log returns are preferred for statistical modeling as they are
    approximately normal and time-additive.
    """
    return np.log(price).diff()


def ewma_vol(r: pd.Series, alpha: np.float64 = 0.06, min_periods: int = 20) -> pd.Series:
    """Exponentially Weighted Moving Average (EWMA) volatility estimator.

    Uses RiskMetrics-style variance estimation with decay factor alpha.
    Lower alpha = slower adaptation to new information.

    Args:
        r: Return series
        alpha: Decay factor (0.06 ≈ 94% weight on past)
        min_periods: Minimum observations required

    Returns:
        Volatility (standard deviation) series

    """
    var = r.pow(2).ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()
    v = np.sqrt(var)
    return v.replace(0.0, np.nan)


def scale_returns(r: pd.Series, vol: pd.Series) -> pd.Series:
    """Standardize returns by dividing by volatility.

    Creates approximately unit-variance series for tail event detection.
    This normalization is crucial for GPD fitting across different market regimes.

    Args:
        r: Raw returns
        vol: Volatility estimates

    Returns:
        Volatility-scaled returns (mean ≈ 0, std ≈ 1)

    """
    x = r / vol
    return x.replace([np.inf, -np.inf], np.nan)


def periods_per_year(index: pd.Index) -> np.float64:
    """Infer annualization factor from data frequency.

    Examines median time difference between observations to determine
    whether data is intraday, daily, weekly, or monthly.

    Returns:
        252.0 for daily (or unknown)
        252*390 for intraday (approx minutes per trading year)
        52.0 for weekly
        12.0 for monthly

    """
    if not isinstance(index, pd.DatetimeIndex):
        return 252.0
    diffs = np.diff(index.view(np.int64))  # nanoseconds
    if len(diffs) == 0:
        return 252.0
    med_ns = np.median(diffs)
    day_ns = 86_400_000_000_000
    if med_ns <= 0:
        return 252.0
    per_day = day_ns / med_ns
    # Cap to plausible frequency buckets
    if per_day > 100:  # intraday (minute or second)
        return 252.0 * 390.0  # Approx trading minutes per year
    if 1.5 < per_day < 10:
        return 252.0  # Daily
    if 0.2 < per_day <= 1.5:
        return 52.0  # Weekly
    if per_day <= 0.2:
        return 12.0  # Monthly
    return 252.0


def sharpe_ratio(returns: pd.Series, ppyr: np.float64) -> np.float64:
    """Calculate annualized Sharpe ratio.

    Sharpe = (mean return / std dev of returns) * sqrt(periods per year)

    Returns sentinel value -1e9 if insufficient data or zero volatility.

    Args:
        returns: Return series (not prices)
        ppyr: Periods per year (for annualization)

    Returns:
        Annualized Sharpe ratio, or -1e9 if invalid

    """
    r = returns.dropna()
    if len(r) < 5:  # Need minimum observations
        return -1e9
    mu = r.mean()
    sd = r.std(ddof=0)
    if sd <= 0:  # Avoid division by zero
        return -1e9
    return np.float64((mu / sd) * np.sqrt(ppyr))


# =====================================================================
#          EXTREME VALUE THEORY (EVT) - PEAKS OVER THRESHOLD (POT)
# =====================================================================
# Uses Generalized Pareto Distribution (GPD) to model tail behavior.
# This is the statistical foundation for detecting extreme events.
#
# Key concept: Exceedances over a high threshold follow GPD asymptotically.
# We fit GPD to historical exceedances, then calculate p-values for new observations.
#
# CRITICAL: To avoid lookahead bias, we fit GPD only on data BEFORE each point.


@dataclass
class GPDFit:
    """Container for GPD fit results.

    Attributes:
        u: Threshold value
        frac: Fraction of data exceeding threshold
        nu: Number of exceedances
        n: Total sample size
        survival: Function to compute tail probability P(X > x)

    """

    u: np.float64
    frac: np.float64
    nu: int
    n: int

    def survival(self, z: np.float64) -> np.float64:
        """Tail probability function P(X > u + z). Assigned dynamically."""
        raise NotImplementedError


def fit_gpd_exceedances(x: pd.Series, q: np.float64, side: str) -> GPDFit:
    """Fit Generalized Pareto Distribution to exceedances.

    Uses Peaks-Over-Threshold (POT) method:
    1. Choose threshold u at quantile q (e.g., 95th percentile)
    2. Extract exceedances: values > u (or < u for left tail)
    3. Fit GPD to exceedance sizes
    4. Return fitted distribution for tail probability calculations

    Falls back to empirical distribution if too few exceedances (<30)
    or if scipy not available.

    Args:
        x: Standardized return series
        q: Quantile for threshold (0.95 = 95th percentile)
        side: "right" for positive tail, "left" for negative tail

    Returns:
        GPDFit object with survival probability function

    """
    assert side in ("right", "left")
    x = x.dropna()
    if side == "right":
        u = np.float64(x.quantile(q))
        exc = x[x > u] - u
    else:
        u = np.float64(x.quantile(1 - q))
        exc = (-x[x < u]) - (-u)

    n = len(x)
    nu = len(exc)  # Number of exceedances
    frac = max(nu / max(n, 1), 1e-12)  # Empirical probability of exceeding threshold

    # Use empirical distribution if too few exceedances for reliable GPD fit
    if nu < 30 or not SCIPY_OK:
        exc_sorted = np.sort(np.asarray(exc)) if nu > 0 else np.array([])

        def survival_emp(z: np.float64) -> np.float64:
            """Empirical survival function for small samples."""
            if nu == 0:
                return 1.0
            m = np.searchsorted(exc_sorted, np.float64(max(z, 0.0)), side="right")
            tail_count = nu - m
            return frac * (tail_count / max(nu, 1))

        fit = GPDFit(u=u, frac=frac, nu=nu, n=n)
        fit.survival = survival_emp  # type: ignore
        return fit

    # Fit GPD using Maximum Likelihood Estimation
    c, _loc, scale = genpareto.fit(exc, floc=0.0)  # floc=0 fixes location at 0
    c = np.float64(np.clip(c, -0.49, 0.9))  # Constrain shape parameter for stability
    scale = np.float64(max(scale, 1e-12))  # Prevent numerical issues

    def survival_gpd(z: np.float64) -> np.float64:
        """GPD survival function: P(exceedance > z)."""
        z = np.float64(max(z, 0.0))
        s_ex = 1.0 - genpareto.cdf(z, c, loc=0.0, scale=scale)
        return frac * s_ex  # Scale by empirical tail probability

    fit = GPDFit(u=u, frac=frac, nu=nu, n=n)
    fit.survival = survival_gpd  # type: ignore
    return fit


def tail_pvalue(x_val: np.float64, fit_r: GPDFit, fit_l: GPDFit) -> Tuple[np.float64, str]:
    """Calculate tail p-value for an observation.

    Determines if value is in right tail, left tail, or center,
    then computes probability of observing something more extreme.

    Args:
        x_val: Observed standardized return
        fit_r: GPD fit for right (positive) tail
        fit_l: GPD fit for left (negative) tail

    Returns:
        (p-value, tail_side) where tail_side in {"right", "left", "none"}

    """
    if np.isnan(x_val):
        return np.nan, "none"
    if x_val > fit_r.u:  # Right tail exceedance
        p = fit_r.survival(x_val - fit_r.u)
        return np.float64(p), "right"
    if x_val < fit_l.u:  # Left tail exceedance
        p = fit_l.survival((-x_val) - (-fit_l.u))
        return np.float64(p), "left"
    return 1.0, "none"  # In the center - not extreme


def tail_score(p: np.float64) -> np.float64:
    """Convert p-value to "tail score" using -log10(p).

    Makes small p-values more interpretable:
    - p=0.05 → score=1.3
    - p=0.01 → score=2.0 (1 in 100)
    - p=0.001 → score=3.0 (1 in 1000)
    - p=0.0001 → score=4.0 (1 in 10,000)

    Higher scores = more extreme events.
    """
    p = max(min(np.float64(p), 1.0), 1e-16)  # Clamp to valid range
    return -math.log10(p)


# =====================================================================
#                      REGIME CLASSIFICATION
# =====================================================================
# Identifies market regimes based on rolling higher moments (skewness, kurtosis)
# and drawdown metrics. Used to contextualize tail events.
#
# KEY: Regime is inferred using CURRENT bar (real-time), but trading decisions
# use LAGGED regime (1 bar delay) to avoid lookahead bias.


def rolling_moments(x: pd.Series, window: int = 63) -> pd.DataFrame:
    """Calculate rolling skewness and kurtosis for regime detection.

    INCLUDES current point for real-time regime inference.
    Trading will use lagged regime to avoid lookahead.

    Skewness:
        < 0: Left tail risk (crash-prone)
        > 0: Right tail potential (melt-up)

    Kurtosis:
        High values (>4): Fat tails, extreme events likely
        Low values (<3): Thin tails, more normal distribution

    Args:
        x: Standardized returns
        window: Lookback period (63 ~ 3 months daily)

    Returns:
        DataFrame with 'skew' and 'kurt' columns

    """
    mu = x.rolling(window, min_periods=window).mean()
    sd = x.rolling(window, min_periods=window).std(ddof=0)
    z = x - mu  # Centered returns

    # Third moment (skewness) and fourth moment (kurtosis)
    skew = (z.pow(3).rolling(window, min_periods=window).mean()) / (sd.pow(3) + 1e-12)
    kurt = (z.pow(4).rolling(window, min_periods=window).mean()) / (sd.pow(4) + 1e-12)
    return pd.DataFrame({"skew": skew, "kurt": kurt})


def drawdown_metrics(price: pd.Series, window: int = 126) -> pd.DataFrame:
    """Calculate drawdown (DD) and drawup (DU) metrics.

    Uses current price for real-time risk monitoring.

    Drawdown: (Current Price / Recent High) - 1
        Measures depth of decline from peak

    Drawup: (Current Price / Recent Low) - 1
        Measures recovery from trough

    DD/DU speed: Rate of change in drawdown/drawup
        High speed = rapid moves (increased volatility/risk)

    Args:
        price: Price series
        window: Lookback for max/min (126 ~ 6 months daily)

    Returns:
        DataFrame with dd, du, dd_speed, du_speed

    """
    roll_max = price.rolling(window, min_periods=1).max()
    roll_min = price.rolling(window, min_periods=1).min()
    dd = price / roll_max - 1.0  # Always <= 0
    du = price / roll_min - 1.0  # Always >= 0
    dd_speed = dd.diff().abs()  # Absolute change in drawdown
    du_speed = du.diff().abs()  # Absolute change in drawup
    return pd.DataFrame({"dd": dd, "du": du, "dd_speed": dd_speed, "du_speed": du_speed})


# =====================================================================
#                    TAIL EVENT DETECTION (NO LOOKAHEAD)
# =====================================================================


def detect_tail_events(x: pd.Series, q: np.float64 = 0.95, tau: np.float64 = 2.0, min_history: int = 50) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Walk-forward tail event detection with ZERO lookahead bias.

    CRITICAL ANTI-LOOKAHEAD DESIGN:
    At each time point i, we:
    1. Fit GPD ONLY on data from [0, i-1] (strictly historical)
    2. Evaluate point i against this historical distribution
    3. Never use future data in any way

    This ensures the detector could be run in real-time production.

    Process:
    1. Standardize returns by volatility
    2. For each bar, fit GPD on all previous data
    3. Calculate p-value: how extreme is current observation?
    4. Convert to tail score: -log10(p)
    5. Flag as event if score >= tau threshold

    Args:
        x: Standardized return series
        q: Quantile for GPD threshold (0.95 = use 95th percentile)
        tau: Tail score threshold for event (2.0 = p<0.01, 1 in 100)
        min_history: Minimum bars before detecting events

    Returns:
        (tail_scores, tail_sides, is_event_mask)

    """
    pvals = []
    sides = []
    scores = []

    # Adaptive min_history: use at least 50 bars, or 20% of data, whichever is larger
    # This ensures enough data for stable GPD fits
    min_hist = max(min_history, int(len(x) * 0.2))

    for i in range(len(x)):
        if i < min_hist:
            # Not enough history yet - mark as no event
            pvals.append(1.0)
            sides.append("none")
            scores.append(0.0)
            continue

        # CRITICAL: Fit GPD ONLY on data BEFORE current point (x[0:i])
        # This is the key to avoiding lookahead bias
        x_hist = x.iloc[:i]
        fit_r = fit_gpd_exceedances(x_hist, q=q, side="right")
        fit_l = fit_gpd_exceedances(x_hist, q=q, side="left")

        # Evaluate CURRENT point against HISTORICAL distribution
        x_curr = x.iloc[i]
        p, s = tail_pvalue(x_curr, fit_r, fit_l)
        pvals.append(p)
        sides.append(s)
        scores.append(tail_score(p))

    ts = pd.Series(scores, index=x.index, name="TailScore")
    side_series = pd.Series(sides, index=x.index, name="TailSide")
    is_event = (ts >= tau) & (side_series.isin(["left", "right"]))
    return ts, side_series, is_event


# ----------------------- Trading overlay -----------------------


def strategy_from_events(price: pd.Series, r: pd.Series, vol: pd.Series, event_mask: pd.Series, side_series: pd.Series, lookahead: int, tp_mult: np.float64, sl_mult: np.float64, policy: str, slippage_bps: np.float64 = 2.0) -> pd.DataFrame:
    """Realistic strategy with NO lookahead bias:
    - Event detected at close of bar t
    - Enter at OPEN of bar t+1 (next bar)
    - Use ONLY volatility known at time of entry (lagged)
    - Include slippage

    policy:
    - "long_right_only": enter +1 on right-tail events only; left => flat
    - "long_right_short_left": enter +1 on right, -1 on left
    - "flat_left": explicitly flatten on left events (risk-off), otherwise 0
    """
    idx = price.index
    entry = pd.Series(0.0, index=idx)
    slippage_mult = slippage_bps / 10000.0

    if policy == "long_right_only":
        entry[(event_mask) & (side_series == "right")] = 1.0
    elif policy == "long_right_short_left":
        entry[(event_mask) & (side_series == "right")] = 1.0
        entry[(event_mask) & (side_series == "left")] = -1.0
    elif policy == "flat_left":
        entry[(event_mask) & (side_series == "left")] = 0.0
        entry[(event_mask) & (side_series == "right")] = 1.0
    else:
        entry[:] = 0.0

    pnl = pd.Series(0.0, index=idx)
    exits = pd.Series("", index=idx)

    for t in np.where(entry.values != 0)[0]:
        if t + 2 >= len(price):  # Need t+2 to enter at open of t+1
            continue
        pos = entry.iloc[t]

        # Enter at NEXT bar's open (approximated as next bar's close - realistic)
        # In real trading, you'd use actual open price
        px_in = price.iloc[t + 1]

        # Apply slippage - pay more for longs, receive less for shorts
        if pos > 0:
            px_in = px_in * (1 + slippage_mult)
        else:
            px_in = px_in * (1 - slippage_mult)

        # Use ONLY volatility known at time t (not t+1)
        v = np.float64(vol.iloc[t]) if np.isfinite(vol.iloc[t]) else np.nan
        if not np.isfinite(v) or v <= 0:
            # Fallback: use historical vol up to t
            v = np.float64(r.iloc[max(0, t - 20) : t + 1].std(ddof=0) or 0.0)
            if v <= 0:
                v = 1e-4

        # Set TP/SL levels
        if pos > 0:
            tp = px_in * (1 + tp_mult * v)
            sl = px_in * (1 - sl_mult * v)
        else:  # short
            tp = px_in * (1 - tp_mult * v)
            sl = px_in * (1 + sl_mult * v)

        exit_idx = t + 2
        exit_reason = "time"
        # Check for TP/SL starting from bar t+2 (first bar after entry)
        for k in range(t + 2, min(t + 2 + lookahead, len(price))):
            px = price.iloc[k]
            if pos > 0:
                if px >= tp:
                    exit_idx = k
                    exit_reason = "tp"
                    break
                if px <= sl:
                    exit_idx = k
                    exit_reason = "sl"
                    break
            else:  # short - fixed logic
                if px <= tp:  # Short profits when price falls
                    exit_idx = k
                    exit_reason = "tp"
                    break
                if px >= sl:  # Short loses when price rises
                    exit_idx = k
                    exit_reason = "sl"
                    break

        # Exit price with slippage
        px_out = price.iloc[exit_idx]
        if pos > 0:
            px_out = px_out * (1 - slippage_mult)  # Receive less when selling
        else:
            px_out = px_out * (1 + slippage_mult)  # Pay more when covering short

        pnl.iloc[exit_idx] += pos * (px_out - px_in)
        exits.iloc[exit_idx] = exit_reason

    ret_units = pnl / price.shift(1)
    equity = (1 + ret_units.fillna(0)).cumprod()
    return pd.DataFrame({
        "Entry": entry,
        "TradePnL": pnl,
        "TradeRet": ret_units,
        "EquityCurve": equity,
        "ExitReason": exits,
    })


# ----------------------- Pipeline -----------------------


@dataclass
class Params:
    q: np.float64
    tau: np.float64
    alpha: np.float64
    lookahead: int
    tp_mult: np.float64
    sl_mult: np.float64
    policy: str
    roll_mom_win: int = 63
    dd_win: int = 126


def run_once(price: pd.Series, params: Params) -> pd.DataFrame:
    r = log_returns(price)
    vol = ewma_vol(r, alpha=params.alpha)
    x = scale_returns(r, vol)

    tailscore, tailside, is_event = detect_tail_events(x, q=params.q, tau=params.tau)
    moments = rolling_moments(x, window=params.roll_mom_win)
    dd = drawdown_metrics(price, window=params.dd_win)

    df = pd.concat(
        [price, r.rename("r"), vol.rename("vol"), x.rename("x"), tailscore, tailside, is_event.rename("IsTailEvent"), moments, dd],
        axis=1,
    )

    # Regime classification uses CURRENT bar's stats (real-time inference)
    df["Regime"] = np.where((df["kurt"] >= 4.0) & (df["skew"] <= -0.5), "LeftRisk", np.where((df["kurt"] >= 4.0) & (df["skew"] >= 0.5), "RightBurst", "Normal"))

    # For trading: lag regime by 1 bar to avoid lookahead
    # You see regime change at close of bar t, trade on bar t+1
    df["Regime_tradeable"] = df["Regime"].shift(1).fillna("Normal")

    strat = strategy_from_events(
        df["price"],
        df["r"],
        df["vol"],
        df["IsTailEvent"].fillna(False),
        df["TailSide"].fillna("none"),
        lookahead=params.lookahead,
        tp_mult=params.tp_mult,
        sl_mult=params.sl_mult,
        policy=params.policy,
    )
    out = pd.concat([df, strat], axis=1)
    return out


# ----------------------- Optimization -----------------------


def train_test_split_by_time(s: pd.Series, frac: np.float64 = 0.7) -> Tuple[pd.Index, pd.Index]:
    n = len(s)
    k = int(n * frac)
    return s.index[:k], s.index[k:]


def make_objective(price: pd.Series, ppyr: np.float64):
    """Optimization objective with proper walk-forward validation.
    Train/test split happens HERE, before any processing.
    """
    train_idx, test_idx = train_test_split_by_time(price, frac=0.7)
    price_train = price.loc[train_idx]

    def objective(trial: "optuna.trial.Trial") -> np.float64:
        q = trial.suggest_float("q", 0.90, 0.99)
        tau = trial.suggest_float("tau", 1.5, 3.5)
        alpha = trial.suggest_float("alpha", 0.01, 0.20)
        lookahead = trial.suggest_int("lookahead", 2, 10)
        tp_mult = trial.suggest_float("tp_mult", 0.5, 3.0)
        sl_mult = trial.suggest_float("sl_mult", 0.5, 3.0)
        policy = trial.suggest_categorical("policy", ["long_right_only", "long_right_short_left", "flat_left"])

        params = Params(q=q, tau=tau, alpha=alpha, lookahead=lookahead, tp_mult=tp_mult, sl_mult=sl_mult, policy=policy)
        # Process ONLY train data - no contamination from test set
        res = run_once(price_train, params)
        sharpe = sharpe_ratio(res["TradeRet"], ppyr)
        # Regularize to discourage excessive trades: small penalty on trade count
        trades = int((res["Entry"] != 0).sum())
        penalty = 0.0005 * trades
        return sharpe - penalty

    return objective


# ----------------------- Diagnostics -----------------------


def summarize_results(df: pd.DataFrame, ppyr: np.float64) -> Dict[str, np.float64]:
    out = {}
    out["n_rows"] = np.float64(len(df))
    out["n_events"] = np.float64(df["IsTailEvent"].fillna(False).sum())
    out["n_left"] = np.float64(((df["IsTailEvent"]) & (df["TailSide"] == "left")).sum())
    out["n_right"] = np.float64(((df["IsTailEvent"]) & (df["TailSide"] == "right")).sum())
    out["sharpe"] = np.float64(sharpe_ratio(df["TradeRet"], ppyr))
    eq = df["EquityCurve"].dropna()
    out["equity_final"] = np.float64(eq.iloc[-1]) if len(eq) else 1.0
    return out


def print_summary(df: pd.DataFrame, label: str, ppyr: np.float64) -> None:
    s = summarize_results(df, ppyr)
    print(f"[{label}] rows={int(s['n_rows'])} events={int(s['n_events'])} left={int(s['n_left'])} right={int(s['n_right'])} sharpe={s['sharpe']:.3f} equity_final={s['equity_final']:.3f}")


# ----------------------- Plots -----------------------


def plot_all_diagnostics(df: pd.DataFrame, title: str) -> None:
    """Combined plot: Price (with regimes and tail events), TailScore, EquityCurve, Regime counts.
    Annotates practical usage.
    Note: Plot shows real-time 'Regime' for visualization.
    Trading uses 'Regime_tradeable' (lagged by 1 bar) to avoid lookahead.
    """
    import seaborn as sns

    sns.set_theme(style="whitegrid", font_scale=1.1, rc={"axes.titlesize": 15, "axes.labelsize": 13})
    regimes = df["Regime"].fillna("Normal")
    colors = {"LeftRisk": sns.color_palette("rocket")[2], "RightBurst": sns.color_palette("crest")[3], "Normal": sns.color_palette("Set2")[1]}
    evt = df["IsTailEvent"].fillna(False)
    right_evt = evt & (df["TailSide"] == "right")
    left_evt = evt & (df["TailSide"] == "left")

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1.5, 0.7]})

    # 1. Price with regime coloring and tail events
    ax = axes[0]
    ax.plot(df.index, df["price"], color="black", lw=1.5, label="Price")
    # Regime background
    last = None
    for idx, regime in regimes.items():
        if last is None or regime != last[1]:
            if last is not None:
                ax.axvspan(last[0], idx, color=colors.get(last[1], "#e0e0e0"), alpha=0.13, zorder=0)
            last = (idx, regime)
    if last is not None:
        ax.axvspan(last[0], df.index[-1], color=colors.get(last[1], "#e0e0e0"), alpha=0.13, zorder=0)
    # Tail events
    ax.scatter(df.index[right_evt], df["price"][right_evt], color=sns.color_palette("crest")[3], s=32, marker="^", label="Right Tail Event", edgecolor="k", linewidth=0.5, zorder=3)
    ax.scatter(df.index[left_evt], df["price"][left_evt], color=sns.color_palette("rocket")[2], s=32, marker="v", label="Left Tail Event", edgecolor="k", linewidth=0.5, zorder=3)
    ax.set_ylabel("Price", fontsize=13)
    ax.set_title(f"{title} — Price, Regimes, Tail Events", fontsize=15, weight="bold")
    ax.legend(loc="upper left", fontsize=11, frameon=True)

    # 2. TailScore
    ax2 = axes[1]
    ax2.plot(df.index, df["TailScore"], color=sns.color_palette("dark")[2], lw=1.3, label="TailScore")
    ax2.scatter(df.index[right_evt], df["TailScore"][right_evt], color=sns.color_palette("crest")[3], s=24, marker="^", label="Right Tail Event", edgecolor="k", linewidth=0.4)
    ax2.scatter(df.index[left_evt], df["TailScore"][left_evt], color=sns.color_palette("rocket")[2], s=24, marker="v", label="Left Tail Event", edgecolor="k", linewidth=0.4)
    ax2.set_ylabel("TailScore\n(-log10 p)", fontsize=13)
    ax2.set_title("TailScore (tail risk intensity)", fontsize=14)
    ax2.legend(loc="upper left", fontsize=10, frameon=True)

    # 3. Equity Curve
    ax3 = axes[2]
    ax3.plot(df.index, df["EquityCurve"], color=sns.color_palette("deep")[0], lw=1.7, label="Equity Curve")
    ax3.set_ylabel("Equity (×)", fontsize=13)
    ax3.set_title("Strategy Equity Curve", fontsize=14)
    ax3.legend(loc="upper left", fontsize=10, frameon=True)

    # 4. Regime counts
    ax4 = axes[3]
    counts = regimes.value_counts(dropna=False).sort_index()
    bar_colors = [colors.get(k, "#e0e0e0") for k in counts.index]
    bars = ax4.bar(counts.index, counts.values, color=bar_colors, edgecolor="#333", linewidth=0.7)
    ax4.set_ylabel("Bars", fontsize=13)
    ax4.set_title("Regime Counts", fontsize=14)
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=11)

    # Practical usage annotation
    usage = "How to use:\n- Green triangles: right-tail (positive) events, red triangles: left-tail (risk) events.\n- Regime shading: red=LeftRisk, green=RightBurst, gray=Normal (real-time inference).\n- Trading uses LAGGED regime (1 bar delay) to avoid lookahead bias.\n- Use right-tail events as long signals, left-tail as risk-off or short.\n- TailScore > threshold = actionable event.\n- Equity curve shows cumulative strategy performance.\n- Regime counts: how often each regime occurs.\n\nPractical tip: Combine tail events with regime context for robust entries/exits.\nNote: Strategy enters at NEXT bar's price with slippage for realistic results."
    fig.text(0.99, 0.01, usage, ha="right", va="bottom", fontsize=11, color="#222", bbox=dict(facecolor="#f9f9f9", edgecolor="#bbb", boxstyle="round,pad=0.5"))

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    sns.despine()
    plt.show()


# =====================================================================
#                         MAIN EXECUTION
# =====================================================================
#
# WORKFLOW:
# 1. Load and clean price data
# 2. Run Optuna hyperparameter optimization on TRAIN set only
# 3. Evaluate best parameters on TRAIN, TEST, and FULL datasets
# 4. Generate comprehensive diagnostic plots
#
# OUTPUT:
# - Console: Performance metrics (Sharpe, events, equity) for each split
# - Visualization: Multi-panel plot showing price, tail scores, equity, regimes


def main():
    """Main execution pipeline.

    Loads data, optimizes parameters, evaluates strategy, and plots results.
    All operations respect temporal ordering and avoid lookahead bias.
    """
    warnings.filterwarnings("ignore")

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Load and prepare data
    raw_df, _ = load_backtest_prices()  # User-provided data loader
    raw_df = _as_dt_index(raw_df)  # Convert to DatetimeIndex if possible
    price = _infer_price_series(raw_df)  # Extract clean price series

    # Determine annualization factor for Sharpe calculation
    ppyr = periods_per_year(price.index)

    # Hyperparameter optimization using Optuna (Bayesian optimization)
    # Optimizes on TRAIN set only to prevent overfitting
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(make_objective(price, ppyr), n_trials=100, show_progress_bar=False)

    # Extract best parameters found by optimization
    best = study.best_trial.params
    params = Params(
        q=np.float64(best["q"]),
        tau=np.float64(best["tau"]),
        alpha=np.float64(best["alpha"]),
        lookahead=int(best["lookahead"]),
        tp_mult=np.float64(best["tp_mult"]),
        sl_mult=np.float64(best["sl_mult"]),
        policy=str(best["policy"]),
    )

    # Evaluate strategy on train, test, and full datasets
    # This provides insight into overfitting and generalization
    train_idx, test_idx = train_test_split_by_time(price, frac=0.7)
    res_train = run_once(price.loc[train_idx], params)
    res_test = run_once(price.loc[test_idx], params)
    res_full = run_once(price, params)

    # Print performance summaries
    print_summary(res_train, "TRAIN", ppyr)
    print_summary(res_test, "TEST", ppyr)
    print_summary(res_full, "FULL", ppyr)
    print(f"[BEST] q={params.q:.4f} tau={params.tau:.3f} alpha={params.alpha:.4f} lookahead={params.lookahead} tp_mult={params.tp_mult:.3f} sl_mult={params.sl_mult:.3f} policy={params.policy}")

    # Generate comprehensive diagnostic visualization
    plot_all_diagnostics(res_full, "Tail-Regime Overlay")


if __name__ == "__main__":
    main()
