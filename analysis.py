"""
analysis.py - the analytical engine.

Pure functions only: fetch data, compute returns / risk / beta, run normality
tests, build portfolio stats. The Streamlit app AND the Excel exporter both call
these same functions, so the dashboard and the spreadsheet can never disagree.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

TRADING_DAYS = 252

# Friendly label -> yfinance "period" string
PERIOD_MAP = {
    "1 Week": "5d",
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "YTD": "ytd",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Max": "max",
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def fetch_prices(tickers, period, interval="1d") -> pd.DataFrame:
    """Download adjusted-close prices. Columns = tickers. Empty frame on failure."""
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return pd.DataFrame()

    raw = yf.download(tickers, period=period, interval=interval,
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        field = "Close" if "Close" in lvl0 else lvl0[0]
        prices = raw[field].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers

    prices = prices.dropna(how="all").ffill().dropna(how="all")
    # preserve requested order, drop tickers that returned nothing
    cols = [t for t in tickers if t in prices.columns]
    return prices[cols] if cols else prices


def compute_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    if prices.empty:
        return prices
    rets = np.log(prices / prices.shift(1)) if log else prices.pct_change()
    return rets.dropna(how="all")


def normalized_growth(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Rebase every series to `base` at the start so series are visually comparable."""
    p = prices.dropna(how="all").ffill()
    if p.empty:
        return p
    return p.divide(p.iloc[0]) * base


# --------------------------------------------------------------------------- #
# Single-series metrics
# --------------------------------------------------------------------------- #
def total_return(daily_rets: pd.Series) -> float:
    r = daily_rets.dropna()
    return float((1 + r).prod() - 1) if len(r) else np.nan


def annualized_return(daily_rets: pd.Series) -> float:
    r = daily_rets.dropna()
    n = len(r)
    if n == 0:
        return np.nan
    cum = (1 + r).prod()
    if cum <= 0:
        return np.nan
    return float(cum ** (TRADING_DAYS / n) - 1)


def annualized_vol(daily_rets: pd.Series) -> float:
    r = daily_rets.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_rets: pd.Series, rf: float = 0.0) -> float:
    vol = annualized_vol(daily_rets)
    if not vol or np.isnan(vol) or vol == 0:
        return np.nan
    return float((annualized_return(daily_rets) - rf) / vol)


def compute_beta(asset_rets: pd.Series, market_rets: pd.Series) -> float:
    df = pd.concat([asset_rets, market_rets], axis=1).dropna()
    if len(df) < 3:
        return np.nan
    y, x = df.iloc[:, 0].values, df.iloc[:, 1].values
    var = np.var(x, ddof=1)
    return float(np.cov(y, x, ddof=1)[0, 1] / var) if var > 0 else np.nan


def alpha_beta(asset_rets: pd.Series, market_rets: pd.Series):
    """OLS of asset on market. Returns (annualized_alpha, beta, r_squared)."""
    df = pd.concat([asset_rets, market_rets], axis=1).dropna()
    if len(df) < 3:
        return np.nan, np.nan, np.nan
    y, x = df.iloc[:, 0].values, df.iloc[:, 1].values
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return float(intercept * TRADING_DAYS), float(slope), r2


def normality(daily_rets: pd.Series) -> dict:
    """Shapiro-Wilk + Jarque-Bera p-values, plus skew & excess kurtosis."""
    r = daily_rets.dropna().values
    n = len(r)
    out = {"n": n, "skew": np.nan, "kurtosis": np.nan,
           "shapiro_p": np.nan, "jarque_bera_p": np.nan}
    if n >= 3:
        out["skew"] = float(stats.skew(r))
        out["kurtosis"] = float(stats.kurtosis(r))  # excess (normal = 0)
    if 3 <= n <= 5000:
        try:
            out["shapiro_p"] = float(stats.shapiro(r).pvalue)
        except Exception:
            pass
    if n >= 8:
        try:
            out["jarque_bera_p"] = float(stats.jarque_bera(r).pvalue)
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------- #
# Tables & portfolio
# --------------------------------------------------------------------------- #
def per_asset_table(returns_df: pd.DataFrame, market_rets=None, rf=0.0) -> pd.DataFrame:
    rows = []
    for col in returns_df.columns:
        r = returns_df[col]
        nrm = normality(r)
        rows.append({
            "Ticker": col,
            "Total Return": total_return(r),
            "Annual Return": annualized_return(r),
            "Annual Vol": annualized_vol(r),
            "Sharpe": sharpe_ratio(r, rf),
            "Beta": compute_beta(r, market_rets) if market_rets is not None else np.nan,
            "Skew": nrm["skew"],
            "Excess Kurtosis": nrm["kurtosis"],
            "Shapiro p": nrm["shapiro_p"],
            "Jarque-Bera p": nrm["jarque_bera_p"],
            "N": nrm["n"],
        })
    return pd.DataFrame(rows)


def portfolio_series(returns_df: pd.DataFrame, weights):
    """Weighted daily return series. Std of THIS series already embeds covariance."""
    w = np.array(weights, dtype=float)
    if w.sum() == 0:
        w = np.ones(len(w))
    w = w / w.sum()
    aligned = returns_df.dropna()
    return aligned @ w, w


def portfolio_stats(returns_df: pd.DataFrame, weights, rf=0.0) -> dict:
    series, w = portfolio_series(returns_df, weights)
    return {
        "weights": w,
        "total_return": total_return(series),
        "annual_return": annualized_return(series),
        "annual_vol": annualized_vol(series),
        "sharpe": sharpe_ratio(series, rf),
        "series": series,
    }


def fit_trendline(y, kind="linear"):
    """Return fitted y-values for a 'linear' or 'exponential' trend, or None."""
    y = np.asarray(y, dtype=float)
    x = np.arange(len(y))
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return None
    if kind == "linear":
        return np.polyval(np.polyfit(x[mask], y[mask], 1), x)
    if kind == "exponential":
        if np.any(y[mask] <= 0):
            return None
        c = np.polyfit(x[mask], np.log(y[mask]), 1)
        return np.exp(np.polyval(c, x))
    return None


# --------------------------------------------------------------------------- #
# Monte Carlo & Efficient frontier
# --------------------------------------------------------------------------- #
def monte_carlo(daily_rets, init_value=10000.0, horizon_days=252,
                n_sims=2000, method="bootstrap", seed=None):
    """Forward value paths. method='bootstrap' resamples real days (keeps fat
    tails); 'normal' draws from N(mean, std). Returns summary dict or None."""
    r = np.asarray(daily_rets.dropna(), dtype=float)
    if len(r) < 2:
        return None
    rng = np.random.default_rng(seed)
    if method == "normal":
        draws = rng.normal(r.mean(), r.std(ddof=1), size=(n_sims, horizon_days))
    else:
        idx = rng.integers(0, len(r), size=(n_sims, horizon_days))
        draws = r[idx]
    growth = np.cumprod(1 + draws, axis=1)
    paths = np.concatenate([np.ones((n_sims, 1)), growth], axis=1) * init_value
    ending = paths[:, -1]
    return {
        "paths": paths,
        "percentiles": {p: np.percentile(paths, p, axis=0) for p in (5, 25, 50, 75, 95)},
        "ending": ending,
        "median_end": float(np.median(ending)),
        "p5_end": float(np.percentile(ending, 5)),
        "p95_end": float(np.percentile(ending, 95)),
        "prob_loss": float((ending < init_value).mean()),
        "init_value": init_value,
        "horizon_days": horizon_days,
    }


def efficient_frontier(returns_df, n_portfolios=4000, rf=0.0, seed=None):
    """Random long-only weight cloud with annualized ret/vol/Sharpe.
    Returns dict with cloud DataFrame + max-Sharpe and min-vol rows, or None."""
    cols = list(returns_df.columns)
    k = len(cols)
    if k < 2:
        return None
    rng = np.random.default_rng(seed)
    mean_daily = returns_df.mean().values
    cov_daily = returns_df.cov().values
    W = rng.random((n_portfolios, k))
    W /= W.sum(axis=1, keepdims=True)
    ann_ret = (W @ mean_daily) * TRADING_DAYS
    var = np.einsum("ij,jk,ik->i", W, cov_daily, W)
    vol = np.sqrt(np.clip(var, 0, None) * TRADING_DAYS)
    sharpe = np.where(vol > 0, (ann_ret - rf) / vol, np.nan)
    df = pd.DataFrame({"ret": ann_ret, "vol": vol, "sharpe": sharpe})
    for i, c in enumerate(cols):
        df[c] = W[:, i]
    return {
        "cloud": df,
        "max_sharpe": df.loc[df["sharpe"].idxmax()],
        "min_vol": df.loc[df["vol"].idxmin()],
        "cols": cols,
    }


def weights_stats(returns_df, weights, rf=0.0):
    """Annualized (ret, vol, sharpe) for a given weight vector — used to place
    a custom point on the frontier."""
    w = np.array(weights, dtype=float)
    w = w / w.sum() if w.sum() else np.ones(len(w)) / len(w)
    ann_ret = float((returns_df.mean().values @ w) * TRADING_DAYS)
    var = float(w @ returns_df.cov().values @ w)
    vol = float(np.sqrt(max(var, 0) * TRADING_DAYS))
    sharpe = (ann_ret - rf) / vol if vol > 0 else np.nan
    return ann_ret, vol, sharpe


def fetch_prices_range(tickers, start, end=None, interval="1d") -> pd.DataFrame:
    """Like fetch_prices but for an explicit [start, end] date range."""
    tickers = [t.strip().upper() for t in tickers if t and t.strip()]
    if not tickers:
        return pd.DataFrame()
    raw = yf.download(tickers, start=start, end=end, interval=interval,
                      auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0)
        field = "Close" if "Close" in lvl0 else lvl0[0]
        prices = raw[field].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers
    prices = prices.dropna(how="all").ffill().dropna(how="all")
    cols = [t for t in tickers if t in prices.columns]
    return prices[cols] if cols else prices
