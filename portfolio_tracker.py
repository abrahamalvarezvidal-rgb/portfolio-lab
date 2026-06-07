"""
portfolio_tracker.py - ledger-based tracking of a real portfolio.

Everything is computed from a transaction ledger (one row per buy/sell), so it
naturally supports a portfolio that changes every month. Holdings, FIFO cost
basis, realized/unrealized P&L, time-weighted return, broker breakdown and the
rebalance plan are all DERIVED from the ledger.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

import analysis as A

LEDGER_COLUMNS = ["date", "ticker", "action", "quantity", "price", "fees", "broker"]


# --------------------------------------------------------------------------- #
# Ledger handling
# --------------------------------------------------------------------------- #
def empty_ledger() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "ticker": pd.Series(dtype="object"),
        "action": pd.Series(dtype="object"),
        "quantity": pd.Series(dtype="float"),
        "price": pd.Series(dtype="float"),
        "fees": pd.Series(dtype="float"),
        "broker": pd.Series(dtype="object"),
    })


def clean_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop blank rows, normalize, sort by date."""
    df = df.copy()
    for c in LEDGER_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[LEDGER_COLUMNS]
    df = df.dropna(subset=["ticker"])
    df = df[df["ticker"].astype(str).str.strip() != ""]
    if df.empty:
        return empty_ledger()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["action"] = df["action"].astype(str).str.strip().str.upper().replace(
        {"B": "BUY", "S": "SELL"})
    for c in ("quantity", "price", "fees"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["fees"] = df["fees"].fillna(0.0)
    df = df.dropna(subset=["quantity", "price"])
    df["broker"] = df["broker"].fillna("").astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# FIFO cost basis
# --------------------------------------------------------------------------- #
def fifo_per_ticker(txns: pd.DataFrame) -> dict:
    """FIFO over date-sorted rows for ONE ticker. Buy fees raise cost basis;
    sell fees reduce proceeds. Returns remaining shares/cost + realized P&L."""
    lots = []          # each: [qty_remaining, cost_per_share]
    realized = 0.0
    oversold = False
    for _, t in txns.iterrows():
        qty, price, fee = float(t["quantity"]), float(t["price"]), float(t["fees"])
        if qty <= 0:
            continue
        if t["action"] == "BUY":
            cps = price + (fee / qty if qty else 0.0)
            lots.append([qty, cps])
        elif t["action"] == "SELL":
            remaining, cost_of_sold = qty, 0.0
            while remaining > 1e-9 and lots:
                lot = lots[0]
                take = min(lot[0], remaining)
                cost_of_sold += take * lot[1]
                lot[0] -= take
                remaining -= take
                if lot[0] <= 1e-9:
                    lots.pop(0)
            if remaining > 1e-9:
                oversold = True
            realized += (qty * price - fee) - cost_of_sold
    shares = sum(l[0] for l in lots)
    cost_basis = sum(l[0] * l[1] for l in lots)
    return {
        "shares": shares,
        "cost_basis": cost_basis,
        "avg_cost": cost_basis / shares if shares > 1e-9 else 0.0,
        "realized_pnl": realized,
        "oversold": oversold,
    }


def holdings_table(ledger: pd.DataFrame, prices_latest: dict) -> pd.DataFrame:
    """Per-ticker current state. Includes fully-exited tickers (shares 0) so
    their realized P&L still shows."""
    rows = []
    for tk, g in ledger.groupby("ticker"):
        f = fifo_per_ticker(g)
        price = float(prices_latest.get(tk, np.nan))
        mv = f["shares"] * price if not np.isnan(price) else np.nan
        unreal = mv - f["cost_basis"] if not np.isnan(mv) else np.nan
        unreal_pct = (unreal / f["cost_basis"]) if f["cost_basis"] > 1e-9 else np.nan
        brokers = ", ".join(sorted({b for b in g["broker"] if str(b).strip()}))
        rows.append({
            "Ticker": tk, "Shares": f["shares"], "Avg Cost": f["avg_cost"],
            "Cost Basis": f["cost_basis"], "Price": price, "Market Value": mv,
            "Unrealized P&L": unreal, "Unrealized %": unreal_pct,
            "Realized P&L": f["realized_pnl"], "Broker": brokers,
            "_oversold": f["oversold"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Time-weighted return
# --------------------------------------------------------------------------- #
def holdings_timeline(ledger: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Cumulative net shares per ticker on the price calendar (0 before first buy)."""
    tickers = sorted(ledger["ticker"].unique())
    shares = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    for tk, g in ledger.groupby("ticker"):
        signed = g.apply(
            lambda r: r["quantity"] if r["action"] == "BUY" else -r["quantity"],
            axis=1)
        daily = pd.Series(signed.values, index=pd.to_datetime(g["date"].values))
        daily = daily.groupby(level=0).sum().cumsum()
        cum = daily.reindex(prices.index, method="ffill").fillna(0.0)
        shares[tk] = cum.values
    return shares


def twr(ledger: pd.DataFrame, prices: pd.DataFrame) -> dict | None:
    """True time-weighted return: each day's return uses only the shares held
    entering that day, so deposit/withdrawal timing can't distort it."""
    if ledger.empty:
        return None
    tickers = [c for c in sorted(ledger["ticker"].unique()) if c in prices.columns]
    if not tickers:
        return None
    px = prices[tickers].ffill()
    shares = holdings_timeline(ledger, px)[tickers]
    entering = shares.shift(1)
    v_prev = (entering * px.shift(1)).sum(axis=1)        # V_{t-1}
    v_preflow = (entering * px).sum(axis=1)              # value before today's flow
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(v_prev > 0, v_preflow / v_prev - 1, 0.0)
    daily = pd.Series(r, index=px.index).fillna(0.0)
    held = shares.sum(axis=1) > 0
    if not held.any():
        return None
    start = held.idxmax()
    daily = daily.loc[daily.index >= start]
    growth = 100 * (1 + daily).cumprod()
    total = float(growth.iloc[-1] / 100 - 1) if len(growth) else np.nan
    n = len(daily)
    ann = float((1 + total) ** (A.TRADING_DAYS / n) - 1) if n > 1 and total > -1 else np.nan
    return {"growth": growth, "daily_returns": daily, "twr_total": total, "twr_annual": ann}


# --------------------------------------------------------------------------- #
# FX & rebalancing
# --------------------------------------------------------------------------- #
def usd_to_eur_rate() -> float:
    """How many EUR per 1 USD (latest close). NaN on failure."""
    try:
        d = yf.download("EURUSD=X", period="5d", interval="1d",
                        auto_adjust=True, progress=False)
        eurusd = float(pd.Series(d["Close"].values.ravel()).dropna().iloc[-1])
        return 1.0 / eurusd if eurusd else np.nan
    except Exception:
        return np.nan


def rebalance_plan(holdings_df: pd.DataFrame, target_weights: dict,
                   prices_latest: dict, extra_cash: float = 0.0,
                   min_trade: float = 1.0):
    """Orders to move from current holdings to target weights. Each ticker
    traded = 1 transaction (counts against the 10/month budget)."""
    cur_val = {r["Ticker"]: (0.0 if pd.isna(r["Market Value"]) else r["Market Value"])
               for _, r in holdings_df.iterrows()}
    total = float(sum(cur_val.values())) + extra_cash
    all_t = sorted(set(cur_val) | set(target_weights))
    tw = {t: float(target_weights.get(t, 0.0)) for t in all_t}
    s = sum(tw.values())
    tw = {t: w / s for t, w in tw.items()} if s > 0 else tw
    orders = []
    for t in all_t:
        price = float(prices_latest.get(t, np.nan))
        if np.isnan(price) or price <= 0:
            continue
        delta = tw[t] * total - cur_val.get(t, 0.0)
        if abs(delta) < min_trade:
            continue
        orders.append({
            "Ticker": t, "Action": "BUY" if delta > 0 else "SELL",
            "Shares": round(abs(delta) / price, 4),
            "Approx $": round(abs(delta), 2), "Price": round(price, 2),
        })
    return pd.DataFrame(orders), len(orders)
