"""
app.py - Streamlit portfolio dashboard.

Run with:  streamlit run app.py

Tabs:
  Compare assets · Charts & trends · Normality · Portfolio ·
  Monte Carlo · Efficient frontier · Benchmark comparison ·
  My portfolio (real tracker) · Export to Excel
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

import analysis as A
import excel_export as X
import portfolio_tracker as PT

st.set_page_config(page_title="Portfolio Lab", layout="wide")
st.title("📊 Portfolio Lab")
st.caption("by ottniel77")
st.caption("Compare stocks & ETFs, simulate a portfolio, project the future, track "
           "your real holdings. Educational analysis only — not investment advice.")

FALLBACK_TICKERS = "AAPL, GOOGL, NVDA"


def ledger_holdings_shares():
    """Current holdings {ticker: shares>0} from the in-session ledger (whatever the
    user has entered or uploaded this session). No disk access — multi-user safe.
    Empty on a fresh session."""
    led = st.session_state.get("ledger")
    if led is None or led.empty:
        return {}
    out = {}
    for tk, g in led.groupby("ticker"):
        f = PT.fifo_per_ticker(g)
        if f["shares"] > 1e-9:
            out[tk] = f["shares"]
    return out


def load_uploaded_ledger(uploaded):
    """Read a ledger from an uploaded CSV, or from a my_portfolio.xlsx export (its
    'Ledger' sheet). Returns a cleaned DataFrame, or None if it can't be parsed."""
    name = (getattr(uploaded, "name", "") or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            raw = pd.read_excel(uploaded, sheet_name="Ledger")
        else:
            raw = pd.read_csv(uploaded)
    except Exception:
        return None
    return PT.clean_ledger(raw)


# A ledger upload (in the My portfolio tab) stashes new analysis tickers here; we
# apply them before the sidebar widget is created on the following run.
if "_pending_tickers" in st.session_state:
    st.session_state["tickers_input"] = st.session_state.pop("_pending_tickers")

_held_shares = ledger_holdings_shares()
default_tickers_str = ", ".join(sorted(_held_shares)) if _held_shares else FALLBACK_TICKERS
st.session_state.setdefault("tickers_input", default_tickers_str)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Inputs")
    tickers_raw = st.text_input("Tickers (comma separated)", key="tickers_input",
                                help="Defaults to AAPL, GOOGL, NVDA. Once you import a "
                                     "portfolio in the My portfolio tab, these switch to "
                                     "your holdings. Edit freely.")
    benchmark = st.text_input("Benchmark / market index", value="^GSPC",
                              help="Used for beta and comparisons. ^GSPC = S&P 500, "
                                   "or an ETF like SPY, or any ticker.").strip().upper()
    period_label = st.selectbox("Time window", list(A.PERIOD_MAP.keys()), index=5)
    rf_pct = st.number_input("Risk-free rate (%, annual)", value=0.0, step=0.25,
                             help="Used in the Sharpe ratio. Leave 0 if unsure.")
    use_log = st.checkbox("Use log returns", value=False)
    st.divider()
    st.caption("Import a portfolio in the 'My portfolio' tab to set these tickers "
               "and the Portfolio-tab weights from your real holdings.")

tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
rf = rf_pct / 100.0
period = A.PERIOD_MAP[period_label]


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
class _NoData(Exception):
    pass


@st.cache_data(ttl=21600, show_spinner="Fetching prices…")
def load(all_tickers_tuple, period, use_log):
    prices = A.fetch_prices(list(all_tickers_tuple), period)
    if prices.empty:
        raise _NoData()          # exceptions are never cached — forces a fresh fetch next time
    return prices, A.compute_returns(prices, log=use_log)


@st.cache_data(ttl=21600, show_spinner="Fetching holding history…")
def load_range(tickers_tuple, start_iso):
    result = A.fetch_prices_range(list(tickers_tuple), start_iso)
    if result.empty:
        raise _NoData()
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def latest_prices(tickers_tuple):
    p = A.fetch_prices(list(tickers_tuple), "5d")
    return {} if p.empty else {k: float(v) for k, v in p.iloc[-1].items()}


@st.cache_data(ttl=43200, show_spinner=False)
def get_fx():
    return PT.usd_to_eur_rate()


def cur_weights(symbols):
    w = [st.session_state.get(f"w_{t}", 1.0 / len(symbols)) for t in symbols]
    return w if sum(w) else [1.0] * len(symbols)


if not tickers:
    st.info("Add at least one ticker in the sidebar to begin.")
    st.stop()

all_tickers = tickers + ([benchmark] if benchmark and benchmark not in tickers else [])
try:
    prices, rets = load(tuple(all_tickers), period, use_log)
except _NoData:
    st.error("No data returned from Yahoo Finance for these tickers / time window.")
    st.caption(
        "This sometimes happens when Yahoo briefly rate-limits the server. "
        "The failed result is **not** cached, so clicking Refresh will try again immediately.")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

got = [t for t in tickers if t in prices.columns]
missing = [t for t in tickers if t not in prices.columns]
if missing:
    st.warning(f"No data for: {', '.join(missing)}. Continuing with: {', '.join(got)}")
have_bench = benchmark in prices.columns
mkt_rets = rets[benchmark] if have_bench else None
asset_prices, asset_rets = prices[got], rets[got]

(tab_cmp, tab_chart, tab_norm, tab_port, tab_mc, tab_ef,
 tab_bench, tab_track, tab_xl) = st.tabs(
    ["Compare assets", "Charts & trends", "Normality", "Portfolio",
     "Monte Carlo", "Efficient frontier", "Benchmark comparison",
     "My portfolio", "Export"])


# --------------------------------------------------------------------------- #
# 1. Compare assets
# --------------------------------------------------------------------------- #
with tab_cmp:
    st.subheader("Per-asset metrics")
    table = A.per_asset_table(asset_rets, market_rets=mkt_rets, rf=rf)
    show = table.copy()
    for c in ("Total Return", "Annual Return", "Annual Vol"):
        show[c] = show[c].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
    for c in ("Sharpe", "Beta", "Skew", "Excess Kurtosis", "Shapiro p", "Jarque-Bera p"):
        show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption(f"Window: {period_label} · Beta benchmark: "
               f"{benchmark if have_bench else 'unavailable'} · "
               f"Annualized with {A.TRADING_DAYS} trading days.")


# --------------------------------------------------------------------------- #
# 2. Charts & trends
# --------------------------------------------------------------------------- #
with tab_chart:
    st.subheader("Growth of 100")
    pick = st.multiselect("Series", got, default=got)
    trend = st.radio("Trendline", ["None", "Linear", "Exponential"], horizontal=True)
    if pick:
        growth = A.normalized_growth(asset_prices[pick])
        fig = go.Figure()
        for col in pick:
            fig.add_trace(go.Scatter(x=growth.index, y=growth[col], mode="lines", name=col))
            if trend != "None":
                fitted = A.fit_trendline(growth[col].values, trend.lower())
                if fitted is not None:
                    fig.add_trace(go.Scatter(x=growth.index, y=fitted, mode="lines",
                                             name=f"{col} {trend.lower()} trend",
                                             line=dict(dash="dash")))
        fig.update_layout(height=480, hovermode="x unified",
                          yaxis_title="Value (start = 100)",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one series.")


# --------------------------------------------------------------------------- #
# 3. Normality
# --------------------------------------------------------------------------- #
with tab_norm:
    st.subheader("Are returns normally distributed?")
    st.caption("Most risk measures assume normal returns. If they're not, those "
               "numbers understate extreme moves. p < 0.05 → reject normality.")
    sym = st.selectbox("Ticker", got, key="norm_sym")
    series = asset_rets[sym].dropna()
    nrm = A.normality(series)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sample size", nrm["n"])
    c2.metric("Skew", f"{nrm['skew']:.3f}" if pd.notna(nrm["skew"]) else "—")
    c3.metric("Excess kurtosis", f"{nrm['kurtosis']:.3f}" if pd.notna(nrm["kurtosis"]) else "—")
    c4.metric("Shapiro p", f"{nrm['shapiro_p']:.4f}" if pd.notna(nrm["shapiro_p"]) else "—")
    g1, g2 = st.columns(2)
    with g1:
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=series, nbinsx=40, histnorm="probability density",
                                    name="returns"))
        xs = np.linspace(series.min(), series.max(), 200)
        hist.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, series.mean(), series.std()),
                                  mode="lines", name="normal fit"))
        hist.update_layout(title="Histogram vs normal", height=380,
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(hist, use_container_width=True)
    with g2:
        (osm, osr), (slope, intercept, _) = stats.probplot(series, dist="norm")
        qq = go.Figure()
        qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="data"))
        qq.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode="lines", name="normal line"))
        qq.update_layout(title="Q-Q plot", height=380, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(qq, use_container_width=True)
    if pd.notna(nrm["shapiro_p"]):
        if nrm["shapiro_p"] < 0.05:
            st.warning(f"Shapiro-Wilk p = {nrm['shapiro_p']:.4f} (< 0.05): not normal. "
                       "Std-based risk numbers are optimistic about extreme moves.")
        else:
            st.success(f"Shapiro-Wilk p = {nrm['shapiro_p']:.4f} (≥ 0.05): "
                       "no strong evidence against normality here.")
    else:
        st.info("Too few data points for a reliable test — pick a longer window.")


# --------------------------------------------------------------------------- #
# 4. Portfolio
# --------------------------------------------------------------------------- #
with tab_port:
    st.subheader("Simulate a portfolio")
    st.caption("Weights default to your real holdings (by market value) when available; "
               "change them freely. Risk uses covariance, so it's usually below the "
               "average of individual vols — diversification.")

    # One-time seed from the saved portfolio, by market value (reusing fetched prices).
    if not st.session_state.get("_weights_seeded") and _held_shares:
        mv = {}
        for t in got:
            if t in _held_shares and t in prices.columns:
                px = prices[t].dropna()
                if len(px):
                    mv[t] = _held_shares[t] * float(px.iloc[-1])
        tot = sum(mv.values())
        if tot > 0:
            for t, v in mv.items():
                st.session_state[f"w_{t}"] = round(v / tot, 3)
        st.session_state["_weights_seeded"] = True
    # Ensure every current ticker has a starting weight (covers no-ledger and added tickers).
    for t in got:
        st.session_state.setdefault(f"w_{t}", round(1.0 / len(got), 3))

    cols = st.columns(min(len(got), 4) or 1)
    weights = []
    for i, t in enumerate(got):
        with cols[i % len(cols)]:
            weights.append(st.number_input(f"{t} weight", min_value=0.0,
                                           step=0.05, key=f"w_{t}"))
    if sum(weights) == 0:
        st.info("Give at least one holding a non-zero weight.")
    else:
        ps = A.portfolio_stats(asset_rets, weights, rf=rf)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total return", f"{ps['total_return']:.2%}")
        m2.metric("Annualized return", f"{ps['annual_return']:.2%}")
        m3.metric("Annualized vol (risk)", f"{ps['annual_vol']:.2%}")
        m4.metric("Sharpe ratio", f"{ps['sharpe']:.3f}")
        st.dataframe(pd.DataFrame({"Ticker": got,
                                   "Weight": [f"{w:.1%}" for w in ps["weights"]]}),
                     hide_index=True, use_container_width=True)
        if len(got) > 1:
            st.markdown("**Correlation matrix**")
            corr = asset_rets.corr()
            heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                        zmin=-1, zmax=1, colorscale="RdYlGn",
                                        text=corr.round(2).values, texttemplate="%{text}"))
            heat.update_layout(height=360)
            st.plotly_chart(heat, use_container_width=True)


# --------------------------------------------------------------------------- #
# 5. Monte Carlo
# --------------------------------------------------------------------------- #
with tab_mc:
    st.subheader("Monte Carlo projection")
    st.caption("Projects your portfolio forward over many simulated paths. "
               "Bootstrap resamples your real daily returns (keeps fat tails / crash "
               "days); Normal draws from a bell curve (smoother, but understates "
               "extremes for non-normal stocks — see the Normality tab).")
    weights = cur_weights(got)
    cc1, cc2, cc3, cc4 = st.columns(4)
    init_val = cc1.number_input("Starting value ($)", min_value=100.0, value=10000.0, step=500.0)
    horizon_m = cc2.slider("Horizon (months)", 1, 60, 12)
    n_sims = cc3.select_slider("Simulations", [500, 1000, 2000, 5000, 10000], value=2000)
    method = cc4.radio("Engine", ["Bootstrap", "Normal"])
    port_series, _ = A.portfolio_series(asset_rets, weights)
    mc = A.monte_carlo(port_series, init_value=init_val,
                       horizon_days=int(horizon_m / 12 * A.TRADING_DAYS),
                       n_sims=int(n_sims), method=method.lower(), seed=42)
    if mc is None:
        st.info("Not enough return history to simulate.")
    else:
        x = np.arange(mc["paths"].shape[1])
        p = mc["percentiles"]
        fan = go.Figure()
        fan.add_trace(go.Scatter(x=x, y=p[95], line=dict(width=0), showlegend=False))
        fan.add_trace(go.Scatter(x=x, y=p[5], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(99,150,255,0.15)", name="5–95%"))
        fan.add_trace(go.Scatter(x=x, y=p[75], line=dict(width=0), showlegend=False))
        fan.add_trace(go.Scatter(x=x, y=p[25], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(99,150,255,0.30)", name="25–75%"))
        fan.add_trace(go.Scatter(x=x, y=p[50], line=dict(color="#1f5fff", width=2),
                                 name="median"))
        fan.update_layout(height=440, title="Projected value (percentile fan)",
                          xaxis_title="Trading days ahead", yaxis_title="Value ($)",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fan, use_container_width=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median ending", f"${mc['median_end']:,.0f}")
        m2.metric("5th percentile", f"${mc['p5_end']:,.0f}")
        m3.metric("95th percentile", f"${mc['p95_end']:,.0f}")
        m4.metric("Prob. of loss", f"{mc['prob_loss']:.1%}")
        hist = go.Figure(go.Histogram(x=mc["ending"], nbinsx=60))
        hist.add_vline(x=init_val, line_dash="dash", annotation_text="start")
        hist.update_layout(height=320, title="Distribution of ending values",
                           xaxis_title="Ending value ($)")
        st.plotly_chart(hist, use_container_width=True)


# --------------------------------------------------------------------------- #
# 6. Efficient frontier
# --------------------------------------------------------------------------- #
with tab_ef:
    st.subheader("Efficient frontier")
    st.caption("Each dot is a random mix of your assets (long-only). The upper-left "
               "edge is the 'efficient frontier' — best return for a given risk. Your "
               "current weights are marked so you can see how far off the edge you sit.")
    if len(got) < 2:
        st.info("Add at least two tickers to build a frontier.")
    else:
        npf = st.select_slider("Random portfolios", [1000, 2000, 5000, 10000], value=5000)
        ef = A.efficient_frontier(asset_rets, n_portfolios=int(npf), rf=rf, seed=7)
        if ef is None:
            st.warning(
                "Can't build a frontier with this combination of assets. This usually "
                "happens when one ticker has near-zero volatility (e.g. a money-market "
                "or very stable bond ETF like IBCH) — there's not enough price variation "
                "to generate meaningful risk/return trade-offs. Try swapping it for a "
                "higher-volatility asset, or check the time window (a very short window "
                "can also produce too few data points).")
        else:
            cloud, ms, mv = ef["cloud"], ef["max_sharpe"], ef["min_vol"]
            cw = cur_weights(got)
            cr, cvol, csh = A.weights_stats(asset_rets, cw, rf=rf)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cloud["vol"], y=cloud["ret"], mode="markers",
                                     marker=dict(size=4, color=cloud["sharpe"],
                                                 colorscale="Viridis", showscale=True,
                                                 colorbar=dict(title="Sharpe")),
                                     name="portfolios", opacity=0.5))
            fig.add_trace(go.Scatter(x=[ms["vol"]], y=[ms["ret"]], mode="markers",
                                     marker=dict(size=16, symbol="star", color="gold",
                                                 line=dict(width=1, color="black")),
                                     name="max Sharpe"))
            fig.add_trace(go.Scatter(x=[mv["vol"]], y=[mv["ret"]], mode="markers",
                                     marker=dict(size=14, symbol="diamond", color="cyan",
                                                 line=dict(width=1, color="black")),
                                     name="min variance"))
            fig.add_trace(go.Scatter(x=[cvol], y=[cr], mode="markers",
                                     marker=dict(size=14, symbol="x", color="red"),
                                     name="your weights"))
            fig.update_layout(height=480, xaxis_title="Annualized volatility (risk)",
                              yaxis_title="Annualized return",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, use_container_width=True)
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Max-Sharpe weights**")
                st.dataframe(pd.DataFrame({"Ticker": ef["cols"],
                                           "Weight": [f"{ms[c]:.1%}" for c in ef["cols"]]}),
                             hide_index=True, use_container_width=True)
            with cB:
                st.markdown("**Min-variance weights**")
                st.dataframe(pd.DataFrame({"Ticker": ef["cols"],
                                           "Weight": [f"{mv[c]:.1%}" for c in ef["cols"]]}),
                             hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# 7. Benchmark comparison
# --------------------------------------------------------------------------- #
with tab_bench:
    st.subheader("Compare against a benchmark")
    left, right = st.columns(2)
    with left:
        subj_kind = st.radio("Evaluate", ["Portfolio", "Single ticker"])
        subj_ticker = (st.selectbox("Which ticker", got, key="subj")
                       if subj_kind == "Single ticker" else None)
    with right:
        opts = list(prices.columns)
        comp = st.selectbox("Compare against", opts,
                            index=opts.index(benchmark) if have_bench else 0)
    if subj_kind == "Portfolio":
        w = cur_weights(got)
        subj_rets, _ = A.portfolio_series(asset_rets, w)
        subj_label = "Portfolio"
        g = A.normalized_growth(asset_prices)
        subj_price = (g * np.array(w) / sum(w)).sum(axis=1)
    else:
        subj_rets = asset_rets[subj_ticker]
        subj_label = subj_ticker
        subj_price = A.normalized_growth(asset_prices[[subj_ticker]])[subj_ticker]
    comp_rets = rets[comp]
    comp_price = A.normalized_growth(prices[[comp]])[comp]
    alpha, beta, r2 = A.alpha_beta(subj_rets, comp_rets)
    aligned = pd.concat([subj_rets, comp_rets], axis=1).dropna()
    corr_val = aligned.iloc[:, 0].corr(aligned.iloc[:, 1]) if len(aligned) > 2 else np.nan
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Beta vs {comp}", f"{beta:.3f}" if pd.notna(beta) else "—")
    k2.metric("Annualized alpha", f"{alpha:.2%}" if pd.notna(alpha) else "—")
    k3.metric("R²", f"{r2:.3f}" if pd.notna(r2) else "—")
    k4.metric("Correlation", f"{corr_val:.3f}" if pd.notna(corr_val) else "—")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=subj_price.index, y=subj_price.values, mode="lines", name=subj_label))
    fig.add_trace(go.Scatter(x=comp_price.index, y=comp_price.values, mode="lines",
                             name=comp, line=dict(dash="dot")))
    fig.update_layout(title="Growth of 100 — subject vs benchmark", height=460,
                      yaxis_title="Value (start = 100)", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# 8. My portfolio (real tracker)
# --------------------------------------------------------------------------- #
def fmt_money(x, factor, sym):
    return "—" if pd.isna(x) else f"{sym}{x * factor:,.2f}"


def _render_tracker(ledger, factor, sym, ccy, fx, benchmark):
    """Analyze the ledger and render the My-portfolio tab. Returns a dict of
    data for the Excel export, or None if prices couldn't be fetched."""
    led_tickers = sorted(ledger["ticker"].unique())
    start = (ledger["date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    fetch_list = led_tickers + ([benchmark] if benchmark not in led_tickers else [])
    try:
        range_prices = load_range(tuple(fetch_list), start)
    except _NoData:
        st.error("Couldn't fetch price history for your holdings.")
        st.caption("Yahoo Finance may be temporarily unavailable. Click Refresh to retry.")
        if st.button("🔄 Refresh data", key="refresh_tracker"):
            st.cache_data.clear()
            st.rerun()
        return None
    latest = {t: float(range_prices[t].dropna().iloc[-1])
              for t in led_tickers if t in range_prices.columns}

    holdings = PT.holdings_table(ledger, latest)
    if holdings["_oversold"].any():
        st.warning("One or more tickers show more shares sold than bought — "
                   "check your ledger; results for those may be off.")

    held = holdings[holdings["Shares"] > 1e-9].copy()
    total_mv = held["Market Value"].sum(skipna=True)
    total_cb = held["Cost Basis"].sum(skipna=True)
    total_unreal = total_mv - total_cb
    total_unreal_pct = total_unreal / total_cb if total_cb > 1e-9 else np.nan
    total_realized = holdings["Realized P&L"].sum(skipna=True)

    st.markdown("**Current holdings**")
    disp = held.drop(columns=["_oversold"]).copy()
    disp.insert(disp.columns.get_loc("Market Value") + 1, "Weight",
                (held["Market Value"] / total_mv) if total_mv else np.nan)
    for c in ("Avg Cost", "Cost Basis", "Price", "Market Value", "Unrealized P&L", "Realized P&L"):
        disp[c] = disp[c].map(lambda v: fmt_money(v, factor, sym))
    disp["Weight"] = disp["Weight"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    disp["Unrealized %"] = disp["Unrealized %"].map(lambda v: f"{v:.2%}" if pd.notna(v) else "—")
    disp["Shares"] = disp["Shares"].map(lambda v: f"{v:,.4f}")
    st.dataframe(disp, hide_index=True, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Market value", fmt_money(total_mv, factor, sym))
    m2.metric("Cost basis", fmt_money(total_cb, factor, sym))
    m3.metric("Unrealized P&L", fmt_money(total_unreal, factor, sym),
              f"{total_unreal_pct:.2%}" if pd.notna(total_unreal_pct) else None)
    m4.metric("Realized P&L (sells)", fmt_money(total_realized, factor, sym))

    if not held.empty and held["Broker"].str.strip().any():
        st.markdown("**By broker**")
        bro = held.assign(Broker=held["Broker"].replace("", "(unspecified)")) \
                  .groupby("Broker")["Market Value"].sum().reset_index()
        bro["Market Value"] = bro["Market Value"].map(lambda v: fmt_money(v, factor, sym))
        st.dataframe(bro, hide_index=True, use_container_width=True)

    st.markdown("**Returns**")
    tw = PT.twr(ledger, range_prices[[t for t in led_tickers if t in range_prices.columns]])
    r1, r2, r3 = st.columns(3)
    r1.metric("Cost-basis return (unrealized)",
              f"{total_unreal_pct:.2%}" if pd.notna(total_unreal_pct) else "—",
              help="Current value vs what you paid for shares you still hold.")
    if tw:
        r2.metric("Time-weighted return (total)",
                  f"{tw['twr_total']:.2%}" if pd.notna(tw['twr_total']) else "—",
                  help="Strips out the timing of your buys/sells — comparable to the index.")
        r3.metric("Time-weighted (annualized)",
                  f"{tw['twr_annual']:.2%}" if pd.notna(tw['twr_annual']) else "—")

    alpha = beta = r2v = corr_val = np.nan
    bench_growth = None
    if tw and benchmark in range_prices.columns:
        st.markdown(f"**vs {benchmark}**")
        port_daily = tw["daily_returns"]
        bench_daily = range_prices[benchmark].pct_change().reindex(port_daily.index)
        alpha, beta, r2v = A.alpha_beta(port_daily, bench_daily)
        al = pd.concat([port_daily, bench_daily], axis=1).dropna()
        corr_val = al.iloc[:, 0].corr(al.iloc[:, 1]) if len(al) > 2 else np.nan
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Beta vs {benchmark}", f"{beta:.3f}" if pd.notna(beta) else "—")
        k2.metric("Annualized alpha", f"{alpha:.2%}" if pd.notna(alpha) else "—")
        k3.metric("R²", f"{r2v:.3f}" if pd.notna(r2v) else "—")
        k4.metric("Correlation", f"{corr_val:.3f}" if pd.notna(corr_val) else "—")
        bench_growth = 100 * (1 + bench_daily.fillna(0)).cumprod()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tw["growth"].index, y=tw["growth"].values,
                                 mode="lines", name="My portfolio (TWR)"))
        fig.add_trace(go.Scatter(x=bench_growth.index, y=bench_growth.values,
                                 mode="lines", name=benchmark, line=dict(dash="dot")))
        fig.update_layout(title="Growth of 100 — my portfolio vs benchmark", height=440,
                          yaxis_title="Value (start = 100)", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    # ---- assemble export data (display currency) ----
    he = held.drop(columns=["_oversold"]).copy()
    he.insert(he.columns.get_loc("Market Value") + 1, "Weight",
              (held["Market Value"] / total_mv) if total_mv else np.nan)
    for c in ("Avg Cost", "Cost Basis", "Price", "Market Value", "Unrealized P&L", "Realized P&L"):
        he[c] = he[c] * factor
    my_summary = pd.DataFrame([
        {"Metric": "Currency", "Value": ccy},
        {"Metric": "USD -> EUR rate", "Value": (f"{fx:.4f}" if ccy == "EUR" and pd.notna(fx) else "—")},
        {"Metric": "Market value", "Value": fmt_money(total_mv, factor, sym)},
        {"Metric": "Cost basis", "Value": fmt_money(total_cb, factor, sym)},
        {"Metric": "Unrealized P&L", "Value": fmt_money(total_unreal, factor, sym)},
        {"Metric": "Cost-basis return", "Value": (f"{total_unreal_pct:.2%}" if pd.notna(total_unreal_pct) else "—")},
        {"Metric": "Realized P&L (sells)", "Value": fmt_money(total_realized, factor, sym)},
        {"Metric": "Time-weighted return (total)", "Value": (f"{tw['twr_total']:.2%}" if tw and pd.notna(tw['twr_total']) else "—")},
        {"Metric": "Time-weighted return (annual)", "Value": (f"{tw['twr_annual']:.2%}" if tw and pd.notna(tw['twr_annual']) else "—")},
        {"Metric": f"Beta vs {benchmark}", "Value": (f"{beta:.3f}" if pd.notna(beta) else "—")},
        {"Metric": "Annualized alpha", "Value": (f"{alpha:.2%}" if pd.notna(alpha) else "—")},
        {"Metric": "R-squared", "Value": (f"{r2v:.3f}" if pd.notna(r2v) else "—")},
        {"Metric": "Correlation", "Value": (f"{corr_val:.3f}" if pd.notna(corr_val) else "—")},
    ])
    growth_export = None
    if tw:
        g = pd.DataFrame({"My Portfolio": tw["growth"]})
        if bench_growth is not None:
            g[benchmark] = bench_growth.reindex(tw["growth"].index)
        growth_export = g
    export_data = {"summary": my_summary, "holdings": he,
                   "ledger": ledger.copy(), "growth": growth_export, "currency": ccy}

    # ---- rebalance planner ----
    st.divider()
    st.markdown("**Rebalance planner** — design a target mix and get the exact orders.")
    st.caption("Each ticker traded counts as one of your 10 monthly transactions.")
    seed = pd.DataFrame({
        "Ticker": held["Ticker"].tolist(),
        "Target %": [round(100 * v / total_mv, 2) if total_mv else 0.0
                     for v in held["Market Value"]],
    }) if total_mv else pd.DataFrame({"Ticker": held["Ticker"].tolist(),
                                      "Target %": [0.0] * len(held)})
    target_edit = st.data_editor(seed, num_rows="dynamic", use_container_width=True,
                                 key="target_editor",
                                 column_config={"Target %": st.column_config.NumberColumn(
                                     "Target %", min_value=0.0, max_value=100.0, format="%.2f")})
    extra_disp = st.number_input(f"Add/withdraw cash ({sym}, optional)", value=0.0, step=100.0)
    extra_usd = extra_disp / factor

    target_edit = target_edit.dropna(subset=["Ticker"])
    target_edit = target_edit[target_edit["Ticker"].astype(str).str.strip() != ""]
    targets = {str(r["Ticker"]).strip().upper(): float(r["Target %"] or 0)
               for _, r in target_edit.iterrows()}
    union = sorted(set(led_tickers) | set(targets))
    px_for_plan = latest_prices(tuple(union))
    px_for_plan.update(latest)

    orders, n_orders = PT.rebalance_plan(held[["Ticker", "Market Value"]], targets,
                                         px_for_plan, extra_cash=extra_usd)
    this_month = ledger[ledger["date"].dt.to_period("M") == pd.Timestamp.today().to_period("M")]
    used = len(this_month)
    st.metric("Transactions this month",
              f"{used} used + {n_orders} planned = {used + n_orders} / 10")
    if used + n_orders > 10:
        st.warning(f"This plan would use {used + n_orders} transactions — over your "
                   "10/month limit. Trim the smallest trades or split across months.")
    if orders.empty:
        st.info("No trades needed — you're already at target (within the $1 threshold).")
    else:
        od = orders.copy()
        od["Approx"] = od["Approx $"].map(lambda v: fmt_money(v, factor, sym))
        od["Price"] = od["Price"].map(lambda v: fmt_money(v, factor, sym))
        st.dataframe(od[["Ticker", "Action", "Shares", "Approx", "Price"]],
                     hide_index=True, use_container_width=True)
        if st.button("➕ Add these orders to my ledger (dated today)"):
            today = pd.Timestamp.today().normalize()
            new = pd.DataFrame([{
                "date": today, "ticker": o.Ticker, "action": o.Action,
                "quantity": o.Shares, "price": o.Price, "fees": 0.0,
                "broker": "(rebalance)"} for o in orders.itertuples()])
            st.session_state["ledger"] = PT.clean_ledger(pd.concat([ledger, new], ignore_index=True))
            st.success("Added to your ledger. Download to save it — the ledger and "
                       "holdings above will refresh.")
            st.rerun()

    return export_data


with tab_track:
    st.subheader("My portfolio")
    st.caption("Log every buy and sell below. Holdings, FIFO cost basis, returns, broker "
               "split and rebalancing are all computed from this ledger. Prices are in "
               "USD; choose the display currency. Nothing is stored on the server — use "
               "Save (download) to keep your data and Import to load it back.")

    if "ledger" not in st.session_state:
        st.session_state["ledger"] = PT.empty_ledger()

    cset = st.columns([1, 1, 2])
    ccy = cset[0].radio("Display currency", ["USD", "EUR"], horizontal=True)
    fx = get_fx()
    if ccy == "EUR" and (fx is None or np.isnan(fx)):
        cset[2].warning("Couldn't fetch USD→EUR rate; showing USD.")
        ccy = "USD"
    factor = 1.0 if ccy == "USD" else fx
    sym = "$" if ccy == "USD" else "€"
    if ccy == "EUR":
        cset[1].metric("USD → EUR", f"{fx:.4f}")

    st.markdown("**Transaction ledger**")
    colcfg = {
        "date": st.column_config.DateColumn("Date"),
        "ticker": st.column_config.TextColumn("Ticker"),
        "action": st.column_config.SelectboxColumn("Action", options=["BUY", "SELL"]),
        "quantity": st.column_config.NumberColumn("Quantity", min_value=0.0, format="%.4f"),
        "price": st.column_config.NumberColumn("Price (USD)", min_value=0.0, format="%.2f"),
        "fees": st.column_config.NumberColumn("Fees (USD)", min_value=0.0, format="%.2f"),
        "broker": st.column_config.TextColumn("Broker"),
    }
    edited = st.data_editor(st.session_state["ledger"], num_rows="dynamic",
                            column_config=colcfg, use_container_width=True,
                            key="ledger_editor")
    ledger = PT.clean_ledger(edited)

    b1, b2 = st.columns(2)
    b1.download_button("💾 Save my portfolio (download)", data=ledger.to_csv(index=False),
                       file_name="my_portfolio_ledger.csv", mime="text/csv",
                       use_container_width=True,
                       help="Downloads your transactions to a file on your device. "
                            "Nothing is stored on the server — upload it next time to "
                            "restore. (The full analysis report is in the Export tab.)")
    up = b2.file_uploader("Import (CSV or my_portfolio.xlsx)", type=["csv", "xlsx"],
                          help="Load a saved ledger CSV, or a my_portfolio.xlsx export "
                               "(its Ledger sheet is read back in).")
    up_id = (getattr(up, "file_id", None) or
             (f"{up.name}:{up.size}" if up is not None else None))
    if up is not None and up_id != st.session_state.get("_last_upload_id"):
        st.session_state["_last_upload_id"] = up_id
        new_led = load_uploaded_ledger(up)
        if new_led is None:
            st.error("Couldn't read a ledger from that file. For Excel, use a "
                     "my_portfolio.xlsx export that contains a 'Ledger' sheet.")
        else:
            st.session_state["ledger"] = new_led
            held = {tk for tk, g in new_led.groupby("ticker")
                    if PT.fifo_per_ticker(g)["shares"] > 1e-9}
            if held:
                # drive the analysis tickers & weights from the imported portfolio
                st.session_state["_pending_tickers"] = ", ".join(sorted(held))
                st.session_state.pop("_weights_seeded", None)
            st.rerun()

    if ledger.empty:
        st.info("Add transactions above, or Import a saved file, to see your portfolio.")
        st.session_state["export_portfolio"] = None
    else:
        st.session_state["export_portfolio"] = _render_tracker(
            ledger, factor, sym, ccy, fx, benchmark)


# --------------------------------------------------------------------------- #
# 9. Export
# --------------------------------------------------------------------------- #
with tab_xl:
    st.subheader("Export to Excel")
    st.caption("Static workbooks (frozen values). The Growth sheets carry a native "
               "Excel chart with a trendline you can tweak in Excel.")

    st.markdown("**Analysis portfolio** (the tickers & weights from the sidebar / Portfolio tab)")
    w = cur_weights(got)
    ps = A.portfolio_stats(asset_rets, w, rf=rf)
    summary_df = pd.DataFrame(
        [{"Metric": "Window", "Value": period_label},
         {"Metric": "Benchmark", "Value": benchmark},
         {"Metric": "Risk-free rate", "Value": f"{rf_pct:.2f}%"},
         {"Metric": "Portfolio total return", "Value": f"{ps['total_return']:.2%}"},
         {"Metric": "Portfolio annual return", "Value": f"{ps['annual_return']:.2%}"},
         {"Metric": "Portfolio annual vol (risk)", "Value": f"{ps['annual_vol']:.2%}"},
         {"Metric": "Portfolio Sharpe", "Value": f"{ps['sharpe']:.3f}"}]
        + [{"Metric": f"{t} weight", "Value": f"{wv / sum(w):.1%}"}
           for t, wv in zip(got, w)])
    holdings_df = A.per_asset_table(asset_rets, market_rets=mkt_rets, rf=rf)
    growth_df = A.normalized_growth(asset_prices)
    corr_df = asset_rets.corr()
    xlsx_bytes = X.build_excel(summary_df, holdings_df, growth_df, corr_df)
    st.download_button("⬇️ Download portfolio.xlsx", data=xlsx_bytes,
                       file_name="portfolio.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.markdown("**My portfolio** (your real holdings from the My portfolio tab)")
    mp = st.session_state.get("export_portfolio")
    if not mp:
        st.info("Add transactions in the My portfolio tab (and Save) to enable this export.")
    else:
        my_bytes = X.build_my_portfolio_excel(
            mp["summary"], mp["holdings"], mp["ledger"], mp["growth"], currency=mp["currency"])
        st.download_button("⬇️ Download my_portfolio.xlsx", data=my_bytes,
                           file_name="my_portfolio.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption(f"Values in {mp['currency']}. Sheets: Summary, Holdings (with Weight), "
                   "Ledger, Growth (TWR vs benchmark).")
        st.dataframe(mp["summary"], hide_index=True, use_container_width=True)
