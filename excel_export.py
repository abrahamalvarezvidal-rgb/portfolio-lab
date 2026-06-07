"""
excel_export.py - build a static (values-only) .xlsx workbook in memory.

Numbers are frozen (no live formulas, per your choice) but the Growth sheet
carries a NATIVE Excel line chart with a linear trendline, so the trend is
still drawn by Excel and you can switch it to exponential with a right-click.
"""
import io
import pandas as pd

# which Holdings columns are percentages vs plain numbers
_PCT_COLS = {"Total Return", "Annual Return", "Annual Vol"}
_NUM_COLS = {"Sharpe", "Beta", "Skew", "Excess Kurtosis", "Shapiro p", "Jarque-Bera p"}


def build_excel(summary_df: pd.DataFrame,
                holdings_df: pd.DataFrame,
                growth_df: pd.DataFrame,
                corr_df: pd.DataFrame) -> bytes:
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        title = wb.add_format({"bold": True, "font_size": 14})
        hdr = wb.add_format({"bold": True, "bg_color": "#1F2937",
                             "font_color": "#FFFFFF", "border": 1})
        pct = wb.add_format({"num_format": "0.00%"})
        num = wb.add_format({"num_format": "0.000"})

        # ---------- Summary ----------
        summary_df.to_excel(writer, sheet_name="Summary", index=False, startrow=2)
        ws = writer.sheets["Summary"]
        ws.write(0, 0, "Portfolio Summary", title)
        for c, name in enumerate(summary_df.columns):
            ws.write(2, c, name, hdr)
        ws.set_column(0, max(len(summary_df.columns) - 1, 0), 22)

        # ---------- Holdings ----------
        holdings_df.to_excel(writer, sheet_name="Holdings", index=False)
        wsh = writer.sheets["Holdings"]
        for c, name in enumerate(holdings_df.columns):
            wsh.write(0, c, name, hdr)
            if name in _PCT_COLS:
                wsh.set_column(c, c, 14, pct)
            elif name in _NUM_COLS:
                wsh.set_column(c, c, 14, num)
            else:
                wsh.set_column(c, c, 12)

        # ---------- Growth (raw series + native chart w/ trendline) ----------
        growth_df.to_excel(writer, sheet_name="Growth", index=True)
        wsg = writer.sheets["Growth"]
        nrows = len(growth_df)
        chart = wb.add_chart({"type": "line"})
        for i, col in enumerate(growth_df.columns):
            chart.add_series({
                "name":       ["Growth", 0, i + 1],
                "categories": ["Growth", 1, 0, nrows, 0],
                "values":     ["Growth", 1, i + 1, nrows, i + 1],
                "trendline":  {"type": "linear", "line": {"dash_type": "dash"}},
            })
        chart.set_title({"name": "Growth of 100  (dashed = linear trendline)"})
        chart.set_x_axis({"name": "Date"})
        chart.set_y_axis({"name": "Value (start = 100)"})
        chart.set_size({"width": 760, "height": 430})
        wsg.insert_chart(1, len(growth_df.columns) + 2, chart)
        wsg.set_column(0, 0, 12)

        # ---------- Correlation ----------
        corr_df.to_excel(writer, sheet_name="Correlation")
        wsc = writer.sheets["Correlation"]
        n = len(corr_df)
        if n > 0:
            wsc.conditional_format(1, 1, n, n, {
                "type": "3_color_scale",
                "min_type": "num", "min_value": -1, "min_color": "#F8696B",
                "mid_type": "num", "mid_value": 0,  "mid_color": "#FFEB84",
                "max_type": "num", "max_value": 1,  "max_color": "#63BE7B",
            })
        wsc.set_column(0, n, 12, num)

    out.seek(0)
    return out.getvalue()


def build_my_portfolio_excel(summary_df: pd.DataFrame,
                             holdings_df: pd.DataFrame,
                             ledger_df: pd.DataFrame,
                             growth_df=None,
                             currency: str = "USD") -> bytes:
    """Static workbook for the REAL portfolio: Summary, Holdings (numeric, in the
    chosen display currency), Ledger (raw USD records), and an optional Growth
    sheet (TWR vs benchmark) with a native chart + trendline."""
    money_money = '"€"#,##0.00' if currency == "EUR" else '"$"#,##0.00'
    money_cols = {"Avg Cost", "Cost Basis", "Price", "Market Value",
                  "Unrealized P&L", "Realized P&L"}
    pct_cols = {"Weight", "Unrealized %"}

    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        wb = writer.book
        title = wb.add_format({"bold": True, "font_size": 14})
        hdr = wb.add_format({"bold": True, "bg_color": "#1F2937",
                             "font_color": "#FFFFFF", "border": 1})
        money = wb.add_format({"num_format": money_money})
        pct = wb.add_format({"num_format": "0.00%"})
        num = wb.add_format({"num_format": "0.0000"})
        usd = wb.add_format({"num_format": '"$"#,##0.00'})

        # ---------- Summary ----------
        summary_df.to_excel(writer, sheet_name="Summary", index=False, startrow=2)
        ws = writer.sheets["Summary"]
        ws.write(0, 0, "My Portfolio", title)
        for c, name in enumerate(summary_df.columns):
            ws.write(2, c, name, hdr)
        ws.set_column(0, 0, 30)
        ws.set_column(1, 1, 22)

        # ---------- Holdings ----------
        holdings_df.to_excel(writer, sheet_name="Holdings", index=False)
        wsh = writer.sheets["Holdings"]
        for c, name in enumerate(holdings_df.columns):
            wsh.write(0, c, name, hdr)
            if name in money_cols:
                wsh.set_column(c, c, 14, money)
            elif name in pct_cols:
                wsh.set_column(c, c, 11, pct)
            elif name == "Shares":
                wsh.set_column(c, c, 12, num)
            else:
                wsh.set_column(c, c, 16)

        # ---------- Ledger (raw records, prices in USD) ----------
        if ledger_df is not None and not ledger_df.empty:
            ledger_df.to_excel(writer, sheet_name="Ledger", index=False)
            wsl = writer.sheets["Ledger"]
            for c, name in enumerate(ledger_df.columns):
                wsl.write(0, c, name, hdr)
                if name in ("price", "fees"):
                    wsl.set_column(c, c, 12, usd)
                else:
                    wsl.set_column(c, c, 14)

        # ---------- Growth + native chart ----------
        if growth_df is not None and not growth_df.empty:
            growth_df.to_excel(writer, sheet_name="Growth", index=True)
            wsg = writer.sheets["Growth"]
            nrows = len(growth_df)
            chart = wb.add_chart({"type": "line"})
            for i, col in enumerate(growth_df.columns):
                chart.add_series({
                    "name": ["Growth", 0, i + 1],
                    "categories": ["Growth", 1, 0, nrows, 0],
                    "values": ["Growth", 1, i + 1, nrows, i + 1],
                    "trendline": {"type": "linear", "line": {"dash_type": "dash"}},
                })
            chart.set_title({"name": "Growth of 100 (dashed = linear trendline)"})
            chart.set_y_axis({"name": "Value (start = 100)"})
            chart.set_size({"width": 760, "height": 430})
            wsg.insert_chart(1, len(growth_df.columns) + 2, chart)
            wsg.set_column(0, 0, 12)

    out.seek(0)
    return out.getvalue()
