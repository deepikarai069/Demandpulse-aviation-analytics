"""Data, method, limitations – and a read-only SQL explorer over the SQLite layer."""
from __future__ import annotations

import re
import sqlite3

import pandas as pd
import streamlit as st

import ui
from data import load_all
from demandpulse import config, db


def render():
    d = load_all()
    ins = d["insights"]
    cov = ins["coverage"]
    ui.hero("Data, method & SQL", "Where the numbers come from, what was cleaned, and what this project cannot tell you.")

    tab1, tab2, tab3 = st.tabs(["Data & cleaning", "Method & limitations", "SQL explorer"])
    with tab1:
        st.markdown(f"""
**Source.** Directorate General of Civil Aviation (India) – *Monthly Statistics, Domestic Air Transport*, consolidated into CSV by
[Vonter/india-aviation-traffic](https://github.com/Vonter/india-aviation-traffic) (Open Database Licence). The bundled files cover
**{cov['first_month']} → {cov['last_month']}** ({cov['months_with_data']} months, {cov['city_pairs']:,} city pairs).

**Cleaning that mattered**
- **One history per city.** DGCA relabelled airports over time (`MUMBAI` → `MUMBAI (MUMBAI)` + `MUMBAI (NAVI MUMBAI)` from Jan-2026,
  Goa → Dabolim + Mopa, `AYODHYA INTERNATIONAL AIRPORT` → `AYODHYA`, `VISHAKHAPATNAM (VISAKHAPATNAM)` → `VISAKHAPATNAM`, ~25 spellings in all). Labels are normalised and duplicates summed.
- **Goa is excluded on purpose.** Goa pairs roughly double between Dec-2025 and Jan-2026 when DGCA starts listing Dabolim and Mopa
  separately – a reporting change, not demand – so any Goa forecast would be fiction.
- **Canonical pairs.** Each pair is stored once (A < B) with passengers in both directions.
- **Two months never published** ({', '.join(ins.get('estimated_months', []))}) are estimated (seasonal scaling outside COVID, linear inside) and flagged.
- **Independent validation.** City-pair totals were reconciled with DGCA's separate carrier-level totals: median gap
  **{cov['median_abs_reconciliation_gap_pct']:.3f}%**, worst month **{cov['max_abs_reconciliation_gap_pct']:.2f}%**.
""")
        rec = d["recon"]
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(x=rec["date"], y=rec["diff_pct"], marker_color=ui.C["sky"],
                               hovertemplate="%{x|%b %Y}: %{y:+.3f}%<extra></extra>"))
        ui.style(fig, 230, legend=False)
        fig.update_yaxes(ticksuffix="%", title="City-pair total vs carrier total")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Two independent DGCA tables agree to within a fraction of a percent in almost every month.")
    with tab2:
        st.markdown("""
**Forecasting.** Five models compete in a rolling-origin backtest (seasonal naive, seasonal naive + growth, Holt-Winters ETS,
a global direct multi-horizon XGBoost, and an ETS + XGBoost ensemble). The model with the lowest average WAPE across the 1/3/6/12-month
horizons produces the forecast. Prediction intervals are empirical quantiles of backtest log-errors.

**Opportunity score.** Percentile-ranked blend of forecast growth, realised growth, recent momentum, market size and forecast reliability –
weights are user-adjustable, and ranking stability is tested with 2,000 random re-weightings.

**What this project deliberately does *not* claim**
- **No capacity or revenue view.** DGCA does not publish route-level seats or fares, so there is no "capacity gap" or yield estimate.
  The score ranks *demand attractiveness*, not profitability.
- **Demand ≠ passengers carried.** Traffic is capped by seats; strong routes can be under-counted when planes are full.
- **Shocks are unforecastable.** Airline failures, airport closures and geopolitical events (e.g. the May-2025 closures visible on
  Jammu/Srinagar routes) show up as errors, and distort year-on-year comparisons for a year afterwards.
- **City pairs, not flights.** No schedules, no connecting-passenger split, no international traffic.
""")
    with tab3:
        st.markdown("The same data lives in a small SQLite database (`data/demandpulse.db`). Run a prepared query or write your own read-only `SELECT`.")
        queries = db.parse_queries(db.QUERY_FILE.read_text(encoding="utf-8"))
        name = st.selectbox("Prepared query", list(queries), format_func=lambda s: s.replace("_", " ").capitalize())
        sql = st.text_area("SQL", queries[name], height=230, key=f"sql_{name}")
        if st.button("▶ Run query", type="primary"):
            if not re.match(r"^\s*(with|select)\b", sql, re.I) or re.search(r"\b(insert|update|delete|drop|alter|attach|pragma|create|replace)\b", sql, re.I):
                st.error("Only read-only SELECT / WITH queries are allowed.")
            else:
                try:
                    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
                    st.dataframe(pd.read_sql_query(sql, con), width="stretch", hide_index=True)
                    con.close()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Query failed: {e}")
        st.caption("Tables: pairs_monthly, city_monthly, carrier_monthly, forecasts, opportunities, backtest_summary")
