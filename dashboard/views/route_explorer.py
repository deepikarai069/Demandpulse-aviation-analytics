"""Route Explorer: history, forecast with 80 % band, seasonality heat-map and directional split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from data import fmt_pax, fmt_pct, load_all
from demandpulse import config


def render():
    d = load_all()
    wide, fc, opp, long = d["wide"], d["forecast"], d["opp"].set_index("market"), d["long"]
    ui.hero("Route explorer", "Pick a city pair to see its history, seasonality and 12-month forecast "
            "with an 80% prediction interval calibrated on the backtest.")

    order = opp.sort_values("ttm_pax", ascending=False).index.tolist()
    c1, c2 = st.columns([2, 1])
    with c1:
        market = st.selectbox("City pair", order, index=0, format_func=lambda m: m.title().replace(" – ", " ↔ "))
    with c2:
        view = st.segmented_control("History window", ["3Y", "6Y", "All"], default="6Y") or "6Y"

    row = opp.loc[market]
    f = fc[fc["market"] == market].sort_values("horizon")
    seg_color = {"Growth engine": "green", "Emerging star": "sky", "Core trunk": "violet",
                 "Steady": "", "Watch – softening": "rose"}[row["segment"]]
    st.markdown(f'<span class="chip {seg_color}">{row["segment"]}</span> '
                f'<span class="chip">Opportunity rank #{int(row["rank"])} of {len(opp)}</span> '
                f'<span class="chip amber">Peak month: {ui.MONTH_NAMES[int(row["peak_month"])-1]}</span> '
                f'<span class="chip">Trough: {ui.MONTH_NAMES[int(row["trough_month"])-1]}</span>'
                + (' <span class="chip rose">Year-ago base was disrupted – growth may be overstated</span>'
                   if bool(row["base_effect"]) else ""), unsafe_allow_html=True)
    st.write("")

    k = st.columns(5)
    with k[0]:
        ui.kpi("Passengers · TTM", fmt_pax(row["ttm_pax"]), fmt_pct(row["ttm_growth"]),
               ui.direction(row["ttm_growth"]), "vs prior 12 mo", ui.C["sky"])
    with k[1]:
        ui.kpi("Next 12 months", fmt_pax(row["forecast_12m"]), fmt_pct(row["forecast_growth"]),
               ui.direction(row["forecast_growth"]), "vs TTM", ui.C["amber"])
    with k[2]:
        ui.kpi("Range (80%)", f"{fmt_pax(row['forecast_12m_low'])}–{fmt_pax(row['forecast_12m_high'])}",
               None, "flat", "sum of monthly bounds*", ui.C["violet"])
    with k[3]:
        ui.kpi("Recent momentum", fmt_pct(row["momentum"]), None, ui.direction(row["momentum"]),
               "last 3 mo vs a year ago", ui.C["green"])
    with k[4]:
        ui.kpi("Backtest error", f"{row['backtest_wape']*100:.1f}%", None, "flat",
               "avg. WAPE, this market", ui.C["rose"])

    # --------------------------------------------------- history + forecast
    ui.section("History and forecast")
    s = wide[market]
    start = {"3Y": s.index[-1] - pd.DateOffset(years=3), "6Y": s.index[-1] - pd.DateOffset(years=6)}.get(view, s.index[0])
    s = s[s.index >= start]
    fig = go.Figure()
    if start < pd.Timestamp(config.COVID_END):
        ui.covid_band(fig, config)
    fig.add_trace(go.Scatter(x=f["target_date"], y=f["upper"], mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=f["target_date"], y=f["lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(251,191,36,.16)", name="80% interval",
                             hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>Lower</extra>"))
    fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Actual", mode="lines+markers",
                             line=dict(color=ui.C["sky"], width=2.4), marker=dict(size=4),
                             hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>Actual</extra>"))
    fig.add_trace(go.Scatter(x=[s.index[-1]] + list(f["target_date"]), y=[s.values[-1]] + list(f["forecast"]),
                             name="Forecast", mode="lines+markers",
                             line=dict(color=ui.C["amber"], width=2.6, dash="dash"), marker=dict(size=5),
                             hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>Forecast</extra>"))
    ui.style(fig, 420)
    fig.update_yaxes(tickformat=".2s", rangemode="tozero")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("*The monthly bounds are added for the KPI, which slightly overstates the true 12-month uncertainty.")

    # ------------------------------------------ heat-map + directional split
    l, r = st.columns(2)
    with l:
        ui.section("Seasonality pattern", "Each cell vs that year's monthly average (COVID months hidden).")
        h = wide[market].to_frame("pax")
        h["year"], h["month"] = h.index.year, h.index.month
        h.loc[(h.index >= config.COVID_START) & (h.index <= config.COVID_END), "pax"] = np.nan
        h = h[h["year"] >= 2016]
        piv = h.pivot(index="year", columns="month", values="pax")
        rel = piv.div(piv.mean(axis=1), axis=0)
        fig = go.Figure(go.Heatmap(
            z=rel.values, x=ui.MONTH_NAMES, y=rel.index.astype(str), customdata=piv.values,
            colorscale=[[0, "#16233B"], [0.5, "#2A6F9E"], [1, "#BAE6FD"]], zmin=0.8, zmax=1.2,
            hovertemplate="%{y} %{x}<br>%{customdata:,.0f} pax<br>%{z:.2f}× year avg<extra></extra>",
            colorbar=dict(thickness=10, len=0.7)))
        ui.style(fig, 400, legend=False)
        fig.update_yaxes(autorange="reversed", type="category", dtick=1)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with r:
        ui.section("Direction of travel", "Passengers each way, last 36 months.")
        m = long[(long["market"] == market)].sort_values("date").tail(36)
        a, b = market.split(" – ")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=m["date"], y=m["pax_a_to_b"], name=f"{a.title()} → {b.title()}", marker_color=ui.C["sky"]))
        fig.add_trace(go.Bar(x=m["date"], y=m["pax_b_to_a"], name=f"{b.title()} → {a.title()}", marker_color=ui.C["violet"]))
        ui.style(fig, 400)
        fig.update_layout(barmode="stack")
        fig.update_yaxes(tickformat=".2s")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("A gap means DGCA did not publish that month (May 2024).")

    # ------------------------------------------------------ forecast table
    ui.section("Month-by-month forecast")
    t = f[["target_date", "forecast", "lower", "upper"]].copy()
    ly = wide[market]
    t["same month last year"] = [ly.get(x - pd.DateOffset(years=1), np.nan) for x in t["target_date"]]
    t["vs last year"] = (t["forecast"] / t["same month last year"] - 1) * 100
    t["Month"] = t["target_date"].dt.strftime("%b %Y")
    t = t[["Month", "forecast", "lower", "upper", "same month last year", "vs last year"]]
    st.dataframe(t, hide_index=True, width="stretch", column_config={
        "forecast": st.column_config.NumberColumn("Forecast", format="localized"),
        "lower": st.column_config.NumberColumn("Lower (80%)", format="localized"),
        "upper": st.column_config.NumberColumn("Upper (80%)", format="localized"),
        "same month last year": st.column_config.NumberColumn("Same month LY", format="localized"),
        "vs last year": st.column_config.NumberColumn("vs LY", format="%+.1f%%"),
    })
