"""Airlines & airports: carrier share, load factors and airport ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from data import fmt_pax, load_all
from demandpulse import config

EXCLUDE = {"Total Domestic", "Total International"}
AIRLINE_COLORS = ["#38BDF8", "#FBBF24", "#34D399", "#A78BFA", "#FB7185", "#F97316", "#22D3EE", "#94A3B8", "#64748B"]


def render():
    d = load_all()
    car, city = d["carriers"], d["city"]
    car = car[~car["airline"].isin(EXCLUDE)]
    last = car["date"].max()
    ui.hero("Airlines & airports", "Who carries the passengers, how full are the planes and which airports are growing "
            "– straight from DGCA's carrier statistics.")

    # ---------------------------------------------------------- airline share
    ttm = car[car["date"] > last - pd.DateOffset(months=12)].groupby("airline")["pax"].sum().sort_values(ascending=False)
    top = [a for a in ttm.index if ttm[a] > 0][:6]
    ui.section("Market share by airline", "Share of scheduled domestic passengers. Defunct carriers (Jet Airways, Go First, "
               "Vistara after its 2024 merger) fade out of the picture.")
    hist_top = (car[car["date"] >= "2015-04-01"].groupby("airline")["pax"].sum().sort_values(ascending=False).head(9).index.tolist())
    keep = list(dict.fromkeys(hist_top + top))
    p = car[car["date"] >= "2015-04-01"].pivot_table(index="date", columns="airline", values="pax", aggfunc="sum").fillna(0)
    p["Others"] = p.drop(columns=[c for c in keep if c in p.columns], errors="ignore").sum(axis=1)
    p = p[[c for c in keep if c in p.columns] + ["Others"]]
    tot = p.sum(axis=1)
    share = p.div(tot, axis=0).where(tot > 0.05 * tot.median())      # hide the near-zero lockdown months
    fig = go.Figure()
    for i, col in enumerate(share.columns):
        fig.add_trace(go.Scatter(x=share.index, y=share[col], name=col, mode="lines", stackgroup="one",
                                 line=dict(width=0.5, color=AIRLINE_COLORS[i % len(AIRLINE_COLORS)]),
                                 fillcolor=AIRLINE_COLORS[i % len(AIRLINE_COLORS)],
                                 hovertemplate="%{x|%b %Y}: %{y:.1%}<extra>" + col + "</extra>"))
    ui.covid_band(fig, config)
    ui.style(fig, 430)
    fig.update_yaxes(tickformat=".0%", range=[0, 1], title=None)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ------------------------------------------------------ load factor + KPIs
    l, r = st.columns([1.6, 1])
    with l:
        ui.section("Passenger load factor", "Share of seats filled (12-month moving average, post-2019).")
        fig = go.Figure()
        for i, a in enumerate(top[:5]):
            g = car[(car["airline"] == a) & (car["date"] >= "2019-01-01") & (car["plf"] > 0)].sort_values("date")
            if g.empty:
                continue
            fig.add_trace(go.Scatter(x=g["date"], y=g["plf"].rolling(12, min_periods=6).mean(), name=a, mode="lines",
                                     line=dict(width=2.4, color=AIRLINE_COLORS[i]),
                                     hovertemplate="%{x|%b %Y}: %{y:.1f}%<extra>" + a + "</extra>"))
        ui.covid_band(fig, config)
        ui.style(fig, 400)
        fig.update_yaxes(ticksuffix="%", title=None)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with r:
        ui.section("Last 12 months")
        tot = ttm.sum()
        for a in top[:4]:
            plf = car[(car["airline"] == a) & (car["date"] > last - pd.DateOffset(months=12)) & (car["plf"] > 0)]["plf"].mean()
            ui.kpi(a, f"{ttm[a]/tot*100:.1f}%", None, "flat", f"{fmt_pax(ttm[a])} pax · load factor {plf:.0f}%",
                   AIRLINE_COLORS[top.index(a)])
            st.write("")

    # ------------------------------------------------------------- airports
    ui.section("Airport ranking", "Arrivals + departures on domestic city pairs (all airports serving the city).")
    lc = city["date"].max()
    cur = city[city["date"] > lc - pd.DateOffset(months=12)].groupby("city")["pax"].sum()
    prev = city[(city["date"] <= lc - pd.DateOffset(months=12)) & (city["date"] > lc - pd.DateOffset(months=24))].groupby("city")["pax"].sum()
    a = pd.DataFrame({"ttm": cur, "prev": prev}).dropna()
    a = a.drop(index=[c for c in config.EXCLUDED_CITIES if c in a.index])   # reporting break, see About page
    a["growth"] = a["ttm"] / a["prev"] - 1
    top15 = a.sort_values("ttm", ascending=False).head(15).iloc[::-1]
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(y=top15.index.str.title(), x=top15["ttm"], orientation="h",
                               marker=dict(color=top15["growth"], colorscale=[[0, ui.C["rose"]], [0.5, "#33455F"], [1, ui.C["green"]]],
                                           cmin=-0.15, cmax=0.15, colorbar=dict(title="YoY", tickformat=".0%", thickness=10)),
                               customdata=top15["growth"] * 100,
                               hovertemplate="<b>%{y}</b><br>%{x:,.0f} pax · %{customdata:+.1f}% YoY<extra></extra>"))
        ui.style(fig, 470, legend=False)
        fig.update_xaxes(tickformat=".2s")
        st.markdown("**15 busiest airports** · colour = year-on-year growth")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with c2:
        cities = a.sort_values("ttm", ascending=False).index.tolist()
        pick = st.selectbox("Airport / city", cities, index=0, format_func=str.title)
        s = city[city["city"] == pick].set_index("date")["pax"].sort_index()
        fig = go.Figure()
        ui.covid_band(fig, config)
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=ui.C["sky"], width=2.4),
                                 fill="tozeroy", fillcolor="rgba(56,189,248,.08)",
                                 hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra></extra>"))
        ui.style(fig, 420, legend=False)
        fig.update_yaxes(tickformat=".2s")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Mumbai combines CSMIA + Navi Mumbai so it has one continuous history. Goa is left out: DGCA's Jan-2026 switch to separate Dabolim/Mopa reporting roughly doubles its figures.")
