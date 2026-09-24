"""Overview: headline KPIs, demand + forecast, seasonality, route map and top markets."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from data import fmt_pax, fmt_pct, load_all
from demandpulse import config
from demandpulse.geo import CITY_COORDS


def _route_map(d: dict, top_n: int = 60) -> go.Figure:
    """Route network drawn on a plain lon/lat canvas: fully offline, no basemap or CDN needed."""
    opp = d["opp"].sort_values("ttm_pax", ascending=False).head(top_n)
    city = d["city"]
    last = city["date"].max()
    ttm = city[city["date"] > last - pd.DateOffset(months=12)].groupby("city")["pax"].sum()
    fig = go.Figure()
    q = opp["ttm_pax"].quantile([0.25, 0.5, 0.75]).values
    buckets = [(0, q[0], 0.8, 0.28), (q[0], q[1], 1.4, 0.36), (q[1], q[2], 2.4, 0.5), (q[2], 1e12, 4.0, 0.8)]
    for lo, hi, width, alpha in buckets:
        for r in opp[(opp["ttm_pax"] >= lo) & (opp["ttm_pax"] < hi)].itertuples():
            if r.a in CITY_COORDS and r.b in CITY_COORDS:
                (la, lo_a), (lb, lo_b) = CITY_COORDS[r.a], CITY_COORDS[r.b]
                # gentle curve so overlapping routes stay readable
                t = np.linspace(0, 1, 24)
                mx, my = (lo_a + lo_b) / 2, (la + lb) / 2
                dx, dy = lo_b - lo_a, lb - la
                bend = 0.10
                cx, cy = mx - dy * bend, my + dx * bend
                xs = (1 - t) ** 2 * lo_a + 2 * (1 - t) * t * cx + t ** 2 * lo_b
                ys = (1 - t) ** 2 * la + 2 * (1 - t) * t * cy + t ** 2 * lb
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines", showlegend=False, line=dict(width=width, color=f"rgba(56,189,248,{alpha})"),
                    text=[r.market.title()] * len(xs), hovertemplate="<b>%{text}</b><br>" + fmt_pax(r.ttm_pax) +
                    " passengers (TTM)<extra></extra>"))
    cities = [c for c in ttm.index if c in CITY_COORDS and c in set(opp["a"]) | set(opp["b"])]
    t = ttm[cities]
    fig.add_trace(go.Scatter(
        x=[CITY_COORDS[c][1] for c in cities], y=[CITY_COORDS[c][0] for c in cities],
        text=[c.title() for c in cities], mode="markers",
        marker=dict(size=np.clip(np.sqrt(t.values) / 110, 6, 32), color=ui.C["amber"], opacity=0.92,
                    line=dict(width=1.5, color="#0B1220")),
        customdata=[fmt_pax(v) for v in t.values],
        hovertemplate="<b>%{text}</b><br>%{customdata} passengers (TTM)<extra></extra>", showlegend=False))
    lab = t.sort_values(ascending=False).head(11).index
    pos = {"MUMBAI": "middle left", "PUNE": "bottom left", "AHMEDABAD": "middle left", "CHENNAI": "middle right",
           "BENGALURU": "bottom left", "KOLKATA": "middle right", "HYDERABAD": "bottom right", "KOCHI": "bottom center"}
    fig.add_trace(go.Scatter(
        x=[CITY_COORDS[c][1] for c in lab], y=[CITY_COORDS[c][0] for c in lab], text=[c.title() for c in lab],
        mode="text", textposition=[pos.get(c, "top center") for c in lab],
        textfont=dict(size=11, color="#E6EDF7"), hoverinfo="skip", showlegend=False))
    ui.style(fig, 560, legend=False)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), plot_bgcolor="rgba(17,27,46,.55)")
    fig.update_xaxes(range=[64, 102], visible=False, showgrid=False)
    fig.update_yaxes(range=[6.5, 37], visible=False, showgrid=False, scaleanchor="x", scaleratio=1.09)
    return fig


def render():
    d = load_all()
    ins, wide, fc, opp = d["insights"], d["wide"], d["forecast"], d["opp"]
    cov = ins["coverage"]
    ui.hero("DemandPulse · India domestic air demand",
            "Real monthly passenger data from DGCA, turned into 12-month forecasts with honest error bars "
            "and a transparent route-opportunity ranking.",
            [(f"Data through {pd.Timestamp(cov['last_month']).strftime('%b %Y')}", "sky"),
             (f"{cov['city_pairs']:,} city pairs · {cov['cities']} cities", ""),
             ("Source: DGCA (ODbL)", ""),
             (f"Model: {ins['model']['selected'].upper()} · backtested", "green")])

    # ---------------------------------------------------------------- KPIs
    top_share = wide.iloc[-12:].sum().sum() / d["national"]["pax"].iloc[-12:].sum()
    f12 = opp["forecast_12m"].sum() / opp["ttm_pax"].sum() - 1
    c = st.columns(5)
    with c[0]:
        ui.kpi("Passengers · last 12 mo", f"{ins['ttm']['pax_millions']:.1f}M",
               f"{ins['ttm']['growth_pct']:+.1f}%", ui.direction(ins['ttm']['growth_pct']), "vs prior 12 mo", ui.C["sky"])
    with c[1]:
        ui.kpi("2025 vs 2019", f"{ins.get('pct_of_2019_in_2025', 0):.0f}%", None, "flat",
               "of pre-COVID traffic", ui.C["green"])
    with c[2]:
        ui.kpi("Forecast · next 12 mo", fmt_pax(opp["forecast_12m"].sum()), fmt_pct(f12), ui.direction(f12),
               f"top {len(opp)} markets vs last 12 mo", ui.C["amber"])
    with c[3]:
        ui.kpi("Domestic load factor", f"{ins.get('domestic_plf_ttm_avg', 0):.1f}%", None, "flat",
               "scheduled carriers, TTM avg", ui.C["violet"])
    with c[4]:
        ui.kpi("Market concentration", f"{ins['top10_share_pct']:.0f}%", None, "flat",
               f"of traffic on the top 10 city pairs", ui.C["rose"])

    # ---------------------------------------------------- demand + forecast
    ui.section("Demand and 12-month outlook",
               f"Modelled markets = the {len(opp)} largest city pairs (≈{top_share*100:.0f}% of all domestic passengers).")
    left, right = st.columns([2.1, 1])
    with left:
        st.markdown("**Monthly passengers** · modelled markets, actual and forecast")
        act = wide.sum(axis=1)
        ff = fc.groupby("target_date")[["forecast"]].sum()
        fig = go.Figure()
        ui.covid_band(fig, config)
        fig.add_trace(go.Scatter(x=act.index, y=act.values, name="Actual", mode="lines",
                                 line=dict(color=ui.C["sky"], width=2.4),
                                 fill="tozeroy", fillcolor="rgba(56,189,248,.07)",
                                 hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>Actual</extra>"))
        fx = [act.index[-1]] + list(ff.index)
        fy = [act.values[-1]] + list(ff["forecast"].values)
        fig.add_trace(go.Scatter(x=fx, y=fy, name="Forecast", mode="lines",
                                 line=dict(color=ui.C["amber"], width=2.6, dash="dash"),
                                 hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>Forecast</extra>"))
        ui.style(fig, 400)
        fig.update_layout(legend=dict(orientation="h", x=1, xanchor="right", y=1.06, bgcolor="rgba(0,0,0,0)"))
        fig.update_yaxes(tickformat=".2s", title=None)
        fig.update_xaxes(rangeselector=dict(buttons=[
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(count=6, label="6Y", step="year", stepmode="backward"), dict(step="all", label="All")],
            bgcolor="#111B2E", activecolor="#22314D", font=dict(color="#C9D6EC")))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        nat = d["national"]
        nat = nat[nat["date"].between("2023-01-01", "2025-12-01")]
        idx = nat.groupby(nat["date"].dt.month)["pax"].mean()
        idx = idx / idx.mean()
        colors = [ui.C["amber"] if v == idx.max() else ui.C["rose"] if v == idx.min() else "#2F4A73" for v in idx.values]
        fig = go.Figure(go.Bar(x=ui.MONTH_NAMES, y=idx.values, marker_color=colors,
                               hovertemplate="%{x}: %{y:.2f}× average month<extra></extra>"))
        ui.style(fig, 400, legend=False)
        fig.update_yaxes(range=[0.85, 1.12], tickformat=".2f", title=None)
        st.markdown("**Seasonality** · demand vs an average month (2023–25)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ------------------------------------------------------- map + top pairs
    ui.section("Where the passengers fly",
               "Each curve is a city pair (thicker = more passengers, last 12 months); bubbles = airport traffic. Schematic map – positions are approximate.")
    m1, m2 = st.columns([1.0, 1.0])
    with m1:
        st.plotly_chart(_route_map(d), width="stretch", config={"displayModeBar": False})
    with m2:
        top = opp.sort_values("ttm_pax", ascending=False).head(12).iloc[::-1]
        fig = go.Figure(go.Bar(
            y=top["market"].str.title().str.replace(" – ", " ↔ "), x=top["ttm_pax"], orientation="h",
            marker=dict(color=top["ttm_growth"], colorscale=[[0, ui.C["rose"]], [0.5, "#33455F"], [1, ui.C["green"]]],
                        cmin=-0.15, cmax=0.15, colorbar=dict(title="TTM growth", tickformat=".0%", len=0.5, thickness=10)),
            customdata=np.stack([top["ttm_growth"] * 100], axis=-1),
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} pax · %{customdata[0]:+.1f}% YoY<extra></extra>"))
        ui.style(fig, 560, legend=False)
        fig.update_xaxes(tickformat=".2s")
        st.markdown("**Twelve biggest city pairs** · colour = year-on-year growth")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ------------------------------------------------------------- findings
    ui.section("What the data says")
    a, b, c3 = st.columns(3)
    tm = ins["top_markets"][0]
    with a:
        ui.callout(f"{tm['market'].title().replace(' – ', ' ↔ ')}",
                   f"is the biggest domestic market: <b>{tm['pax_millions']:.1f}M</b> passengers "
                   f"({tm['share_pct']}% of traffic) in the last 12 months.", ui.C["sky"])
    with b:
        s = ins["seasonality"]
        ui.callout(f"{s['peak_month']} peaks, {s['trough_month']} dips",
                   f"Demand runs about <b>{(s['peak_index']-1)*100:.0f}% above</b> an average month in "
                   f"{s['peak_month']} and <b>{(1-s['trough_index'])*100:.0f}% below</b> in {s['trough_month']}.", ui.C["amber"])
    with c3:
        share = ins["airline_share_pct"]
        first = next(iter(share))
        ui.callout(f"{first} carries {share[first]:.0f}%",
                   "of domestic passengers; the top two carriers together move "
                   f"<b>{sum(list(share.values())[:2]):.0f}%</b>.", ui.C["violet"])
