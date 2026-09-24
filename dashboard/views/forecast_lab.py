"""Forecast Lab: transparent model comparison from the rolling-origin backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from data import load_all
from demandpulse import backtest, config

LABELS = backtest.MODEL_LABELS


def render():
    d = load_all()
    s, bt, cov, opp, ins = d["bt_summary"], d["bt"], d["coverage"], d["opp"], d["insights"]
    best = ins["model"]["selected"]
    ui.hero("Forecast lab",
            "How good is the forecast, really? Every number below comes from a rolling-origin backtest: models are "
            "re-trained at each origin using only data available then, and scored on the months that followed.",
            [(f"{bt['origin'].nunique()} forecast origins", "sky"), (f"{len(bt):,} out-of-sample forecasts", ""),
             (f"Selected: {LABELS[best]}", "green")])

    # ----- headline tiles
    piv = s.pivot(index="model", columns="horizon", values="wape")
    skill = s[s["model"] == best].set_index("horizon")["skill_vs_snaive"]
    k = st.columns(4)
    with k[0]:
        ui.kpi("1-month-ahead error", f"{piv.loc[best, 1]*100:.1f}%", f"{skill[1]*100:+.0f}% vs naive",
               "up", "WAPE, selected model", ui.C["sky"])
    with k[1]:
        ui.kpi("3-month-ahead error", f"{piv.loc[best, 3]*100:.1f}%", f"{skill[3]*100:+.0f}% vs naive", "up",
               "WAPE", ui.C["green"])
    with k[2]:
        ui.kpi("12-month-ahead error", f"{piv.loc[best, 12]*100:.1f}%", f"{skill[12]*100:+.0f}% vs naive",
               "up" if skill[12] >= 0.03 else "flat", "WAPE", ui.C["amber"])
    with k[3]:
        c80 = cov["coverage"].mean()
        ui.kpi("80% interval hit-rate", f"{c80*100:.0f}%", None, "flat",
               "honest, leave-one-origin-out", ui.C["violet"])

    if skill[12] < 0.03:
        tail = ("By 12 months ahead the model is barely better than 'same month last year' – demand shocks "
                "(airline failures, closures, weather) cannot be predicted from history alone.")
    else:
        tail = f"Even 12 months ahead the model still beats 'same month last year' by {skill[12]*100:.0f}%."
    st.info(f"**Reading this honestly:** at 1 month ahead the selected model cuts error by {skill[1]*100:.0f}% versus "
            f"a seasonal-naive baseline; at 3 months by {skill[3]*100:.0f}%. " + tail, icon="🧭")

    # ----- comparison charts
    left, right = st.columns(2)
    with left:
        ui.section("Error by horizon", "WAPE = total absolute error ÷ total passengers (lower is better).")
        fig = go.Figure()
        for m in backtest.MODEL_COLS:
            g = s[s["model"] == m].sort_values("horizon")
            fig.add_trace(go.Bar(x=[f"{h} mo" for h in g["horizon"]], y=g["wape"], name=LABELS[m],
                                 marker_color=ui.MODEL_COLORS[m], opacity=1 if m == best else 0.72,
                                 hovertemplate="%{x}: %{y:.1%}<extra>" + LABELS[m] + "</extra>"))
        ui.style(fig, 400)
        fig.update_layout(barmode="group")
        fig.update_yaxes(tickformat=".0%", title=None)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        ui.section("Skill vs seasonal naive", "Positive = better than repeating the same month last year.")
        fig = go.Figure()
        for m in ["ets", "xgb", "ensemble", "snaive_drift"]:
            g = s[s["model"] == m].sort_values("horizon")
            fig.add_trace(go.Scatter(x=[f"{h} mo" for h in g["horizon"]], y=g["skill_vs_snaive"], name=LABELS[m],
                                     mode="lines+markers", line=dict(color=ui.MODEL_COLORS[m], width=3 if m == best else 2),
                                     hovertemplate="%{x}: %{y:.0%}<extra>" + LABELS[m] + "</extra>"))
        fig.add_hline(y=0, line=dict(color="#64748B", dash="dot"))
        ui.style(fig, 400)
        fig.update_yaxes(tickformat=".0%", title=None)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ----- actual vs predicted
    ui.section("Actual vs predicted", "Every dot is one market-month forecast that the model never saw.")
    c1, c2 = st.columns([1, 3])
    with c1:
        h = st.select_slider("Horizon (months ahead)", options=list(config.BACKTEST_HORIZONS), value=3)
        model = st.selectbox("Model", backtest.MODEL_COLS, index=backtest.MODEL_COLS.index(best),
                             format_func=lambda m: LABELS[m])
        g = bt[bt["horizon"] == h]
        st.metric("WAPE", f"{backtest.wape(g['actual'], g[model])*100:.1f}%")
        st.metric("Bias", f"{backtest.bias(g['actual'], g[model])*100:+.1f}%",
                  help="Average over- (+) or under- (−) forecast.")
    with c2:
        lo, hi = g["actual"].min() * 0.9, g["actual"].max() * 1.1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#64748B", dash="dot"),
                                 name="Perfect forecast", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=g["actual"], y=g[model], mode="markers", name="Forecasts",
                                 marker=dict(size=6, color=ui.MODEL_COLORS[model], opacity=0.55),
                                 customdata=np.stack([g["market"].str.title(), g["target_date"].dt.strftime("%b %Y")], axis=-1),
                                 hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>actual %{x:,.0f}<br>"
                                               "forecast %{y:,.0f}<extra></extra>"))
        ui.style(fig, 400, legend=False)
        ticks = [30e3, 100e3, 300e3, 1e6, 3e6, 6e6]
        labels = ["30K", "100K", "300K", "1M", "3M", "6M"]
        fig.update_xaxes(type="log", title="Actual passengers", tickvals=ticks, ticktext=labels)
        fig.update_yaxes(type="log", title="Forecast passengers", tickvals=ticks, ticktext=labels)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ----- who is predictable
    l, r = st.columns(2)
    with l:
        ui.section("Bigger markets are easier to forecast")
        o = opp.copy()
        fig = go.Figure(go.Scatter(
            x=o["ttm_pax"], y=o["backtest_wape"], mode="markers",
            marker=dict(size=10, color=o["ttm_growth"], colorscale=[[0, ui.C["rose"]], [0.5, "#33455F"], [1, ui.C["green"]]],
                        cmin=-0.15, cmax=0.15, line=dict(width=1, color="#0B1220"),
                        colorbar=dict(title="TTM growth", tickformat=".0%", thickness=10)),
            text=o["market"].str.title(),
            hovertemplate="<b>%{text}</b><br>%{x:,.0f} pax · error %{y:.1%}<extra></extra>"))
        ui.style(fig, 380, legend=False)
        fig.update_xaxes(type="log", title="Passengers (TTM, log scale)", tickvals=[6e5, 1e6, 2e6, 4e6, 7e6],
                         ticktext=["600K", "1M", "2M", "4M", "7M"])
        fig.update_yaxes(tickformat=".0%", title="Backtest WAPE")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with r:
        ui.section("Prediction-interval calibration", "Share of actuals that landed inside the 80% band.")
        fig = go.Figure(go.Bar(x=[f"{h} mo" for h in cov["horizon"]], y=cov["coverage"],
                               marker_color=ui.C["violet"], text=[f"{v:.0%}" for v in cov["coverage"]],
                               textposition="inside", insidetextanchor="middle", textfont=dict(color="#0B1220", size=14),
                               hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
        fig.add_hline(y=config.INTERVAL_LEVEL, line=dict(color=ui.C["amber"], dash="dash"),
                      annotation_text="target 80%", annotation_position="top left", annotation_font_color=ui.C["amber"])
        ui.style(fig, 380, legend=False)
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    with st.expander("How the evaluation avoids fooling itself"):
        st.markdown("""
- **Rolling origin, expanding window.** At each origin the XGBoost model is re-trained on samples whose *target date is
  on or before the origin*; ETS is re-fitted on history up to the origin. Nothing from the future leaks in.
- **Direct multi-horizon.** One XGBoost model per horizon predicts `y[t+h] / level_t` from information known at `t`
  (13 look-back ratios, 3-month YoY, national momentum, calendar month) – no recursive error accumulation.
- **COVID handled explicitly.** Any training sample whose look-back or target touches Mar-2020 → Mar-2022 is dropped;
  ETS is fitted only on post-COVID months.
- **Fair baselines.** *Seasonal naive* (same month last year) is the bar to beat; a growth-adjusted variant is shown too.
- **Intervals are checked, not assumed.** Quantiles come from backtest residuals and coverage is measured
  leave-one-origin-out.
- **Caveat.** The backtest window (Apr-2024 →) includes real shocks – so errors here are realistic, not flattering.
""")
        st.dataframe((s.pivot(index="label", columns="horizon", values="wape") * 100).round(2)
                     .rename(columns=lambda c: f"h={c} (WAPE %)"), width="stretch")
