"""Route Opportunity Score: interactive re-weighting, ranking, segments and rank stability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from data import fmt_pax, load_all
from demandpulse import config, opportunity

DEFAULTS = {k: int(v * 100) for k, v in config.SCORE_WEIGHTS.items()}
HELP = {
    "forecast_growth": "Expected next-12-month passengers vs the last 12 months (model forecast).",
    "ttm_growth": "Realised growth: last 12 months vs the 12 months before.",
    "momentum": "Last 3 months vs the same 3 months a year earlier.",
    "market_size": "Trailing-12-month passengers – bigger markets matter more.",
    "reliability": "How well the model has forecast this market in the backtest (lower error = higher score).",
}


def _reset():
    for k, v in DEFAULTS.items():
        st.session_state[f"w_{k}"] = v


@st.cache_data(show_spinner=False)
def _stability(df: pd.DataFrame, weights: tuple) -> pd.DataFrame:
    return opportunity.rank_stability(df, dict(weights))


def render():
    d = load_all()
    base = d["opp"].drop(columns=["rank", "opportunity_score", "top10_prob", "rank_p10", "rank_p90"], errors="ignore")
    ui.hero("Route opportunity ranking",
            "Which city pairs deserve attention? A transparent 0–100 score built from real demand signals – "
            "change the weights and the ranking updates instantly.",
            [("No black box: 5 inputs, percentile-ranked", "sky"), ("Rank stability tested by simulation", "green")])

    # ---------------------------------------------------------------- weights
    with st.expander("⚖️  Tune the score weights", expanded=True):
        cols = st.columns(5)
        w = {}
        for col, key in zip(cols, opportunity.COMPONENTS):
            with col:
                w[key] = st.slider(opportunity.COMPONENT_LABELS[key], 0, 100, DEFAULTS[key], 5,
                                   key=f"w_{key}", help=HELP[key])
        st.button("Reset to defaults", on_click=_reset)
    if sum(w.values()) == 0:
        st.warning("All weights are zero – set at least one above 0.")
        return
    total = sum(w.values())
    st.caption("Effective weights: " + " · ".join(f"{opportunity.COMPONENT_LABELS[k]} {v/total:.0%}" for k, v in w.items()))

    scored = opportunity.score_with_weights(base, w)
    stab = _stability(scored, tuple(w.items()))
    scored = scored.merge(stab, on="market")

    # ------------------------------------------------------------- top 3 cards
    top = scored.head(3)
    cards = st.columns(3)
    for col, r in zip(cards, top.itertuples()):
        with col:
            chip = ui.SEGMENT_COLORS[r.segment]
            st.markdown(
                f'<div class="rank-card"><div class="n">RANK #{r.rank}</div>'
                f'<div class="m">{r.market.title().replace(" – ", " ↔ ")}</div>'
                f'<div class="s">{r.opportunity_score:.1f}<span style="font-size:.9rem;color:#8A9BB8"> / 100</span></div>'
                f'<div style="margin-top:8px"><span class="chip" style="color:{chip};border-color:{chip}55">{r.segment}</span> '
                f'<span class="chip">TTM {fmt_pax(r.ttm_pax)}</span> '
                f'<span class="chip">top-10 in {r.top10_prob:.0%} of sims</span></div>'
                f'<div class="a">{r.suggested_action}</div></div>', unsafe_allow_html=True)

    # ------------------------------------------------------------------ table
    ui.section("Full ranking")
    f1, f2 = st.columns([3, 1])
    with f1:
        segs = st.multiselect("Segments", list(ui.SEGMENT_COLORS), default=list(ui.SEGMENT_COLORS))
    with f2:
        st.write("")
        st.write("")
        st.download_button("⬇ Download CSV", scored.drop(columns=["base_effect"], errors="ignore")
                           .to_csv(index=False).encode("utf-8"), "route_opportunities.csv", "text/csv")
    t = scored[scored["segment"].isin(segs)].copy()
    t["Route"] = t["market"].str.title().str.replace(" – ", " ↔ ")
    t["Base effect"] = np.where(t["base_effect"], "⚠", "")
    for c in ["ttm_growth", "forecast_growth", "momentum", "backtest_wape", "top10_prob"]:
        t[c] = t[c] * 100
    show = t[["rank", "Route", "segment", "opportunity_score", "ttm_pax", "ttm_growth", "forecast_growth",
              "momentum", "backtest_wape", "top10_prob", "Base effect"]]
    st.dataframe(show, hide_index=True, width="stretch", height=440, column_config={
        "rank": st.column_config.NumberColumn("#", width="small"),
        "segment": st.column_config.TextColumn("Segment"),
        "opportunity_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
        "ttm_pax": st.column_config.NumberColumn("Passengers (TTM)", format="localized"),
        "ttm_growth": st.column_config.NumberColumn("TTM growth", format="%+.1f%%"),
        "forecast_growth": st.column_config.NumberColumn("Forecast growth", format="%+.1f%%"),
        "momentum": st.column_config.NumberColumn("3-mo momentum", format="%+.1f%%"),
        "backtest_wape": st.column_config.NumberColumn("Backtest error", format="%.1f%%"),
        "top10_prob": st.column_config.ProgressColumn("Top-10 stability", min_value=0, max_value=100, format="%.0f%%"),
        "Base effect": st.column_config.TextColumn("Base effect", width="small"),
    })
    st.caption("⚠ = base effect: the year-ago comparison window was itself depressed (e.g. May–Jul 2025 closures), "
               "so momentum / growth can look inflated.")

    # ----------------------------------------------------------------- charts
    l, r = st.columns(2)
    with l:
        ui.section("Growth landscape", "Bubble size = passengers. Top-right = growing now and expected to keep growing.")
        fig = go.Figure()
        for seg, color in ui.SEGMENT_COLORS.items():
            g = scored[scored["segment"] == seg]
            if g.empty:
                continue
            fig.add_trace(go.Scatter(
                x=g["ttm_growth"], y=g["forecast_growth"], mode="markers", name=seg,
                marker=dict(size=np.sqrt(g["ttm_pax"]) / 75, color=color, opacity=0.78,
                            line=dict(width=1, color="#0B1220"), sizemode="diameter"),
                text=g["market"].str.title(),
                hovertemplate="<b>%{text}</b><br>TTM growth %{x:+.1%}<br>forecast growth %{y:+.1%}<extra>" + seg + "</extra>"))
        fig.add_hline(y=0, line=dict(color="#475569", dash="dot"))
        fig.add_vline(x=0, line=dict(color="#475569", dash="dot"))
        ui.style(fig, 470)
        fig.update_xaxes(tickformat=".0%", title="Realised growth (TTM)")
        fig.update_yaxes(tickformat=".0%", title="Forecast growth (next 12m)")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with r:
        ui.section("How stable is the top 15?", "Range of ranks across 2,000 random re-weightings (10th–90th percentile).")
        tp = scored.head(15).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tp["rank_p90"], y=tp["market"].str.title().str.replace(" – ", " ↔ "), mode="markers", showlegend=False,
            marker=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            error_x=dict(type="data", symmetric=False, array=np.zeros(len(tp)), arrayminus=(tp["rank_p90"] - tp["rank_p10"]).values,
                         color="#38BDF8", thickness=4, width=0)))
        fig.add_trace(go.Scatter(x=tp["rank"], y=tp["market"].str.title().str.replace(" – ", " ↔ "), mode="markers",
                                 name="Your ranking", marker=dict(size=11, color=ui.C["amber"], line=dict(width=1.5, color="#0B1220")),
                                 hovertemplate="rank %{x}<extra></extra>"))
        ui.style(fig, 470)
        fig.update_xaxes(title="Rank (1 = best)", autorange="reversed")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
