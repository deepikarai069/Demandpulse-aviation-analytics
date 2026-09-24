"""Look & feel: CSS, KPI cards, chips and a shared Plotly style."""
from __future__ import annotations

import streamlit as st

C = dict(bg="#0B1220", panel="#111B2E", border="#22314D", text="#E6EDF7", muted="#8A9BB8",
         sky="#38BDF8", amber="#FBBF24", green="#34D399", rose="#FB7185", violet="#A78BFA",
         slate="#94A3B8")
SEGMENT_COLORS = {"Growth engine": C["green"], "Emerging star": C["sky"], "Core trunk": C["violet"],
                  "Steady": C["slate"], "Watch – softening": C["rose"]}
MODEL_COLORS = {"snaive": "#64748B", "snaive_drift": "#94A3B8", "ets": C["violet"],
                "xgb": C["sky"], "ensemble": C["green"]}
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stHeader"] { background: rgba(11,18,32,.85); backdrop-filter: blur(8px); }
[data-testid="stAppDeployButton"], #MainMenu { display: none !important; }
footer { visibility: hidden; }

.hero { padding: 26px 30px; border-radius: 20px; margin-bottom: 18px;
  background: radial-gradient(1200px 300px at 0% 0%, rgba(56,189,248,.20), transparent 60%),
              radial-gradient(900px 300px at 100% 100%, rgba(167,139,250,.16), transparent 60%),
              linear-gradient(135deg, #111B2E, #0E1729);
  border: 1px solid #22314D; }
.hero h1 { margin: 0; font-size: 2.05rem; font-weight: 800;
  background: linear-gradient(90deg, #E6EDF7, #7DD3FC); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: 6px 0 0 0; color: #9FB0CB; font-size: .98rem; max-width: 900px; }
.hero .meta { margin-top: 14px; display: flex; gap: 8px; flex-wrap: wrap; }

.chip { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: .76rem; font-weight: 600;
  border: 1px solid #22314D; background: rgba(148,163,184,.10); color: #C9D6EC; }
.chip.sky { background: rgba(56,189,248,.14); color: #7DD3FC; border-color: rgba(56,189,248,.35); }
.chip.green { background: rgba(52,211,153,.13); color: #6EE7B7; border-color: rgba(52,211,153,.35); }
.chip.rose { background: rgba(251,113,133,.13); color: #FDA4AF; border-color: rgba(251,113,133,.35); }
.chip.amber { background: rgba(251,191,36,.13); color: #FCD34D; border-color: rgba(251,191,36,.35); }
.chip.violet { background: rgba(167,139,250,.14); color: #C4B5FD; border-color: rgba(167,139,250,.35); }

.kpi { background: linear-gradient(150deg, #121D31, #0F1A2D); border: 1px solid #22314D; border-radius: 16px;
  padding: 16px 18px 14px 18px; height: 100%; position: relative; overflow: hidden; }
.kpi:before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px; background: var(--accent, #38BDF8); }
.kpi .label { color: #8A9BB8; font-size: .72rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.kpi .value { font-size: 1.85rem; font-weight: 800; margin: 4px 0 2px 0; color: #F1F5FD; line-height: 1.15; }
.kpi .sub { color: #8A9BB8; font-size: .8rem; }
.kpi .delta { font-weight: 700; font-size: .84rem; margin-right: 6px; }
.kpi .delta.up { color: #34D399; } .kpi .delta.down { color: #FB7185; } .kpi .delta.flat { color: #94A3B8; }

.sec { display: flex; align-items: center; gap: 10px; margin: 26px 0 4px 0; }
.sec:before { content: ""; width: 4px; height: 20px; border-radius: 4px; background: linear-gradient(#38BDF8, #A78BFA); }
.sec span { font-size: 1.12rem; font-weight: 700; color: #E6EDF7; }
.secsub { color: #8A9BB8; font-size: .86rem; margin: 0 0 8px 14px; }

.callout { background: #101A2C; border: 1px solid #22314D; border-left: 3px solid var(--accent, #38BDF8);
  border-radius: 12px; padding: 14px 16px; height: 100%; }
.callout b { color: #E6EDF7; } .callout { color: #B7C4DC; font-size: .92rem; line-height: 1.5; }
.callout .big { font-size: 1.35rem; font-weight: 800; color: #F1F5FD; display: block; margin-bottom: 2px; }

.rank-card { background: linear-gradient(150deg, #121D31, #0F1A2D); border: 1px solid #22314D; border-radius: 16px; padding: 16px 18px; height: 100%; }
.rank-card .n { font-size: .75rem; color: #8A9BB8; font-weight: 700; letter-spacing: .08em; }
.rank-card .m { font-size: 1.12rem; font-weight: 800; color: #F1F5FD; margin: 2px 0 6px 0; }
.rank-card .s { font-size: 2rem; font-weight: 800; color: #7DD3FC; line-height: 1; }
.rank-card .a { color: #9FB0CB; font-size: .84rem; margin-top: 8px; }

div[data-testid="stTabs"] button[role="tab"] { font-weight: 600; }
div[data-testid="stExpander"] { border-radius: 12px; border-color: #22314D; }
.stDataFrame { border-radius: 12px; }
a { color: #7DD3FC !important; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips: list[tuple[str, str]] | None = None):
    chip_html = "".join(f'<span class="chip {c}">{t}</span>' for t, c in (chips or []))
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><div class="meta">{chip_html}</div></div>',
                unsafe_allow_html=True)


def section(title: str, sub: str | None = None):
    st.markdown(f'<div class="sec"><span>{title}</span></div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="secsub">{sub}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str | None = None, direction: str = "flat",
        sub: str | None = None, accent: str = "#38BDF8"):
    d = f'<span class="delta {direction}">{delta}</span>' if delta else ""
    s = f'<span class="sub">{sub}</span>' if sub else ""
    st.markdown(f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
                f'<div class="value">{value}</div><div>{d}{s}</div></div>', unsafe_allow_html=True)


def callout(big: str, text: str, accent: str = "#38BDF8"):
    st.markdown(f'<div class="callout" style="--accent:{accent}"><span class="big">{big}</span>{text}</div>',
                unsafe_allow_html=True)


def direction(x: float, tol: float = 0.0005) -> str:
    return "up" if x > tol else "down" if x < -tol else "flat"


def style(fig, height: int = 380, legend: bool = True):
    """Apply the shared dark theme to a Plotly figure."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color="#C9D6EC", size=12),
        margin=dict(l=6, r=6, t=30, b=6), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0F1A2D", bordercolor="#22314D", font_color="#E6EDF7"),
        colorway=[C["sky"], C["amber"], C["green"], C["violet"], C["rose"], C["slate"]],
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False, linecolor="#22314D")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False, linecolor="#22314D")
    return fig


def covid_band(fig, config):
    """Shade the COVID regime so nobody mistakes it for a normal pattern."""
    import pandas as pd
    fig.add_vrect(x0=pd.Timestamp(config.COVID_START), x1=pd.Timestamp(config.COVID_END),
                  fillcolor="rgba(148,163,184,.10)", line_width=0, layer="below",
                  annotation_text="COVID", annotation_position="top left",
                  annotation_font=dict(size=10, color="#8A9BB8"))
    return fig
