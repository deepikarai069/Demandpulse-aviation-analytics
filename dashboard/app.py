"""DemandPulse dashboard entry point:  streamlit run dashboard/app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st  # noqa: E402

import data  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="DemandPulse · India air demand", page_icon="✈️", layout="wide",
                   initial_sidebar_state="collapsed")
ui.inject_css()

if not data.outputs_ready():
    st.error("No model outputs found yet.")
    st.markdown("Run the pipeline once (takes about a minute):\n\n```bash\npython run_pipeline.py\n```")
    st.stop()

from views import about, forecast_lab, market, opportunities, overview, route_explorer  # noqa: E402

PAGES = [
    st.Page(overview.render, title="Overview", icon="🛫", url_path="overview", default=True),
    st.Page(route_explorer.render, title="Route explorer", icon="🗺️", url_path="routes"),
    st.Page(forecast_lab.render, title="Forecast lab", icon="🔬", url_path="forecast-lab"),
    st.Page(opportunities.render, title="Opportunities", icon="🎯", url_path="opportunities"),
    st.Page(market.render, title="Airlines & airports", icon="🏢", url_path="market"),
    st.Page(about.render, title="Data, method & SQL", icon="📚", url_path="about"),
]
try:
    nav = st.navigation(PAGES, position="top")
except TypeError:                       # older Streamlit: fall back to the sidebar
    nav = st.navigation(PAGES)
nav.run()
