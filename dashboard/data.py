"""Cached data loading for the dashboard (reads the files written by run_pipeline.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from demandpulse import config  # noqa: E402

REQUIRED = [config.PROCESSED_DIR / "market_monthly.csv", config.PREDICTIONS_DIR / "forecasts.csv",
            config.PREDICTIONS_DIR / "opportunities.csv", config.REPORTS_DIR / "insights.json"]


def outputs_ready() -> bool:
    return all(p.exists() for p in REQUIRED)


@st.cache_data(show_spinner=False)
def load_all() -> dict:
    P, R, O = config.PROCESSED_DIR, config.REPORTS_DIR, config.PREDICTIONS_DIR
    long = pd.read_csv(P / "market_monthly.csv", parse_dates=["date"])
    d = dict(
        long=long,
        wide=long.pivot(index="date", columns="market", values="pax"),
        national=pd.read_csv(P / "national_monthly.csv", parse_dates=["date"]),
        city=pd.read_csv(P / "city_monthly.csv", parse_dates=["date"]),
        carriers=pd.read_csv(P / "carrier_monthly.csv", parse_dates=["date"]),
        forecast=pd.read_csv(O / "forecasts.csv", parse_dates=["origin", "target_date"]),
        opp=pd.read_csv(O / "opportunities.csv"),
        bt=pd.read_csv(O / "backtest_results.csv", parse_dates=["origin", "target_date"]),
        bt_summary=pd.read_csv(R / "backtest_summary.csv"),
        coverage=pd.read_csv(R / "interval_coverage.csv"),
        recon=pd.read_csv(R / "reconciliation.csv", parse_dates=["date"]),
        insights=json.loads((R / "insights.json").read_text(encoding="utf-8")),
    )
    return d


def fmt_pax(x: float) -> str:
    if x is None or pd.isna(x):
        return "–"
    ax = abs(x)
    if ax >= 1e6:
        return f"{x/1e6:,.2f}M" if ax < 1e7 else f"{x/1e6:,.1f}M"
    if ax >= 1e3:
        return f"{x/1e3:,.0f}K"
    return f"{x:,.0f}"


def fmt_pct(x: float, signed: bool = True) -> str:
    if x is None or pd.isna(x):
        return "–"
    return f"{x*100:+.1f}%" if signed else f"{x*100:.1f}%"
