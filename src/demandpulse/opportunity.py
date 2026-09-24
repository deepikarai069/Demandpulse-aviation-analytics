"""Route Opportunity Score built only from observable, real signals.

DGCA does not publish route-level seats or fares, so the score deliberately does NOT
pretend to measure a capacity gap or revenue.  It ranks city-pair markets by how large,
how fast-growing and how predictable their demand is, and lets the user re-weight it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

COMPONENTS = list(config.SCORE_WEIGHTS)
COMPONENT_LABELS = {
    "forecast_growth": "Forecast growth (next 12m)",
    "ttm_growth": "Realised growth (TTM)",
    "momentum": "Recent momentum (3m YoY)",
    "market_size": "Market size (TTM pax)",
    "reliability": "Forecast reliability",
}


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True, method="average")


def market_metrics(wide: pd.DataFrame, forecast: pd.DataFrame, bt: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per market with raw signals and their percentile-rank components (0..1)."""
    last = wide.iloc[-12:]
    prior = wide.iloc[-24:-12]
    ttm, prev = last.sum(), prior.sum()
    mom = wide.iloc[-3:].sum() / wide.iloc[-15:-12].sum() - 1
    # year-ago comparison window itself depressed by a disruption? (e.g. May-Jul 2025 closures)
    yago_vs_2y = wide.iloc[-15:-12].sum() / wide.iloc[-27:-24].sum()
    fsum = forecast.groupby("market")["forecast"].sum()
    flo = forecast.groupby("market")["lower"].sum()
    fhi = forecast.groupby("market")["upper"].sum()
    err = (bt.assign(ae=(bt["actual"] - bt[model]).abs())
             .groupby("market").agg(ae=("ae", "sum"), act=("actual", "sum")))
    market_wape = (err["ae"] / err["act"]).rename("backtest_wape")

    seas = wide.iloc[-24:].groupby(wide.iloc[-24:].index.month).mean()
    seas_idx = seas / seas.mean()

    df = pd.DataFrame({
        "ttm_pax": ttm, "prior_pax": prev,
        "ttm_growth": ttm / prev - 1,
        "momentum": mom,
        "base_effect": (yago_vs_2y < 0.85) & (mom > 0.20),
        "forecast_12m": fsum.reindex(ttm.index),
        "forecast_12m_low": flo.reindex(ttm.index),
        "forecast_12m_high": fhi.reindex(ttm.index),
        "backtest_wape": market_wape.reindex(ttm.index),
        "peak_month": seas_idx.idxmax(),
        "trough_month": seas_idx.idxmin(),
        "seasonality_amp": seas_idx.max() - seas_idx.min(),
    })
    df["forecast_growth"] = df["forecast_12m"] / df["ttm_pax"] - 1
    df["market_size"] = df["ttm_pax"]
    df["reliability_raw"] = -df["backtest_wape"]
    df.index.name = "market"
    df = df.reset_index()
    parts = df["market"].str.split(" – ", expand=True)
    df["a"], df["b"] = parts[0], parts[1]
    for c in ["forecast_growth", "ttm_growth", "momentum", "market_size"]:
        df[f"{c}_score"] = _pct_rank(df[c])
    df["reliability_score"] = _pct_rank(df["reliability_raw"])
    return df


def score_with_weights(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Weighted 0-100 score.  Weights are re-normalised so they always sum to 1."""
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()
    out = df.copy()
    out["opportunity_score"] = 100 * sum(out[f"{c}_score"] * w[c] for c in COMPONENTS)
    out["rank"] = out["opportunity_score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def segment(df: pd.DataFrame) -> pd.DataFrame:
    """Transparent rule-based market segments with a suggested action.

    Thresholds are absolute (not percentile based) so a market is never called a
    'growth engine' merely because its peers are shrinking:
      fast   = mean(TTM growth, forecast growth) >= +3 %
      large  = top third of markets by trailing-12-month passengers
      soft   = forecast next-12m below current run-rate by more than 2 %
    """
    out = df.copy()
    g = (out["ttm_growth"] + out["forecast_growth"]) / 2
    fast = g >= 0.03
    large = out["market_size_score"] >= 0.67
    soft = out["forecast_growth"] < -0.02

    out["segment"] = np.select(
        [soft, fast & large, fast, large],
        ["Watch – softening", "Growth engine", "Emerging star", "Core trunk"],
        default="Steady")
    action = {
        "Growth engine": "Large and still growing – prioritise extra frequencies / up-gauging.",
        "Emerging star": "Smaller market growing fast – test capacity additions.",
        "Core trunk": "Large, growing slowly – defend share and optimise yield.",
        "Watch – softening": "Forecast below current run-rate – review capacity before adding.",
        "Steady": "Stable – maintain current schedule.",
    }
    out["suggested_action"] = out["segment"].map(action)
    return out


def rank_stability(df: pd.DataFrame, base_weights: dict[str, float] | None = None,
                   n_draws: int = 2000, top_k: int = 10, concentration: float = 60.0) -> pd.DataFrame:
    """Monte-Carlo robustness of the ranking to the (subjective) weights."""
    base = pd.Series(base_weights or config.SCORE_WEIGHTS, dtype=float)
    base = base[COMPONENTS] / base[COMPONENTS].sum()
    rng = np.random.default_rng(config.RANDOM_SEED)
    W = rng.dirichlet(np.clip(base.values * concentration, 0.05, None), size=n_draws)
    S = df[[f"{c}_score" for c in COMPONENTS]].values @ W.T           # (markets, draws)
    ranks = (-S).argsort(axis=0).argsort(axis=0) + 1
    return pd.DataFrame({
        "market": df["market"].values,
        f"top{top_k}_prob": (ranks <= top_k).mean(axis=1),
        "rank_p10": np.percentile(ranks, 10, axis=1),
        "rank_p90": np.percentile(ranks, 90, axis=1),
    })


def build_opportunities(wide, forecast, bt, model) -> pd.DataFrame:
    m = market_metrics(wide, forecast, bt, model)
    scored = score_with_weights(m, config.SCORE_WEIGHTS)
    scored = segment(scored)
    return scored.merge(rank_stability(m), on="market")
