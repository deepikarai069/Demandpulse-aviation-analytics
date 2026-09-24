"""Turn the real data + model results into plain-language findings (JSON + Markdown)."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from . import config

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def compute_insights(pairs, carriers, recon, bt_summary, coverage, best_model, opp, forecast, city, national) -> dict:
    ins: dict = {}
    months = pairs["date"].drop_duplicates().sort_values()
    last = months.iloc[-1]
    ins["coverage"] = dict(
        first_month=months.iloc[0].strftime("%Y-%m"), last_month=last.strftime("%Y-%m"),
        months_with_data=int(len(months)), cities=int(len(set(pairs["a"]) | set(pairs["b"]))),
        city_pairs=int(pairs["market"].nunique()),
        max_abs_reconciliation_gap_pct=float(recon["diff_pct"].abs().max()),
        median_abs_reconciliation_gap_pct=float(recon["diff_pct"].abs().median()))

    nat = national.copy()                       # includes the two months DGCA never published (estimated)
    ymo = nat.groupby(nat["date"].dt.year).agg(pax=("pax", "sum"), m=("date", "count"))
    complete = ymo[ymo["m"] == 12]["pax"]
    ins["yearly_pax_millions"] = {int(y): round(v / 1e6, 1) for y, v in complete.items()}
    ins["estimated_months"] = [d.strftime("%Y-%m") for d in nat.loc[nat["is_imputed"], "date"]]
    if 2019 in complete.index and 2025 in complete.index:
        ins["pct_of_2019_in_2025"] = round(100 * complete[2025] / complete[2019], 1)
    tot = pairs.groupby("date")["pax"].sum()
    ttm, prev = tot.iloc[-12:].sum(), tot.iloc[-24:-12].sum()
    ins["ttm"] = dict(pax_millions=round(ttm / 1e6, 1), growth_pct=round(100 * (ttm / prev - 1), 1))

    mk = pairs[pairs["date"] > last - pd.DateOffset(months=12)].groupby("market")["pax"].sum().sort_values(ascending=False)
    share = mk / mk.sum()
    ins["top_markets"] = [dict(market=m, pax_millions=round(v / 1e6, 2), share_pct=round(100 * share[m], 1))
                          for m, v in mk.head(5).items()]
    ins["top10_share_pct"] = round(100 * share.head(10).sum(), 1)
    ins["markets_for_half_of_traffic"] = int((share.cumsum() < 0.5).sum() + 1)

    ct = city[city["date"] > last - pd.DateOffset(months=12)].groupby("city")["pax"].sum().sort_values(ascending=False)
    ins["top5_city_airport_share_pct"] = round(100 * ct.head(5).sum() / ct.sum(), 1)
    ins["top_cities"] = list(ct.head(5).index)

    nat_s = pairs[pairs["date"].between("2023-01-01", "2025-12-01")].groupby("date")["pax"].sum()
    idx = nat_s.groupby(nat_s.index.month).mean()
    idx = idx / idx.mean()
    ins["seasonality"] = dict(peak_month=MONTHS[int(idx.idxmax()) - 1], peak_index=round(float(idx.max()), 2),
                              trough_month=MONTHS[int(idx.idxmin()) - 1], trough_index=round(float(idx.min()), 2))

    c = carriers[(carriers["airline"] != "Total Domestic") & (carriers["date"] > last - pd.DateOffset(months=12))]
    cs = c.groupby("airline")["pax"].sum().sort_values(ascending=False)
    cs = cs[cs > 0]
    ins["airline_share_pct"] = {k: round(100 * v / cs.sum(), 1) for k, v in cs.head(4).items()}
    plf = carriers[(carriers["airline"] == "Total Domestic") & (carriers["date"] > last - pd.DateOffset(months=12))]
    if len(plf):
        ins["domestic_plf_ttm_avg"] = round(float(plf["plf"].mean()), 1)

    piv = bt_summary.pivot(index="model", columns="horizon", values="wape")
    ins["model"] = dict(
        selected=best_model,
        wape_pct={m: {int(h): round(100 * v, 1) for h, v in row.items()} for m, row in piv.iterrows()},
        skill_vs_snaive_pct={int(r.horizon): round(100 * r.skill_vs_snaive, 1)
                             for r in bt_summary[bt_summary["model"] == best_model].itertuples()},
        interval_coverage_pct={int(r.horizon): round(100 * r.coverage, 1) for r in coverage.itertuples()})

    f12 = forecast.groupby("market")["forecast"].sum().sum()
    ttm_top = opp["ttm_pax"].sum()
    ins["forecast_next12m_vs_ttm_pct_top60"] = round(100 * (f12 / ttm_top - 1), 1)
    ins["top_opportunities"] = [dict(rank=int(r.rank), market=r.market, score=round(float(r.opportunity_score), 1),
                                     segment=r.segment) for r in opp.head(5).itertuples()]
    ins["segment_counts"] = opp["segment"].value_counts().to_dict()
    return ins


def to_markdown(ins: dict) -> str:
    cov, m = ins["coverage"], ins["model"]
    L = ["# DemandPulse – key findings (auto-generated from the real DGCA data)", ""]
    L += [f"**Data:** {cov['months_with_data']} months ({cov['first_month']} → {cov['last_month']}), "
          f"{cov['cities']} cities, {cov['city_pairs']} city pairs. City-pair totals reconcile with DGCA's own "
          f"carrier totals (median gap {cov['median_abs_reconciliation_gap_pct']:.3f}%, max {cov['max_abs_reconciliation_gap_pct']:.2f}%).", ""]
    y = ins["yearly_pax_millions"]
    L += ["## Demand", "",
          "- Passengers per calendar year (millions): " + ", ".join(f"{k}: {v}" for k, v in y.items())
          + f" (months DGCA never published are estimated: {', '.join(ins.get('estimated_months', []))}).",
          f"- 2025 traffic = {ins.get('pct_of_2019_in_2025', '?')}% of 2019.",
          f"- Last 12 months: {ins['ttm']['pax_millions']} M passengers ({ins['ttm']['growth_pct']:+.1f}% vs the 12 months before).",
          f"- Top 10 city pairs carry {ins['top10_share_pct']}% of traffic; {ins['markets_for_half_of_traffic']} pairs make up half of it.",
          f"- Five busiest airports ({', '.join(ins['top_cities'])}) handle {ins['top5_city_airport_share_pct']}% of airport traffic.",
          f"- Seasonality: peak {ins['seasonality']['peak_month']} (index {ins['seasonality']['peak_index']}), "
          f"trough {ins['seasonality']['trough_month']} (index {ins['seasonality']['trough_index']}).", ""]
    L += ["## Airlines", "",
          "- Share of domestic passengers (TTM): " + ", ".join(f"{k} {v}%" for k, v in ins["airline_share_pct"].items()) + "."]
    if "domestic_plf_ttm_avg" in ins:
        L += [f"- Average scheduled-domestic passenger load factor (TTM): {ins['domestic_plf_ttm_avg']}%."]
    L += ["", "## Forecast accuracy (rolling-origin backtest, WAPE %)", "",
          f"Selected model: **{m['selected']}**.", "",
          "| Model | " + " | ".join(f"h={h}" for h in sorted(next(iter(m['wape_pct'].values())))) + " |",
          "|---|" + "---|" * len(next(iter(m['wape_pct'].values())))]
    for name, row in m["wape_pct"].items():
        L.append(f"| {name} | " + " | ".join(f"{v}" for _, v in sorted(row.items())) + " |")
    L += ["", "Skill vs seasonal naive (%): " + ", ".join(f"h={h}: {v:+.1f}" for h, v in m["skill_vs_snaive_pct"].items()),
          "", "80% interval coverage (leave-one-origin-out): " + ", ".join(f"h={h}: {v}%" for h, v in m["interval_coverage_pct"].items()), ""]
    L += ["## Opportunity ranking", ""] + [f"{o['rank']}. {o['market']} – score {o['score']} ({o['segment']})" for o in ins["top_opportunities"]]
    return "\n".join(L) + "\n"


def write_reports(ins: dict):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "insights.json").write_text(json.dumps(ins, indent=2, default=str), encoding="utf-8")
    (config.REPORTS_DIR / "insights.md").write_text(to_markdown(ins), encoding="utf-8")
