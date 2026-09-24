"""End-to-end pipeline:  real DGCA data  ->  clean panel  ->  backtest  ->  forecast  ->  scores  ->  SQLite.

Usage:
    python run_pipeline.py                 # use the bundled data (takes ~1 minute)
    python run_pipeline.py --refresh-data  # download the latest DGCA-derived CSVs first
"""
import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from demandpulse import backtest, config, db, forecast, ingest, insights, opportunity  # noqa: E402


def step(msg):
    print(f"\n▶ {msg}", flush=True)
    return time.time()


def refresh_data():
    for name, url in config.DATA_SOURCE_URLS.items():
        print(f"  downloading {name}")
        urllib.request.urlretrieve(url, config.RAW_DIR / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-data", action="store_true", help="re-download the raw CSVs from GitHub")
    args = ap.parse_args()
    for d in (config.PROCESSED_DIR, config.REPORTS_DIR, config.PREDICTIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.refresh_data:
        t = step("Refreshing raw data"); refresh_data(); print(f"  done in {time.time()-t:.1f}s")

    t = step("1/5  Ingest & clean real DGCA city-pair data")
    r = ingest.run_ingest()
    wide = r["long"].pivot(index="date", columns="market", values="pax")
    print(f"  {len(r['pairs']):,} pair-months | {len(r['markets'])} modelled markets | "
          f"months imputed: {[m.strftime('%Y-%m') for m in r['missing']]} | {time.time()-t:.1f}s")

    t = step("2/5  Rolling-origin backtest (5 models x 4 horizons)")
    bt = backtest.run_backtest(wide, imputed_months=r["missing"])
    summary = backtest.summarise(bt)
    best = backtest.pick_best_model(summary)
    coverage = backtest.loo_origin_coverage(bt, best)
    bt.to_csv(config.PREDICTIONS_DIR / "backtest_results.csv", index=False)
    summary.to_csv(config.REPORTS_DIR / "backtest_summary.csv", index=False)
    coverage.to_csv(config.REPORTS_DIR / "interval_coverage.csv", index=False)
    print(summary.pivot(index="label", columns="horizon", values="wape").mul(100).round(1).to_string())
    print(f"  selected model: {best} | {time.time()-t:.1f}s")

    t = step("3/5  12-month forecast with prediction intervals")
    fc = forecast.make_forecast(wide, bt, best)
    fc.to_csv(config.PREDICTIONS_DIR / "forecasts.csv", index=False)
    print(f"  {len(fc):,} forecast rows | {time.time()-t:.1f}s")

    t = step("4/5  Route Opportunity Score + sensitivity analysis")
    opp = opportunity.build_opportunities(wide, fc, bt, best)
    opp.to_csv(config.PREDICTIONS_DIR / "opportunities.csv", index=False)
    print(opp[["rank", "market", "opportunity_score", "segment"]].head(5).round(1).to_string(index=False))

    t = step("5/5  SQLite layer + findings report")
    city = pd.read_csv(config.PROCESSED_DIR / "city_monthly.csv", parse_dates=["date"])
    db.build_database(r["pairs"], city, r["carriers"], fc, opp, summary)
    ins = insights.compute_insights(r["pairs"], r["carriers"],
                                    pd.read_csv(config.REPORTS_DIR / "reconciliation.csv"),
                                    summary, coverage, best, opp, fc, city,
                                    pd.read_csv(config.PROCESSED_DIR / "national_monthly.csv", parse_dates=["date"]))
    insights.write_reports(ins)
    print(f"  wrote {config.DB_PATH.name}, insights.md/json")
    print(f"\n✔ Pipeline finished in {time.time()-t0:.0f}s. Launch the app:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
