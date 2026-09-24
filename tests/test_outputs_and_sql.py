import sqlite3

import pytest

from demandpulse import config, db


def test_forecast_shape_and_interval_order(outputs):
    fc = outputs["fc"]
    assert fc["market"].nunique() == config.TOP_N_MARKETS
    assert (fc.groupby("market").size() == config.FORECAST_HORIZON).all()
    assert ((fc["lower"] <= fc["forecast"]) & (fc["forecast"] <= fc["upper"])).all()
    assert (fc["forecast"] > 0).all()


def test_forecast_is_beyond_the_last_observation(outputs):
    fc = outputs["fc"]
    assert (fc["target_date"] > fc["origin"]).all()


def test_backtest_targets_are_after_origins(outputs):
    bt = outputs["bt"]
    assert (bt["target_date"] > bt["origin"]).all()
    assert bt["origin"].min() >= __import__("pandas").Timestamp(config.BACKTEST_FIRST_ORIGIN)


def test_opportunity_table_is_consistent(outputs):
    o = outputs["opp"]
    assert o["opportunity_score"].between(0, 100).all()
    assert sorted(o["rank"]) == list(range(1, len(o) + 1))
    assert o["top10_prob"].between(0, 1).all()


def test_all_prepared_sql_queries_run_and_return_rows(outputs):
    results = db.run_all()
    assert len(results) >= 8
    for name, df in results.items():
        assert len(df) > 0, name


def test_yearly_query_matches_pandas(outputs):
    import pandas as pd
    y = db.run_query("yearly_national_traffic")
    con = sqlite3.connect(config.DB_PATH)
    raw = pd.read_sql_query("SELECT date, pax FROM pairs_monthly", con)
    con.close()
    raw["year"] = raw["date"].str[:4].astype(int)
    expect = raw.groupby("year")["pax"].sum()
    got = y.set_index("year")["pax"]
    assert (got - expect.reindex(got.index)).abs().max() < 1


def test_query_parser():
    q = db.parse_queries("-- name: a\nSELECT 1;\n\n-- name: b\nSELECT 2;")
    assert q == {"a": "SELECT 1;", "b": "SELECT 2;"}
