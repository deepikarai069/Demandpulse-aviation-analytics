import numpy as np
import pandas as pd

from demandpulse import config, ingest


def test_city_name_normalisation():
    n = ingest.normalise_city
    assert n("MUMBAI (MUMBAI)") == n("MUMBAI (NAVI MUMBAI)") == n("MUMBAI") == "MUMBAI"
    assert n("GOA (DABOLIM, SOUTH GOA)") == n("GOA (MOPA, NORTH GOA)") == n("MOPA, GOA") == "GOA"
    assert n("AYODHYA INTERNATIONAL AIRPORT") == "AYODHYA"
    assert n("PURNIA AIRPORT\xa0") == "PURNIA"
    assert n("  delhi ") == "DELHI"


def test_pairs_are_canonical_and_unique(pairs):
    assert (pairs["a"] < pairs["b"]).all()
    assert not pairs.duplicated(["date", "a", "b"]).any()
    assert (pairs[["pax_a_to_b", "pax_b_to_a", "pax"]] >= 0).all().all()
    assert np.allclose(pairs["pax"], pairs["pax_a_to_b"] + pairs["pax_b_to_a"])


def test_swap_keeps_directions_consistent(tmp_path):
    csv = tmp_path / "c.csv"
    csv.write_text("Year,Month,City1,City2,PaxToCity2,PaxFromCity2,FreightToCity2,FreightFromCity2,MailToCity2,MailFromCity2\n"
                   "2024,01,PUNE,DELHI,100,40,0,0,0,0\n")
    df = ingest.load_city_pairs(csv)
    r = df.iloc[0]
    assert (r["a"], r["b"]) == ("DELHI", "PUNE")
    assert (r["pax_a_to_b"], r["pax_b_to_a"]) == (40, 100)     # Delhi->Pune is the old "from City2"


def test_split_airports_are_merged(pairs):
    m = pairs[pairs["a"].eq("DELHI") & pairs["b"].eq("MUMBAI")]
    assert m["date"].is_unique                                   # one row per month after merging
    assert m["date"].max() >= pd.Timestamp("2026-01-01")         # continuous across the Jan-2026 split


def test_panel_is_complete_and_missing_months_flagged(panel):
    wide, missing = panel
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    assert wide.index.equals(full)
    assert [m.strftime("%Y-%m") for m in missing] == ["2021-05", "2024-05"]
    assert (wide.loc[missing].sum(axis=1) > 0).all()
    assert not wide.isna().any().any()


def test_imputed_month_is_between_plausible_bounds(panel):
    wide, _ = panel
    may24 = wide.loc["2024-05-01"].sum()
    neighbours = wide.loc[["2024-04-01", "2024-06-01"]].sum(axis=1)
    assert 0.9 * neighbours.min() < may24 < 1.1 * neighbours.max()


def test_city_pair_totals_reconcile_with_carrier_totals(pairs):
    rec = ingest.reconcile_with_carriers(pairs, ingest.load_carriers())
    assert len(rec) > 100
    assert rec["diff_pct"].abs().max() < 1.0


def test_market_selection_requires_complete_postcovid_history(panel):
    wide, _ = panel
    markets = ingest.select_markets(wide, 30)
    assert len(markets) == 30
    post = wide.loc["2022-04-01":, markets]
    assert ((post > 0).mean() >= config.MIN_POSTCOVID_COVERAGE).all().all()


def test_covid_mask():
    m = ingest.covid_mask(["2019-12-01", "2020-03-01", "2021-06-01", "2022-03-01", "2022-04-01"])
    assert m.tolist() == [False, True, True, True, False]


def test_more_spelling_variants_resolve():
    n = ingest.normalise_city
    assert n("DABOLIM") == "GOA"
    assert n("VISHAKHAPATNAM (VISAKHAPATNAM)") == n("VISAKHAPATNAM") == "VISAKHAPATNAM"
    assert n("MANGALORE (MANGALURU)") == n("MANGALORE") == "MANGALURU"
    assert n("TRIVANDRUM") == "THIRUVANANTHAPURAM" and n("DEHRA DUN") == "DEHRADUN"
    assert n("HIRASAR (RAJKOT)") == n("RAJKOT INTERNATIONAL AIRPORT") == "RAJKOT"


def test_no_modelled_city_is_a_fragment_of_another_label(panel):
    wide, _ = panel
    markets = ingest.select_markets(wide, config.TOP_N_MARKETS)
    cities = {c for m in markets for c in m.split(" – ")}
    for c in cities:
        assert "(" not in c and "AIRPORT" not in c and c == c.strip(), c
    assert not {"DABOLIM", "VIJAYWADA", "TRIVANDRUM", "MANGALORE"} & cities


def test_cities_with_a_reporting_break_are_excluded_from_modelling(panel):
    wide, _ = panel
    markets = ingest.select_markets(wide, 200)
    assert not [m for m in markets if "GOA" in m.split(" – ")]
