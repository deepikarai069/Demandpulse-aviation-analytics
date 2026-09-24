"""Load and clean the real DGCA monthly domestic city-pair traffic.

Source: Directorate General of Civil Aviation (India), "Monthly Statistics – Domestic
Air Transport", consolidated into CSV by Vivek Matthew (Vonter/india-aviation-traffic,
ODbL licence).  Each row is one city pair in one month with passengers in both
directions.

Cleaning steps (all deliberate, all tested):
  1. normalise city labels  – DGCA renamed/split airports over time
     ("MUMBAI" -> "MUMBAI (MUMBAI)" + "MUMBAI (NAVI MUMBAI)" from Jan-2026,
      "GOA" -> Dabolim + Mopa, "AYODHYA INTERNATIONAL AIRPORT" -> "AYODHYA" ...)
     so every metro has ONE continuous history;
  2. put each pair in canonical (alphabetical) order, swapping directions to match;
  3. sum duplicate keys created by (1) and by a double-reported month (Sep-2015);
  4. reconcile the pair totals against DGCA's own carrier-level totals.
"""
from __future__ import annotations

import re
import pandas as pd
import numpy as np

from . import config

_CITY_ALIASES = {
    # DGCA re-labelled or split airports over time; every variant maps to ONE continuous city.
    "MUMBAI (MUMBAI)": "MUMBAI", "MUMBAI (NAVI MUMBAI)": "MUMBAI",
    "GOA (DABOLIM, SOUTH GOA)": "GOA", "GOA (MOPA, NORTH GOA)": "GOA", "MOPA, GOA": "GOA", "DABOLIM": "GOA",
    "MANGALORE": "MANGALURU", "MANGALORE (MANGALURU)": "MANGALURU",
    "VISHAKHAPATNAM (VISAKHAPATNAM)": "VISAKHAPATNAM",
    "VIJAYWADA": "VIJAYAWADA", "VIJAYAWADA (GANNAVARAM)": "VIJAYAWADA",
    "DEHRA DUN": "DEHRADUN", "TRIVANDRUM": "THIRUVANANTHAPURAM", "CALICUT": "KOZHIKODE",
    "TIRUCHIRAPALLY": "TIRUCHIRAPPALLI", "ALLAHABAD": "PRAYAGRAJ",
    "HIRASAR (RAJKOT)": "RAJKOT", "HINDON": "GHAZIABAD", "NASIK": "NASHIK", "PONDICHERRY": "PUDUCHERRY",
    "MYSORE": "MYSURU", "CUDDAPAH": "KADAPA",
    "AYODHYA INTERNATIONAL AIRPORT": "AYODHYA", "KUSHINAGAR INTERNATIONAL AIRPORT": "KUSHINAGAR",
    "RAJKOT INTERNATIONAL AIRPORT": "RAJKOT", "BIDAR AIRPORT, KARNATAKA": "BIDAR",
    "HOLLONGI AIRPORT, ITANAGAR": "ITANAGAR",
}


def normalise_city(name: str) -> str:
    """Canonical upper-case city label with airport boilerplate stripped."""
    if not isinstance(name, str):
        return name
    s = name.replace("\xa0", " ").strip().upper()
    s = re.sub(r"\s+", " ", s)
    if s in _CITY_ALIASES:
        return _CITY_ALIASES[s]
    s = re.sub(r"\s*(INTERNATIONAL )?AIRPORT\s*$", "", s).strip()
    return _CITY_ALIASES.get(s, s)


def load_city_pairs(path=None) -> pd.DataFrame:
    """Return one row per (month, city_a, city_b) with a<b, pax in each direction."""
    raw = pd.read_csv(path or config.CITY_PAIRS_FILE)
    df = raw.rename(columns={"City1": "a", "City2": "b", "PaxToCity2": "pax_a_to_b",
                             "PaxFromCity2": "pax_b_to_a"})
    df["a"] = df["a"].map(normalise_city)
    df["b"] = df["b"].map(normalise_city)
    df = df[df["a"] != df["b"]].copy()                       # a merge can create self-pairs
    swap = df["a"] > df["b"]
    df.loc[swap, ["a", "b", "pax_a_to_b", "pax_b_to_a"]] = df.loc[
        swap, ["b", "a", "pax_b_to_a", "pax_a_to_b"]].values
    df["date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))
    df[["pax_a_to_b", "pax_b_to_a"]] = df[["pax_a_to_b", "pax_b_to_a"]].fillna(0).clip(lower=0)
    df = (df.groupby(["date", "a", "b"], as_index=False)[["pax_a_to_b", "pax_b_to_a"]].sum())
    df["market"] = df["a"] + " – " + df["b"]
    df["pax"] = df["pax_a_to_b"] + df["pax_b_to_a"]
    return df.sort_values(["market", "date"]).reset_index(drop=True)


def load_carriers(path=None) -> pd.DataFrame:
    """Scheduled-domestic monthly carrier statistics (pax, seat-km, passenger load factor)."""
    c = pd.read_csv(path or config.CARRIERS_FILE)
    c = c[c["Type"] == "ScheduledDomestic"].copy()
    c["date"] = pd.to_datetime(dict(year=c["Year"], month=c["Month"], day=1))
    c = c.rename(columns={"Airline": "airline", "Passenger Number": "pax",
                          "Passenger Load Factor": "plf", "Seat Kilometers": "seat_km",
                          "Passenger Kilometers": "pax_km", "Aircraft Number": "aircraft"})
    return c[["date", "airline", "pax", "plf", "seat_km", "pax_km", "aircraft"]].reset_index(drop=True)


def reconcile_with_carriers(pairs: pd.DataFrame, carriers: pd.DataFrame) -> pd.DataFrame:
    """Compare city-pair totals with DGCA's 'Total Domestic' carrier row, month by month."""
    a = pairs.groupby("date")["pax"].sum().rename("citypair_pax")
    b = carriers[carriers["airline"] == "Total Domestic"].groupby("date")["pax"].sum().rename("carrier_pax")
    r = pd.concat([a, b], axis=1, sort=True).dropna()
    r["diff_pct"] = (r["citypair_pax"] / r["carrier_pax"] - 1) * 100
    return r.reset_index()


# --------------------------------------------------------------------------- panel
def covid_mask(dates) -> np.ndarray:
    d = pd.to_datetime(dates)
    return np.asarray((d >= pd.Timestamp(config.COVID_START)) & (d <= pd.Timestamp(config.COVID_END)))


def _impute_missing_month(wide: pd.DataFrame, month: pd.Timestamp) -> pd.Series:
    """Estimate a month that DGCA never published (2021-05 and 2024-05).

    Outside COVID we scale the same month of the neighbouring years by the growth
    observed in the adjacent months; inside COVID we interpolate linearly.
    """
    prev_m, next_m = month - pd.DateOffset(months=1), month + pd.DateOffset(months=1)
    lin = (wide.loc[prev_m] + wide.loc[next_m]) / 2
    if bool(covid_mask([month])[0]):
        return lin
    ya, yn = month - pd.DateOffset(years=1), month + pd.DateOffset(years=1)
    if ya in wide.index and yn in wide.index:
        adj = wide.loc[prev_m] + wide.loc[next_m]
        g1 = adj / (wide.loc[ya - pd.DateOffset(months=1)] + wide.loc[ya + pd.DateOffset(months=1)]).replace(0, np.nan)
        g2 = adj / (wide.loc[yn - pd.DateOffset(months=1)] + wide.loc[yn + pd.DateOffset(months=1)]).replace(0, np.nan)
        est = 0.5 * (wide.loc[ya] * g1 + wide.loc[yn] * g2)
        return est.fillna(lin)
    return lin


def build_panel(pairs: pd.DataFrame):
    """Complete monthly grid (market x month) with the missing months imputed & flagged.

    Returns (wide, imputed_months) where wide is indexed by month, columns = markets.
    """
    wide = pairs.pivot_table(index="date", columns="market", values="pax", aggfunc="sum")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="MS")
    missing = [m for m in full if m not in wide.index]
    wide = wide.reindex(full)
    for m in missing:
        wide.loc[m] = _impute_missing_month(wide.fillna(0), m).values
    return wide.fillna(0.0), missing


def select_markets(wide: pd.DataFrame, top_n: int | None = None) -> list[str]:
    """Largest markets by trailing-12-month traffic that have a complete post-COVID history."""
    top_n = top_n or config.TOP_N_MARKETS
    post = wide.loc[pd.Timestamp(config.COVID_END) + pd.DateOffset(months=1):]
    coverage = (post > 0).mean()
    ttm = wide.iloc[-12:].sum()
    eligible = ttm[coverage[coverage >= config.MIN_POSTCOVID_COVERAGE].index]
    # cities whose reporting basis changed (structural break) would poison a demand forecast
    bad = [m for m in eligible.index if any(c in config.EXCLUDED_CITIES for c in m.split(" – "))]
    eligible = eligible.drop(bad)
    return eligible.sort_values(ascending=False).head(top_n).index.tolist()


def city_totals(pairs: pd.DataFrame) -> pd.DataFrame:
    """Airport-level (city) monthly passengers = arrivals + departures over all pairs."""
    a = pairs.assign(city=pairs["a"])[["date", "city", "pax"]]
    b = pairs.assign(city=pairs["b"])[["date", "city", "pax"]]
    return pd.concat([a, b]).groupby(["date", "city"], as_index=False)["pax"].sum()


def run_ingest():
    """Build every processed table used downstream and persist them."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_city_pairs()
    carriers = load_carriers()
    wide, missing = build_panel(pairs)
    markets = select_markets(wide)
    meta = pairs.drop_duplicates("market").set_index("market")[["a", "b"]]

    long = (wide[markets].stack().rename("pax").reset_index()
            .rename(columns={"level_0": "date", "level_1": "market"}))
    long = long.merge(meta, left_on="market", right_index=True)
    long["is_imputed"] = long["date"].isin(missing)
    long["is_covid"] = covid_mask(long["date"])
    long = long.merge(pairs[["date", "market", "pax_a_to_b", "pax_b_to_a"]], on=["date", "market"], how="left")

    national = pd.DataFrame({"date": wide.index, "pax": wide.sum(axis=1).values})
    national["is_imputed"] = national["date"].isin(missing)

    long.to_csv(config.PROCESSED_DIR / "market_monthly.csv", index=False)
    national.to_csv(config.PROCESSED_DIR / "national_monthly.csv", index=False)
    city_totals(pairs).to_csv(config.PROCESSED_DIR / "city_monthly.csv", index=False)
    carriers.to_csv(config.PROCESSED_DIR / "carrier_monthly.csv", index=False)
    reconcile_with_carriers(pairs, carriers).to_csv(config.REPORTS_DIR / "reconciliation.csv", index=False)
    return dict(pairs=pairs, wide=wide, missing=missing, markets=markets, long=long,
                national=national, carriers=carriers)
