"""SQLite analytics layer: load processed tables and run the named queries in sql/analysis_queries.sql."""
from __future__ import annotations

import re
import sqlite3
import pandas as pd

from . import config

QUERY_FILE = config.ROOT / "sql" / "analysis_queries.sql"


def parse_queries(text: str) -> dict[str, str]:
    """Split the SQL file into {name: statement} using the '-- name:' markers."""
    out = {}
    for block in re.split(r"^-- name:\s*", text, flags=re.M)[1:]:
        name, _, body = block.partition("\n")
        out[name.strip()] = body.strip()
    return out


def build_database(pairs: pd.DataFrame, city: pd.DataFrame, carriers: pd.DataFrame,
                   forecasts: pd.DataFrame, opportunities: pd.DataFrame,
                   backtest_summary: pd.DataFrame, path=None):
    path = path or config.DB_PATH
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    def _dates_to_text(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()                                   # never mutate the caller's frames
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
        return df

    _dates_to_text(pairs).to_sql("pairs_monthly", con, index=False)
    _dates_to_text(city).to_sql("city_monthly", con, index=False)
    _dates_to_text(carriers).to_sql("carrier_monthly", con, index=False)
    _dates_to_text(forecasts).to_sql("forecasts", con, index=False)
    opportunities.drop(columns=["base_effect"], errors="ignore").to_sql("opportunities", con, index=False)
    backtest_summary.to_sql("backtest_summary", con, index=False)
    con.execute("CREATE INDEX idx_pairs_market_date ON pairs_monthly(market, date)")
    con.execute("CREATE INDEX idx_city_date ON city_monthly(city, date)")
    con.execute("CREATE INDEX idx_pairs_date ON pairs_monthly(date)")        # makes MAX(date) / range scans instant
    con.execute("CREATE INDEX idx_city_date_only ON city_monthly(date)")
    con.execute("CREATE INDEX idx_carrier_date ON carrier_monthly(date)")
    con.execute("ANALYZE")
    con.commit()
    con.close()


def run_query(name: str, path=None) -> pd.DataFrame:
    queries = parse_queries(QUERY_FILE.read_text(encoding="utf-8"))
    with sqlite3.connect(path or config.DB_PATH) as con:
        return pd.read_sql_query(queries[name], con)


def run_all(path=None) -> dict[str, pd.DataFrame]:
    queries = parse_queries(QUERY_FILE.read_text(encoding="utf-8"))
    with sqlite3.connect(path or config.DB_PATH) as con:
        return {n: pd.read_sql_query(q, con) for n, q in queries.items()}
