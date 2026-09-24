import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import pytest

from demandpulse import config


@pytest.fixture(scope="session")
def pairs():
    from demandpulse import ingest
    return ingest.load_city_pairs()


@pytest.fixture(scope="session")
def panel(pairs):
    from demandpulse import ingest
    wide, missing = ingest.build_panel(pairs)
    return wide, missing


@pytest.fixture(scope="session")
def small_wide(panel):
    """A 12-market slice keeps model tests fast."""
    from demandpulse import ingest
    wide, _ = panel
    return wide[ingest.select_markets(wide, 12)]


@pytest.fixture(scope="session")
def outputs():
    need = [config.PREDICTIONS_DIR / "forecasts.csv", config.PREDICTIONS_DIR / "opportunities.csv",
            config.PREDICTIONS_DIR / "backtest_results.csv", config.DB_PATH]
    if not all(p.exists() for p in need):
        pytest.skip("run `python run_pipeline.py` first")
    return dict(fc=pd.read_csv(config.PREDICTIONS_DIR / "forecasts.csv", parse_dates=["origin", "target_date"]),
                opp=pd.read_csv(config.PREDICTIONS_DIR / "opportunities.csv"),
                bt=pd.read_csv(config.PREDICTIONS_DIR / "backtest_results.csv", parse_dates=["origin", "target_date"]))
