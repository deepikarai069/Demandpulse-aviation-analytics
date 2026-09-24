"""Central configuration. Every tunable lives here so the pipeline stays reproducible."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
MODELS_DIR = OUTPUT_DIR / "models"
DB_PATH = ROOT / "data" / "demandpulse.db"

CITY_PAIRS_FILE = RAW_DIR / "dgca_domestic_city_pairs.csv"
CARRIERS_FILE = RAW_DIR / "dgca_domestic_carriers.csv"

DATA_SOURCE_URLS = {
    CITY_PAIRS_FILE.name: "https://raw.githubusercontent.com/Vonter/india-aviation-traffic/main/aggregated/domestic/city.csv",
    CARRIERS_FILE.name: "https://raw.githubusercontent.com/Vonter/india-aviation-traffic/main/aggregated/domestic/carrier.csv",
}

# --- data preparation -------------------------------------------------------
# COVID-19 regime: months whose traffic is dominated by lock-downs / recovery.
# Samples whose look-back window or target touches this range are never used to
# train the ML model, and ETS is fitted only on data after it.
COVID_START = "2020-03-01"
COVID_END = "2022-03-01"
# Cities excluded from forecasting because DGCA's reporting basis changed (structural break, not demand):
# Goa pairs roughly double between Dec-2025 and Jan-2026 when DGCA starts listing Dabolim and Mopa separately.
EXCLUDED_CITIES = {"GOA": "Dabolim/Mopa reporting change – traffic roughly doubles in Jan-2026"}
TOP_N_MARKETS = 60          # city-pair markets modelled (largest by trailing-12-month pax)
MIN_POSTCOVID_COVERAGE = 0.95

# --- forecasting ------------------------------------------------------------
FORECAST_HORIZON = 12               # months ahead
BACKTEST_HORIZONS = (1, 3, 6, 12)   # horizons evaluated in the rolling-origin backtest
BACKTEST_FIRST_ORIGIN = "2024-04-01"
BACKTEST_STEP_MONTHS = 2
N_LAGS = 13                         # months of look-back fed to the ML model (t..t-12)
INTERVAL_LEVEL = 0.80               # prediction-interval coverage target

XGB_PARAMS = dict(
    n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.85,
    colsample_bytree=0.85, min_child_weight=3, reg_lambda=2.0,
    objective="reg:squarederror", random_state=42, n_jobs=1, verbosity=0,
)

# --- opportunity score ------------------------------------------------------
SCORE_WEIGHTS = {
    "forecast_growth": 0.30,   # expected next-12-month growth vs trailing 12 months
    "ttm_growth": 0.20,        # realised growth, trailing 12 vs prior 12 months
    "momentum": 0.15,          # last 3 months vs same 3 months a year ago
    "market_size": 0.20,       # trailing-12-month passengers
    "reliability": 0.15,       # narrow forecast interval = predictable demand
}
RANDOM_SEED = 42
