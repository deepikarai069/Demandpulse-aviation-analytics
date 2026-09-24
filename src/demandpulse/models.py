"""Forecasting models: naive baselines, ETS (Holt-Winters) and a global XGBoost model.

Design notes
------------
* Monthly data, per-market series (city pairs).  ``wide`` is a (month x market) frame.
* The XGBoost model is *direct multi-horizon*: one model per horizon h, predicting
  ``y[t+h] / level_t`` from information available at the forecast origin ``t`` only.
  This avoids the error accumulation of recursive forecasting.
* Anything touching the COVID window is excluded from training samples.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

from . import config

warnings.filterwarnings("ignore")

FEATURE_COLS = ([f"r{k}" for k in range(config.N_LAGS)] +
                ["yoy3", "nat_yoy3", "target_month", "log_level", "share_nat", "post_covid"])
MIN_ORIGIN_POS = 14      # need y[t-14] for the 3-month year-on-year feature


# --------------------------------------------------------------------- baselines
def snaive(hist: np.ndarray, h: int) -> float:
    """Seasonal naive: the value 12 months before the target month."""
    return float(hist[len(hist) - 1 + h - 12])


def snaive_drift(hist: np.ndarray, h: int) -> float:
    """Seasonal naive scaled by trailing-12-month growth (clipped to a sane range)."""
    last12, prev12 = hist[-12:].sum(), hist[-24:-12].sum()
    g = float(np.clip(last12 / prev12, 0.7, 1.5)) if prev12 > 0 else 1.0
    return snaive(hist, h) * g


def ets_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Damped-trend, multiplicative-seasonal Holt-Winters fitted on post-COVID data only.

    Falls back to seasonal-naive when there is too little clean history or the fit fails.
    """
    y = series.loc[pd.Timestamp(config.COVID_END) + pd.DateOffset(months=1):].astype(float)
    hist = series.values
    fallback = np.array([snaive(hist, h) for h in range(1, horizon + 1)])
    if len(y) < 24 or y.min() <= 0:
        return fallback
    try:
        fit = ExponentialSmoothing(y.values, trend="add", damped_trend=True, seasonal="mul",
                                   seasonal_periods=12, initialization_method="estimated").fit()
        out = np.asarray(fit.forecast(horizon), dtype=float)
        if not np.all(np.isfinite(out)) or out.min() <= 0:
            return fallback
        return out
    except Exception:
        return fallback


# ------------------------------------------------------------- supervised dataset
def build_dataset(wide: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """One row per (forecast origin, market) with features known at the origin.

    Rows are returned even if the target lies beyond the data (``y_actual`` is NaN):
    these are the rows used for genuine out-of-sample prediction.
    """
    dates = wide.index
    arr = wide.values.astype(float)                       # (T, M)
    T, M = arr.shape
    nat = arr.sum(axis=1)
    covid = np.asarray((dates >= pd.Timestamp(config.COVID_START)) &
                       (dates <= pd.Timestamp(config.COVID_END)))
    rows = []
    for i in range(MIN_ORIGIN_POS, T):
        tgt_pos = i + horizon
        window = arr[i - config.N_LAGS + 1: i + 1][::-1]  # r0 = y[t], r12 = y[t-12]
        clean = not covid[i - MIN_ORIGIN_POS: i + 1].any()
        if tgt_pos < T:
            clean = clean and not covid[tgt_pos]
        if not clean:
            continue
        level = arr[i - 11: i + 1].mean(axis=0)
        ok = (window.min(axis=0) > 0) & (level > 0)
        if tgt_pos < T:
            ok = ok & (arr[tgt_pos] > 0)
        if not ok.any():
            continue
        yoy3 = arr[i - 2: i + 1].sum(axis=0) / np.where(arr[i - 14: i - 11].sum(axis=0) > 0,
                                                         arr[i - 14: i - 11].sum(axis=0), np.nan)
        nat_yoy3 = nat[i - 2: i + 1].sum() / nat[i - 14: i - 11].sum()
        target_date = dates[i] + pd.DateOffset(months=horizon)
        feat = window / level                             # (13, M) ratios
        d = {f"r{k}": feat[k] for k in range(config.N_LAGS)}
        d.update(yoy3=yoy3, nat_yoy3=np.full(M, nat_yoy3), target_month=np.full(M, target_date.month),
                 log_level=np.log(level), share_nat=level / nat[i - 11: i + 1].mean(),
                 post_covid=np.full(M, float(dates[i] > pd.Timestamp(config.COVID_END))))
        df = pd.DataFrame(d)
        df["origin"] = dates[i]
        df["market"] = wide.columns
        df["target_date"] = target_date
        df["level"] = level
        df["y_actual"] = arr[tgt_pos] if tgt_pos < T else np.nan
        df["y_ratio"] = df["y_actual"] / level
        rows.append(df[ok])
    out = pd.concat(rows, ignore_index=True)
    out["horizon"] = horizon
    return out


def fit_xgb(train: pd.DataFrame) -> XGBRegressor:
    m = XGBRegressor(**config.XGB_PARAMS)
    m.fit(train[FEATURE_COLS], train["y_ratio"])
    return m


def predict_xgb(model: XGBRegressor, rows: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(rows[FEATURE_COLS]), 0.05, None) * rows["level"].values
