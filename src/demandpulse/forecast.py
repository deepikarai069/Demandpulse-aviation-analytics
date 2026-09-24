"""Produce the genuine out-of-sample 12-month forecast, with backtest-calibrated intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import backtest, config, models


def make_forecast(wide: pd.DataFrame, bt: pd.DataFrame, model: str) -> pd.DataFrame:
    """Forecast every market for the next FORECAST_HORIZON months after the last observed month."""
    origin = wide.index[-1]
    H = config.FORECAST_HORIZON
    horizons = list(range(1, H + 1))

    # --- component forecasts -------------------------------------------------
    xgb_rows = []
    for h in horizons:
        ds = models.build_dataset(wide, h)
        train = ds[ds["y_actual"].notna()]
        test = ds[ds["origin"] == origin]
        m = models.fit_xgb(train)
        xgb_rows.append(test.assign(xgb=models.predict_xgb(m, test))[["market", "horizon", "target_date", "xgb"]])
    xgb = pd.concat(xgb_rows, ignore_index=True)

    ets_rows = []
    for market in xgb["market"].unique():
        f = models.ets_forecast(wide[market], H)
        hist = wide[market].values
        for h in horizons:
            ets_rows.append(dict(market=market, horizon=h, ets=f[h - 1],
                                 snaive=models.snaive(hist, h),
                                 snaive_drift=models.snaive_drift(hist, h)))
    fc = xgb.merge(pd.DataFrame(ets_rows), on=["market", "horizon"])
    fc["ensemble"] = 0.5 * (fc["ets"] + fc["xgb"])
    fc["forecast"] = fc[model]

    # --- intervals from backtest residuals ------------------------------------
    q = backtest.interpolate_quantiles(backtest.residual_quantiles(bt, model), horizons)
    fc = fc.merge(q, on="horizon")
    fc["lower"] = fc["forecast"] * np.exp(fc["q_lo"])
    fc["upper"] = fc["forecast"] * np.exp(fc["q_hi"])
    fc["origin"] = origin
    fc["model"] = model
    a_b = wide.columns.to_series().str.split(" – ", expand=True)
    fc["a"] = fc["market"].map(a_b[0])
    fc["b"] = fc["market"].map(a_b[1])
    cols = ["market", "a", "b", "origin", "target_date", "horizon", "forecast", "lower", "upper",
            "model", "xgb", "ets", "snaive", "ensemble"]
    return fc[cols].sort_values(["market", "horizon"]).reset_index(drop=True)
