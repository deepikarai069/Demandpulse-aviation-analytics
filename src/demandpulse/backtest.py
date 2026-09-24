"""Rolling-origin (expanding-window) backtest with honest, multi-horizon evaluation.

For every forecast origin T the models only see data up to T:
  * XGBoost is re-trained on samples whose *target date* is <= T,
  * ETS is re-fitted on each market's history up to T.
Metrics are computed on the true out-of-sample months that follow T.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config, models

MODEL_COLS = ["snaive", "snaive_drift", "ets", "xgb", "ensemble"]
MODEL_LABELS = {"snaive": "Seasonal naive", "snaive_drift": "Seasonal naive + growth",
                "ets": "Holt-Winters (ETS)", "xgb": "XGBoost (global, direct)",
                "ensemble": "Ensemble (ETS + XGBoost)"}


def backtest_origins(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    first = pd.Timestamp(config.BACKTEST_FIRST_ORIGIN)
    last = index[-1] - pd.DateOffset(months=min(config.BACKTEST_HORIZONS))
    return [d for d in pd.date_range(first, last, freq=f"{config.BACKTEST_STEP_MONTHS}MS")]


def _ets_task(market: str, origin: pd.Timestamp, series: pd.Series):
    return market, origin, models.ets_forecast(series, config.FORECAST_HORIZON)


def run_backtest(wide: pd.DataFrame, imputed_months=(), n_jobs: int = -1) -> pd.DataFrame:
    """Return a long frame: origin, horizon, market, target_date, actual + one column per model."""
    origins = backtest_origins(wide.index)
    imputed = set(pd.to_datetime(list(imputed_months)))

    # ETS: one fit per (market, origin) gives every horizon at once
    tasks = [(m, o, wide.loc[:o, m]) for o in origins for m in wide.columns]
    ets = {(m, o): f for m, o, f in Parallel(n_jobs=n_jobs)(delayed(_ets_task)(*t) for t in tasks)}

    frames = []
    for h in config.BACKTEST_HORIZONS:
        ds = models.build_dataset(wide, h)
        for o in origins:
            if o + pd.DateOffset(months=h) > wide.index[-1]:
                continue
            train = ds[(ds["target_date"] <= o) & ds["y_actual"].notna()]
            test = ds[(ds["origin"] == o)]
            if len(train) < 200 or test.empty:
                continue
            model = models.fit_xgb(train)
            test = test.assign(xgb=models.predict_xgb(model, test))
            rec = []
            for _, r in test.iterrows():
                hist = wide.loc[:o, r["market"]].values
                rec.append((snaive := models.snaive(hist, h), models.snaive_drift(hist, h),
                            float(ets[(r["market"], o)][h - 1])))
            rec = np.array(rec)
            out = pd.DataFrame({"origin": o, "horizon": h, "market": test["market"].values,
                                "target_date": test["target_date"].values, "actual": test["y_actual"].values,
                                "snaive": rec[:, 0], "snaive_drift": rec[:, 1], "ets": rec[:, 2],
                                "xgb": test["xgb"].values})
            frames.append(out)
    bt = pd.concat(frames, ignore_index=True)
    bt["ensemble"] = 0.5 * (bt["ets"] + bt["xgb"])
    bt["target_imputed"] = bt["target_date"].isin(imputed)
    return bt[~bt["target_imputed"]].drop(columns="target_imputed").reset_index(drop=True)


# --------------------------------------------------------------------- metrics
def wape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.abs(actual - pred).sum() / actual.sum())


def bias(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float((pred - actual).sum() / actual.sum())


def smape(actual, pred) -> float:
    actual, pred = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(2 * np.abs(pred - actual) / (np.abs(actual) + np.abs(pred))))


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    """Model x horizon table with WAPE, bias, sMAPE and skill versus seasonal naive."""
    rows = []
    for h, g in bt.groupby("horizon"):
        base = wape(g["actual"], g["snaive"])
        for m in MODEL_COLS:
            w = wape(g["actual"], g[m])
            rows.append(dict(horizon=h, model=m, label=MODEL_LABELS[m], wape=w,
                             bias=bias(g["actual"], g[m]), smape=smape(g["actual"], g[m]),
                             skill_vs_snaive=1 - w / base, n=len(g)))
    return pd.DataFrame(rows)


def pick_best_model(summary: pd.DataFrame) -> str:
    """Lowest mean WAPE across the evaluated horizons."""
    return summary.groupby("model")["wape"].mean().idxmin()


# -------------------------------------------------------------------- intervals
def residual_quantiles(bt: pd.DataFrame, model: str, level: float | None = None) -> pd.DataFrame:
    """Empirical quantiles of log(actual / forecast) per horizon (pooled over markets & origins)."""
    level = level or config.INTERVAL_LEVEL
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for h, g in bt.groupby("horizon"):
        r = np.log(g["actual"] / g[model])
        rows.append(dict(horizon=h, q_lo=r.quantile(lo), q_hi=r.quantile(hi)))
    return pd.DataFrame(rows)


def interpolate_quantiles(q: pd.DataFrame, horizons) -> pd.DataFrame:
    hs = q["horizon"].values
    return pd.DataFrame({"horizon": list(horizons),
                         "q_lo": np.interp(list(horizons), hs, q["q_lo"].values),
                         "q_hi": np.interp(list(horizons), hs, q["q_hi"].values)})


def loo_origin_coverage(bt: pd.DataFrame, model: str, level: float | None = None) -> pd.DataFrame:
    """Honest interval coverage: quantiles for each origin come from *other* origins only."""
    level = level or config.INTERVAL_LEVEL
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    rows = []
    for h, g in bt.groupby("horizon"):
        r = np.log(g["actual"] / g[model])
        hit = []
        for o in g["origin"].unique():
            mask = g["origin"] == o
            qlo, qhi = r[~mask].quantile(lo), r[~mask].quantile(hi)
            hit.extend(((r[mask] >= qlo) & (r[mask] <= qhi)).tolist())
        rows.append(dict(horizon=h, coverage=float(np.mean(hit)), target=level, n=len(hit)))
    return pd.DataFrame(rows)
