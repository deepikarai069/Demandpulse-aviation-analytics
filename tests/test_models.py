import numpy as np
import pandas as pd
import pytest

from demandpulse import config, models


def _origin_rows(ds, origin):
    return ds[ds["origin"] == pd.Timestamp(origin)].set_index("market")


def test_features_use_only_information_up_to_the_origin(small_wide):
    """Changing every value AFTER the origin must not change that origin's features."""
    h, origin = 3, pd.Timestamp("2025-01-01")
    base = models.build_dataset(small_wide, h)
    tampered = small_wide.copy()
    tampered.loc[tampered.index > origin] *= 5.0
    alt = models.build_dataset(tampered, h)
    a, b = _origin_rows(base, origin), _origin_rows(alt, origin)
    cols = models.FEATURE_COLS + ["level"]
    pd.testing.assert_frame_equal(a[cols].sort_index(), b[cols].sort_index())


def test_target_is_exactly_h_months_after_origin(small_wide):
    for h in (1, 6, 12):
        ds = models.build_dataset(small_wide, h).dropna(subset=["y_actual"])
        assert ((ds["target_date"] - ds["origin"]).dt.days.between(28 * h, 31 * h + 3)).all()
        row = ds.iloc[len(ds) // 2]
        assert row["y_actual"] == pytest.approx(small_wide.loc[row["target_date"], row["market"]])


def test_no_training_sample_touches_the_covid_window(small_wide):
    ds = models.build_dataset(small_wide, 3)
    covid = (pd.Timestamp(config.COVID_START), pd.Timestamp(config.COVID_END))
    span_start = ds["origin"] - pd.DateOffset(months=models.MIN_ORIGIN_POS)
    overlaps = (span_start <= covid[1]) & (ds["target_date"] >= covid[0]) & (ds["origin"] <= covid[1]) | \
               ((ds["origin"] >= covid[0]) & (ds["origin"] <= covid[1]))
    assert not overlaps.any()


def test_future_rows_exist_for_the_last_origin(small_wide):
    ds = models.build_dataset(small_wide, 6)
    last = ds[ds["origin"] == small_wide.index[-1]]
    assert len(last) == small_wide.shape[1] and last["y_actual"].isna().all()


def test_seasonal_naive_and_drift():
    y = np.array([100.0] * 12 + [110.0] * 12 + [121.0] * 12)     # 36 months, +10 % per year
    # target = month 37 (index 36); 12 months before it is index 24 -> 121.0
    assert models.snaive(y, 1) == 121.0
    assert models.snaive_drift(y, 1) == pytest.approx(121.0 * 1.1)


def test_ets_falls_back_when_history_is_short():
    s = pd.Series(np.arange(1, 25, dtype=float), index=pd.date_range("2023-01-01", periods=24, freq="MS"))
    f = models.ets_forecast(s, 6)
    assert len(f) == 6 and np.all(np.isfinite(f))


def test_ets_forecast_is_positive_on_real_series(small_wide):
    f = models.ets_forecast(small_wide.iloc[:, 0], 12)
    assert len(f) == 12 and (f > 0).all()


def test_xgb_predictions_are_positive(small_wide):
    ds = models.build_dataset(small_wide, 3)
    train = ds[ds["y_actual"].notna()]
    m = models.fit_xgb(train)
    p = models.predict_xgb(m, ds[ds["y_actual"].isna()])
    assert (p > 0).all()
