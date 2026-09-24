import numpy as np
import pandas as pd
import pytest

from demandpulse import backtest, config, opportunity


def _bt(n=400, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1e4, 1e6, n)
    return pd.DataFrame({"horizon": np.repeat([1, 3], n // 2), "origin": np.tile(pd.date_range("2024-01-01", periods=10, freq="MS"), n // 10),
                         "market": [f"M{i%20}" for i in range(n)], "actual": a,
                         "snaive": a * rng.lognormal(0, .12, n), "snaive_drift": a * rng.lognormal(0, .15, n),
                         "ets": a * rng.lognormal(0, .08, n), "xgb": a * rng.lognormal(0, .05, n),
                         "ensemble": a * rng.lognormal(0, .06, n)})


def test_metrics_known_values():
    assert backtest.wape([100, 100], [90, 120]) == pytest.approx(0.15)
    assert backtest.bias([100, 100], [90, 120]) == pytest.approx(0.05)
    assert backtest.smape([100], [100]) == 0


def test_best_model_and_skill():
    s = backtest.summarise(_bt())
    assert backtest.pick_best_model(s) == "xgb"
    assert (s[s["model"] == "snaive"]["skill_vs_snaive"].abs() < 1e-12).all()


def test_interval_quantiles_and_coverage_are_calibrated():
    bt = _bt(4000)
    q = backtest.residual_quantiles(bt, "xgb")
    assert (q["q_lo"] < 0).all() and (q["q_hi"] > 0).all()
    cov = backtest.loo_origin_coverage(bt, "xgb")
    assert cov["coverage"].between(0.72, 0.88).all()


def test_interpolated_quantiles_are_monotone():
    q = pd.DataFrame({"horizon": [1, 3, 6, 12], "q_lo": [-.1, -.15, -.2, -.25], "q_hi": [.1, .15, .2, .25]})
    i = backtest.interpolate_quantiles(q, range(1, 13))
    assert len(i) == 12 and i["q_hi"].is_monotonic_increasing and i["q_lo"].is_monotonic_decreasing


@pytest.fixture
def metrics():
    rng = np.random.default_rng(1)
    n = 30
    df = pd.DataFrame({"market": [f"A{i} – B{i}" for i in range(n)]})
    for c in ["forecast_growth", "ttm_growth", "momentum", "market_size", "reliability"]:
        df[f"{c}_score"] = pd.Series(rng.random(n)).rank(pct=True)
    df["ttm_growth"], df["forecast_growth"] = rng.normal(0.03, 0.05, n), rng.normal(0.02, 0.03, n)
    df["market_size_score"] = pd.Series(rng.random(n)).rank(pct=True)
    return df


def test_score_is_bounded_ranked_and_weight_scale_invariant(metrics):
    w = config.SCORE_WEIGHTS
    a = opportunity.score_with_weights(metrics, w)
    b = opportunity.score_with_weights(metrics, {k: v * 7 for k, v in w.items()})
    assert a["opportunity_score"].between(0, 100).all()
    assert sorted(a["rank"]) == list(range(1, len(a) + 1))
    assert np.allclose(a["opportunity_score"], b["opportunity_score"])


def test_single_weight_reproduces_that_components_ranking(metrics):
    out = opportunity.score_with_weights(metrics, {**{k: 0 for k in opportunity.COMPONENTS}, "momentum": 1})
    assert out.iloc[0]["momentum_score"] == metrics["momentum_score"].max()


def test_segments_use_absolute_thresholds(metrics):
    m = metrics.copy()
    m["ttm_growth"], m["forecast_growth"] = -0.04, -0.03            # everything shrinking
    assert (opportunity.segment(m)["segment"] == "Watch – softening").all()
    m["ttm_growth"], m["forecast_growth"] = 0.08, 0.06              # everything growing fast
    seg = opportunity.segment(m)["segment"]
    assert set(seg) <= {"Growth engine", "Emerging star"}


def test_rank_stability_handles_zero_weights_and_bounds(metrics):
    w = {**{k: 0 for k in opportunity.COMPONENTS}, "market_size": 1, "momentum": 1}
    s = opportunity.rank_stability(metrics, w, n_draws=300)
    assert s["top10_prob"].between(0, 1).all()
    assert (s["rank_p10"] <= s["rank_p90"]).all()
    assert s["top10_prob"].sum() == pytest.approx(10, abs=1e-6)     # exactly 10 markets are top-10 per draw
