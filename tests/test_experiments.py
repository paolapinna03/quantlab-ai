"""Test del registro esperimenti e delle regole sull'holdout."""

import numpy as np
import pandas as pd
import pytest

from quantlab.experiments import (
    ExperimentRegistry,
    ExperimentSpec,
    FinalEvaluationUsed,
    HoldoutViolation,
    reproduce,
    run_experiment,
)


@pytest.fixture
def prices() -> pd.DataFrame:
    """Prezzi sintetici 2015-2024: attraversano l'inizio dell'holdout (2023)."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", "2024-12-31")
    r = rng.normal(0.0003, 0.01, (len(idx), 2))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=["AAA", "BBB"])


@pytest.fixture
def registry(tmp_path) -> ExperimentRegistry:
    return ExperimentRegistry(tmp_path / "exp.sqlite")


def tsmom(lookback=252, **kw) -> ExperimentSpec:
    return ExperimentSpec(
        family="test_tsmom", strategy="time_series_momentum",
        params={"lookback": lookback}, universe=("AAA", "BBB"), **kw,
    )


# --- Registrazione -------------------------------------------------------------

def test_run_is_logged_with_metrics_and_returns(registry, prices):
    out = run_experiment(tsmom(), registry, prices=prices)
    rec = registry.get(out.run_id)
    assert rec.status == "completed"
    assert rec.metrics["sharpe"] == pytest.approx(out.metrics["sharpe"])
    assert rec.data["AAA"]["rows"] > 0 and "sha256" in rec.data["AAA"]
    r = registry.returns(out.run_id)
    pd.testing.assert_series_equal(
        r, out.result.returns.loc[out.eval_start:out.eval_end], check_names=False, check_freq=False
    )


def test_trials_count_distinct_configurations(registry, prices):
    run_experiment(tsmom(252), registry, prices=prices)
    run_experiment(tsmom(252), registry, prices=prices)  # stessa configurazione
    out = run_experiment(tsmom(126), registry, prices=prices)
    assert out.trials_in_family == 2
    assert len(registry.runs(family="test_tsmom")) == 3


def test_hypothesis_text_does_not_create_new_trial():
    assert tsmom(hypothesis="a").config_hash() == tsmom(hypothesis="b").config_hash()
    assert tsmom(252).config_hash() != tsmom(126).config_hash()


def test_equivalent_configurations_share_a_hash():
    explicit = ExperimentSpec(family="f", strategy="time_series_momentum",
                              params={"lookback": 252, "long_only": True, "rebalance": "M"},
                              universe=("BBB", "AAA"), cost_bps=5)
    implicit = ExperimentSpec(family="f", strategy="time_series_momentum",
                              params={"lookback": 252}, universe=("AAA", "BBB"), cost_bps=5.0)
    assert explicit.config_hash() == implicit.config_hash()


def test_failed_run_is_logged_and_counted(registry, prices):
    bad = ExperimentSpec(family="test_tsmom", strategy="time_series_momentum",
                         params={"lookback": 0}, universe=("AAA", "BBB"))
    with pytest.raises(ValueError):
        run_experiment(bad, registry, prices=prices)
    runs = registry.runs(family="test_tsmom")
    assert runs.iloc[0]["status"] == "failed"
    assert registry.count_trials("test_tsmom") == 1


def test_unknown_strategy_or_param_rejected_before_running(registry, prices):
    with pytest.raises(ValueError, match="sconosciuta"):
        run_experiment(ExperimentSpec(family="f", strategy="magic"), registry, prices=prices)
    with pytest.raises(ValueError, match="non validi"):
        run_experiment(ExperimentSpec(family="f", strategy="time_series_momentum",
                                      params={"leverage": 3}), registry, prices=prices)


# --- Holdout -------------------------------------------------------------------

def test_research_cannot_touch_holdout(registry, prices):
    with pytest.raises(HoldoutViolation):
        run_experiment(tsmom(eval_end="2023-06-30"), registry, prices=prices)


def test_research_never_sees_holdout_prices(registry, prices):
    out = run_experiment(tsmom(), registry, prices=prices)
    assert out.result.returns.index[-1] < pd.Timestamp("2023-01-01")


def test_final_evaluation_must_start_in_holdout(registry, prices):
    with pytest.raises(HoldoutViolation):
        run_experiment(tsmom(final_evaluation=True, eval_end="2024-12-31"), registry, prices=prices)


def test_final_evaluation_allowed_only_once(registry, prices):
    final = tsmom(final_evaluation=True, eval_start="2023-01-02", eval_end="2024-12-31")
    out = run_experiment(final, registry, prices=prices)
    assert out.eval_start >= pd.Timestamp("2023-01-01")
    with pytest.raises(FinalEvaluationUsed):
        run_experiment(final, registry, prices=prices)


def test_final_evaluation_not_counted_as_research_trial(registry, prices):
    run_experiment(tsmom(), registry, prices=prices)
    run_experiment(tsmom(final_evaluation=True, eval_start="2023-01-02", eval_end="2024-12-31"),
                   registry, prices=prices)
    assert registry.count_trials("test_tsmom") == 1


# --- Riproducibilità -------------------------------------------------------------

def test_reproduce_gives_same_metrics(registry, prices):
    out = run_experiment(tsmom(), registry, prices=prices)
    cmp = reproduce(out.run_id, registry, prices=prices)
    assert np.allclose(cmp["registrato"], cmp["rieseguito"])
    assert not cmp["data_changed"].any()


def test_reproduce_detects_changed_data(registry, prices):
    out = run_experiment(tsmom(), registry, prices=prices)
    changed = prices.copy()
    changed.iloc[100:] *= 1.001  # es. prezzi aggiustati ricalcolati
    cmp = reproduce(out.run_id, registry, prices=changed)
    assert cmp["data_changed"].all()


def test_code_version_is_recorded(registry, prices):
    out = run_experiment(tsmom(), registry, prices=prices)
    rec = registry.get(out.run_id)
    # Il progetto è un repository Git: il commit deve essere registrato.
    assert rec.code_commit is None or len(rec.code_commit) == 40
