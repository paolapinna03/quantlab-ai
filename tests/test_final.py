"""Test della valutazione finale sull'holdout: le garanzie devono reggere."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantlab.experiments import ExperimentRegistry, ExperimentSpec
from quantlab.validation import FinalEvaluationRefused, load_record, run_final_evaluation
from quantlab.validation.final import _check_prereg

IDX = pd.bdate_range("2012-01-02", "2025-06-30")
YIELD = pd.Series(2.0, index=pd.bdate_range("2011-01-01", "2025-12-31"))


@pytest.fixture
def prices():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0003, [0.006, 0.012, 0.009], (len(IDX), 3))
    return pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=IDX, columns=["AAA", "BBB", "CCC"])


SPEC = ExperimentSpec(family="fam", strategy="risk_allocation", universe=("AAA", "BBB", "CCC"),
                      params={"method": "risk_parity", "lookback": 126, "momentum_filter": 126},
                      cash="tbill")
BENCH = {"Pesi uguali": ExperimentSpec(family="benchmark", strategy="buy_and_hold",
                                        params={"rebalance": "M"}, universe=("AAA", "BBB", "CCC"))}


def run(tmp_path, prices, registry=None, record=None, prereg=True):
    pre = tmp_path / "prereg.md"
    if prereg:
        pre.write_text("criteri")
    return run_final_evaluation(SPEC, BENCH, registry or ExperimentRegistry(tmp_path / "r.sqlite"),
                                pre, eval_end="2025-06-30", record_path=record or tmp_path / "final.json",
                                require_git=False, prices=prices, tbill_yield=YIELD)


def test_refused_without_preregistration(tmp_path, prices):
    with pytest.raises(FinalEvaluationRefused, match="pre-registrazione mancante"):
        run(tmp_path, prices, prereg=False)


def test_runs_once_and_records(tmp_path, prices):
    reg = ExperimentRegistry(tmp_path / "r.sqlite")
    rep = run(tmp_path, prices, registry=reg)
    assert rep.verdict in ("CONFERMATA", "NON CONFERMATA") and len(rep.checks) == 3
    assert rep.window[0] >= "2023-01-01"                       # holdout soltanto
    assert rep.in_sample["end"] <= "2022-12-31"                 # in-sample soltanto
    rec = load_record(tmp_path / "final.json")
    assert rec["fam"]["verdict"] == rep.verdict
    final_runs = reg.runs(family="fam")
    assert final_runs["final"].sum() == 1
    with pytest.raises(FinalEvaluationRefused, match="già una valutazione finale"):
        run(tmp_path, prices, registry=reg)                    # stesso file di esito


def test_refused_by_registry_even_with_new_record_file(tmp_path, prices):
    reg = ExperimentRegistry(tmp_path / "r.sqlite")
    run(tmp_path, prices, registry=reg)
    with pytest.raises(Exception, match="valutazione finale"):
        run(tmp_path, prices, registry=reg, record=tmp_path / "altro.json")


def test_refused_by_record_even_with_new_registry(tmp_path, prices):
    run(tmp_path, prices)
    with pytest.raises(FinalEvaluationRefused):
        run(tmp_path, prices, registry=ExperimentRegistry(tmp_path / "nuovo.sqlite"))


def test_preregistration_must_be_committed():
    repo = Path(__file__).resolve().parents[1]
    tmp = repo / "docs" / "preregistrazioni" / "_test_non_salvato.md"
    tmp.write_text("bozza")
    try:
        with pytest.raises(FinalEvaluationRefused, match="salvata su Git"):
            _check_prereg(tmp, require_git=True)
    finally:
        tmp.unlink()
