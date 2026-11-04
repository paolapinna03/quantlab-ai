"""Esperimento 05: valutazione finale sull'holdout (UNA SOLA VOLTA).

Pre-registrazione: docs/preregistrazioni/finale_holdout_2026.md (salvata su Git prima).
Il codice rifiuta una seconda esecuzione per la stessa famiglia.

Esecuzione:
    python examples/05_final_evaluation.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantlab.data import load_prices
from quantlab.experiments import ExperimentRegistry, ExperimentSpec
from quantlab.paper.presets import PRESETS
from quantlab.validation import FinalEvaluationRefused, run_final_evaluation

PREREG = Path(__file__).resolve().parents[1] / "docs" / "preregistrazioni" / "finale_holdout_2026.md"
CANDIDATES = {"allocazione_etf": "rp_momentum", "tsmom_etf": "tsmom_9etf"}


def main() -> None:
    registry = ExperimentRegistry()
    last_close = str(load_prices("SPY").index[-1].date())
    for family, preset in CANDIDATES.items():
        s = PRESETS[preset]["spec"]
        spec = ExperimentSpec(family=family, strategy=s["strategy"], params=s["params"],
                              universe=tuple(s["universe"]), cost_bps=s["cost_bps"], cash=s["cash"])
        benchmarks = {
            "Pesi uguali 1/N": ExperimentSpec(family="benchmark", strategy="buy_and_hold",
                                              params={"rebalance": "M"}, universe=spec.universe),
            "SPY buy & hold": ExperimentSpec(family="benchmark", strategy="buy_and_hold",
                                             universe=("SPY",)),
        }
        try:
            rep = run_final_evaluation(spec, benchmarks, registry, PREREG, eval_end=last_close)
        except FinalEvaluationRefused as exc:
            print(f"[{family}] {exc}")
            continue
        print(f"\n=== {family} · {preset} · holdout {rep.window[0]} → {rep.window[1]} ===")
        rows = {"In-sample (ricerca)": rep.in_sample, "Holdout": rep.holdout} | rep.benchmarks
        df = pd.DataFrame(rows).T
        print(pd.DataFrame({
            "CAGR": df["cagr"].map("{:.1%}".format), "Vol": df["volatility"].map("{:.1%}".format),
            "Sharpe": df["sharpe"].map("{:.2f}".format), "Max DD": df["max_drawdown"].map("{:.1%}".format),
        }).to_string())
        for c in rep.checks:
            print(f"  {'✓' if c.passed else '✗'} {c.name}: {c.value:.3f} (soglia {c.threshold})")
        for name, t in rep.extra["benchmark_tests"].items():
            print(f"  · contro {name}: p = {t['p_value']:.2f} (informativo)")
        print(f"VERDETTO: {rep.verdict}")


if __name__ == "__main__":
    main()
