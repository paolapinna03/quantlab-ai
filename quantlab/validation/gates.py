"""Criteri di promozione di una strategia, decisi PRIMA di vedere i risultati.

Una strategia è "promossa" solo se supera TUTTI i controlli. Basta un
fallimento per respingerla. Le soglie stanno in `Criteria` e non si cambiano
dopo aver visto l'esito: abbassarle per far passare una strategia è una forma
di overfitting.

Promossa non significa "profittevole in futuro": significa che l'evidenza
storica è abbastanza solida da giustificare il passo successivo
(valutazione sull'holdout, poi paper trading).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from quantlab.backtest import BacktestConfig
from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.experiments.runner import load_closes
from quantlab.metrics import sharpe_ratio
from quantlab.strategies.catalog import get_strategy
from quantlab.validation.stats import deflated_sharpe_ratio, sharpe_difference_test
from quantlab.validation.walkforward import walk_forward


@dataclass(frozen=True)
class Criteria:
    min_dsr: float = 0.95                 # Deflated Sharpe: P(Sharpe vero > soglia di selezione)
    max_p_vs_benchmark: float = 0.05      # Sharpe significativamente superiore al benchmark
    min_walk_forward_sharpe: float = 0.0  # pseudo-OOS positivo con parametri scelti sul passato
    n_subperiods: int = 3                 # Sharpe positivo in ognuno dei sotto-periodi
    cost_multiplier: float = 3.0          # sopravvive a costi triplicati
    min_stressed_sharpe: float = 0.0
    max_drawdown: float = -0.35           # vincolo di rischio dell'investitore


@dataclass
class Check:
    name: str
    value: float
    threshold: str
    passed: bool
    meaning: str


@dataclass
class ValidationReport:
    spec: ExperimentSpec
    benchmark: ExperimentSpec
    checks: list[Check]
    details: dict = field(default_factory=dict)

    @property
    def promoted(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def verdict(self) -> str:
        return "PROMOSSA" if self.promoted else "RESPINTA"

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.checks]).set_index("name")


def validate_candidate(
    spec: ExperimentSpec,
    benchmark: ExperimentSpec,
    registry: ExperimentRegistry,
    param_grid: list[dict] | None = None,
    criteria: Criteria = Criteria(),
    prices: pd.DataFrame | None = None,
) -> ValidationReport:
    """Applica tutti i controlli a una strategia candidata.

    Il candidato viene eseguito e registrato (se è una configurazione nuova conta
    come tentativo). Il benchmark e gli stress test non vengono registrati.
    """
    cand = run_experiment(spec, registry, prices=prices)
    start = str(cand.eval_start.date())
    bench = run_experiment(replace(benchmark, eval_start=start, eval_end=spec.eval_end),
                           prices=prices)
    r = cand.result.returns.loc[cand.eval_start:cand.eval_end]
    rb = bench.result.returns.loc[cand.eval_start:cand.eval_end]
    checks: list[Check] = []
    details: dict = {"run_id": cand.run_id, "metrics": cand.metrics, "benchmark": bench.metrics}

    # 1. Deflated Sharpe: corregge per numero di tentativi e non-normalità.
    n = registry.count_trials(spec.family)
    ds = deflated_sharpe_ratio(r, n, registry.trial_sharpes(spec.family))
    details["dsr"] = ds
    checks.append(Check(
        "Deflated Sharpe", ds.dsr, f"≥ {criteria.min_dsr}",
        bool(ds.dsr >= criteria.min_dsr) if not np.isnan(ds.dsr) else False,
        f"P(Sharpe vero > {ds.threshold_annual:.2f}, il massimo atteso per caso su "
        f"{n} tentativi). {ds.note}".strip(),
    ))

    # 2. Meglio del benchmark, non solo meglio di zero.
    diff = sharpe_difference_test(r, rb)
    details["vs_benchmark"] = diff
    checks.append(Check(
        "Sharpe > benchmark", diff.p_value, f"p ≤ {criteria.max_p_vs_benchmark}",
        bool(diff.p_value <= criteria.max_p_vs_benchmark),
        f"Sharpe {diff.sharpe_a_annual:.2f} contro {diff.sharpe_b_annual:.2f} "
        f"(correlazione {diff.correlation:.2f}); p = probabilità di una differenza così per caso.",
    ))

    # 3. Walk-forward: parametri scelti solo sul passato.
    if param_grid:
        p = prices if prices is not None else load_closes(spec.universe, end=spec.eval_end)
        wf = walk_forward(
            p.loc[: spec.eval_end, list(spec.universe)], get_strategy(spec.strategy), param_grid,
            config=BacktestConfig(cost_bps=spec.cost_bps, execution_lag=spec.execution_lag),
        )
        wf_sharpe = sharpe_ratio(wf.returns)
        details["walk_forward"] = wf
        checks.append(Check(
            "Walk-forward", wf_sharpe, f"> {criteria.min_walk_forward_sharpe}",
            bool(wf_sharpe > criteria.min_walk_forward_sharpe),
            f"Sharpe pseudo out-of-sample su {len(wf.folds)} anni, parametri scelti "
            f"ogni anno tra {len(param_grid)} combinazioni usando solo il passato.",
        ))

    # 4. Stabilità nel tempo.
    chunks = np.array_split(np.arange(len(r)), criteria.n_subperiods)
    sub = [sharpe_ratio(r.iloc[c]) for c in chunks]
    details["subperiod_sharpes"] = sub
    checks.append(Check(
        "Stabilità", min(sub), "> 0 in ogni sotto-periodo",
        bool(min(sub) > 0),
        "Sharpe dei sotto-periodi: " + ", ".join(
            f"{r.index[c[0]].year}-{r.index[c[-1]].year}: {s:.2f}" for c, s in zip(chunks, sub)
        ),
    ))

    # 5. Costi realistici e pessimistici.
    stressed = run_experiment(
        replace(spec, cost_bps=spec.cost_bps * criteria.cost_multiplier, eval_start=start),
        prices=prices,
    )
    s_sharpe = float(stressed.metrics["sharpe"])
    checks.append(Check(
        "Costi ×{:g}".format(criteria.cost_multiplier), s_sharpe,
        f"> {criteria.min_stressed_sharpe}", bool(s_sharpe > criteria.min_stressed_sharpe),
        f"Sharpe con costi di {spec.cost_bps * criteria.cost_multiplier:g} bp per lato.",
    ))

    # 6. Vincolo di rischio.
    mdd = float(cand.metrics["max_drawdown"])
    checks.append(Check(
        "Max drawdown", mdd, f"≥ {criteria.max_drawdown:.0%}", bool(mdd >= criteria.max_drawdown),
        "Perdita massima dal picco, vincolo di rischio fissato in anticipo.",
    ))

    return ValidationReport(spec=spec, benchmark=benchmark, checks=checks, details=details)

