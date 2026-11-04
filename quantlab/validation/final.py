"""Valutazione finale sull'holdout: una sola volta per ipotesi, dopo una pre-registrazione.

I dati dal 2023 sono rimasti fuori da tutta la ricerca. Guardarli una volta
dice se la strategia regge su anni mai usati per sceglierla; guardarli due
volte, dopo aver visto il primo esito, li trasforma in dati di ricerca.

Garanzie fatte rispettare dal codice:
1. serve un documento di pre-registrazione esistente e già salvato su Git
   (nessuna modifica in sospeso), così criteri e soglie sono datati;
2. il risultato viene scritto in un file versionato (state/final_evaluations.json):
   una seconda valutazione della stessa famiglia viene rifiutata ovunque,
   non solo nel registro locale;
3. il run viene registrato come `final_evaluation` nel registro.

Criteri (pensati per un campione corto, circa 4 anni):
- Sharpe in eccesso sul T-bill > 0: la strategia rende più della liquidità per unità di rischio;
- max drawdown non peggiore del limite del risk manager del paper trading;
- nessun crollo rispetto al backtest: Sharpe dell'holdout ≥ Sharpe in-sample − 2 errori standard.
Il confronto con 1/N e SPY è riportato ma non è un criterio: in 4 anni il test
non avrebbe potenza sufficiente.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.backtest import excess
from quantlab.config import HOLDOUT_START, RESEARCH_END
from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.experiments.runner import git_version
from quantlab.validation.gates import Check
from quantlab.validation.stats import sharpe_difference_test

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_RECORD = _PROJECT_ROOT / "state" / "final_evaluations.json"


class FinalEvaluationRefused(RuntimeError):
    """La valutazione finale non può partire (pre-registrazione mancante o già eseguita)."""


@dataclass(frozen=True)
class FinalCriteria:
    min_excess_sharpe: float = 0.0
    max_drawdown: float = -0.20          # stesso limite del risk manager del paper trading
    max_degradation_se: float = 2.0      # Sharpe holdout ≥ Sharpe in-sample − 2 SE


@dataclass
class FinalReport:
    family: str
    verdict: str                         # "CONFERMATA" | "NON CONFERMATA"
    checks: list[Check]
    in_sample: dict
    holdout: dict
    benchmarks: dict
    window: tuple[str, str]
    run_id: str | None
    prereg: str
    extra: dict = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return self.verdict == "CONFERMATA"


def load_record(path: Path | str | None = None) -> dict:
    p = Path(path) if path else FINAL_RECORD
    return json.loads(p.read_text()) if p.exists() else {}


def _check_prereg(prereg: Path, require_git: bool) -> None:
    if not prereg.exists():
        raise FinalEvaluationRefused(f"pre-registrazione mancante: {prereg}")
    if not require_git:
        return
    root = _PROJECT_ROOT
    rel = prereg.resolve().relative_to(root)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(rel)], cwd=root,
                             capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(rel)], cwd=root).returncode == 0
    if not (tracked and clean):
        raise FinalEvaluationRefused(
            "la pre-registrazione deve essere salvata su Git (commit) prima della valutazione finale, "
            "senza modifiche in sospeso: così criteri e soglie restano datati."
        )


def run_final_evaluation(
    spec: ExperimentSpec,
    benchmarks: dict[str, ExperimentSpec],
    registry: ExperimentRegistry,
    prereg: Path | str,
    eval_end: str | None = None,
    criteria: FinalCriteria = FinalCriteria(),
    record_path: Path | str | None = None,
    require_git: bool = True,
    prices: pd.DataFrame | None = None,
    tbill_yield: pd.Series | None = None,
) -> FinalReport:
    """Esegue la valutazione finale di `spec` (una sola volta per famiglia)."""
    prereg = Path(prereg)
    record_path = Path(record_path) if record_path else FINAL_RECORD
    record = load_record(record_path)
    if spec.family in record:
        raise FinalEvaluationRefused(
            f"la famiglia {spec.family!r} ha già una valutazione finale "
            f"({record[spec.family]['date']}): non si ripete."
        )
    _check_prereg(prereg, require_git)
    registry.check_final_allowed(spec.family)

    end = eval_end or str(pd.Timestamp.now().date())
    inject = dict(prices=prices, tbill_yield=tbill_yield)

    # In-sample: la stessa configurazione sul periodo di ricerca (non è un nuovo tentativo).
    ins = run_experiment(replace(spec, final_evaluation=False, eval_start=None, eval_end=RESEARCH_END),
                         **inject)
    # Holdout: l'unica esecuzione registrata come valutazione finale.
    hold_spec = replace(spec, final_evaluation=True, eval_start=HOLDOUT_START, eval_end=end)
    hold = run_experiment(hold_spec, registry, **inject)
    start, stop = hold.eval_start, hold.eval_end
    r = excess(hold.result, start, stop)

    bench_out, bench_tests = {}, {}
    for name, b in benchmarks.items():
        bo = run_experiment(replace(b, cash=spec.cash, eval_start=str(start.date()),
                                    eval_end=end, final_evaluation=True), **inject)
        bench_out[name] = _metrics(bo.metrics)
        d = sharpe_difference_test(r, excess(bo.result, start, stop))
        bench_tests[name] = {"p_value": d.p_value, "correlation": d.correlation}

    hm, im = hold.metrics, ins.metrics
    floor = float(im["sharpe"]) - criteria.max_degradation_se * float(hm["sharpe_se"])
    checks = [
        Check("Sharpe sopra il T-bill", float(hm["sharpe"]), f"> {criteria.min_excess_sharpe:g}",
              bool(hm["sharpe"] > criteria.min_excess_sharpe),
              "Rendimento in eccesso sulla liquidità per unità di rischio, sugli anni mai usati."),
        Check("Max drawdown", float(hm["max_drawdown"]), f"≥ {criteria.max_drawdown:.0%}",
              bool(hm["max_drawdown"] >= criteria.max_drawdown),
              "Stesso limite che fa scattare l'arresto automatico nel paper trading."),
        Check("Nessun crollo rispetto al backtest", float(hm["sharpe"]), f"≥ {floor:.2f}",
              bool(hm["sharpe"] >= floor),
              f"Sharpe in-sample {float(im['sharpe']):.2f} meno 2 errori standard dell'holdout "
              f"({float(hm['sharpe_se']):.2f})."),
    ]
    verdict = "CONFERMATA" if all(c.passed for c in checks) else "NON CONFERMATA"
    report = FinalReport(
        family=spec.family, verdict=verdict, checks=checks, in_sample=_metrics(im),
        holdout=_metrics(hm), benchmarks=bench_out,
        window=(str(start.date()), str(stop.date())), run_id=hold.run_id, prereg=str(prereg.name),
        extra={"benchmark_tests": bench_tests},
    )
    registry.log_validation(hold.run_id, spec.family, verdict, [asdict(c) for c in checks])

    commit, dirty = git_version()
    record[spec.family] = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verdict": verdict, "window": list(report.window), "run_id": hold.run_id,
        "spec": spec.to_dict(), "prereg": report.prereg, "commit": commit, "dirty": dirty,
        "in_sample": report.in_sample, "holdout": report.holdout, "benchmarks": bench_out,
        "benchmark_tests": bench_tests, "checks": [asdict(c) for c in checks],
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False, default=str))
    return report


def _metrics(m: pd.Series) -> dict:
    keys = ["start", "end", "cagr", "volatility", "sharpe", "sharpe_se", "max_drawdown",
            "cash_rate", "turnover_yr", "invested"]
    return {k: (str(m[k].date()) if hasattr(m.get(k), "date") else
                (float(m[k]) if m.get(k) is not None else None)) for k in keys if k in m}
