"""Strategy Discovery in due fasi: ricerca sul training, conferma su dati mai usati.

    2005 ─────────── training ─────────── 2016 │ 2017 ── conferma ── 2022 │ 2023+ holdout
         tutte le candidate, classificate qui    solo le top K, mai usato      bloccato
                                                 per sceglierle

Fase 1 – Ricerca: ogni candidata viene testata SOLO fino a `train_end`
(i dati successivi non vengono nemmeno caricati). Si scartano quelle con
troppe poche operazioni e si ordinano le altre per Sharpe.

Fase 2 – Conferma: le migliori K vengono rieseguite sul periodo di conferma
e sottoposte ai criteri di promozione. Il Deflated Sharpe della conferma
conta le K candidate confermate; la selezione delle K è avvenuta su dati
diversi, quindi non contamina questa fase.

Ogni candidata di entrambe le fasi viene registrata; i verdetti pure.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

from quantlab.config import DEFAULT_CASH, DEFAULT_COST_BPS, HOLDOUT_START, RESEARCH_END
from quantlab.discovery.space import full_space, sample_space
from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.strategies.rules import describe
from quantlab.validation import Criteria, validate_candidate


@dataclass(frozen=True)
class DiscoveryConfig:
    name: str
    universe: tuple[str, ...]
    budget: int = 60                 # candidate da provare nella fase di ricerca
    top_k: int = 5                   # candidate che passano alla conferma
    seed: int = 7                    # campionamento riproducibile dello spazio
    train_end: str = "2016-12-31"
    confirm_start: str = "2017-01-01"
    confirm_end: str = RESEARCH_END
    min_trades: int = 30             # sotto questa soglia lo Sharpe è troppo rumoroso
    cost_bps: float = DEFAULT_COST_BPS
    cash: str = DEFAULT_CASH         # rendimento della liquidità (e base dello Sharpe in eccesso)
    base_engine: dict = field(default_factory=lambda: {"max_positions": 5, "position_fraction": 0.2})

    def __post_init__(self):
        if pd.Timestamp(self.confirm_start) <= pd.Timestamp(self.train_end):
            raise ValueError("la conferma deve iniziare dopo la fine del training")
        if pd.Timestamp(self.confirm_end) >= pd.Timestamp(HOLDOUT_START):
            raise ValueError("la conferma non può toccare l'holdout")
        object.__setattr__(self, "universe", tuple(self.universe))

    @property
    def search_family(self) -> str:
        return f"discovery_{self.name}"

    @property
    def confirm_family(self) -> str:
        return f"discovery_{self.name}_conferma"


@dataclass
class DiscoveryResult:
    config: DiscoveryConfig
    search: pd.DataFrame        # tutte le candidate, metriche sul training
    confirmed: pd.DataFrame     # top K, metriche sulla conferma + verdetto
    reports: list = field(default_factory=list)


def run_discovery(
    config: DiscoveryConfig,
    registry: ExperimentRegistry,
    criteria: Criteria = Criteria(),
    ohlc: dict[str, pd.DataFrame] | None = None,
    progress: Callable[[str], None] | None = None,
) -> DiscoveryResult:
    say = progress or (lambda msg: None)
    space_size = len(full_space())
    candidates = sample_space(config.budget, config.seed)
    registry.log_discovery(config.name, asdict(config) | {"space_size": space_size,
                                                          "tested": len(candidates)})

    # --- Fase 1: ricerca sul solo training -------------------------------------
    rows = []
    for i, (params, engine) in enumerate(candidates, 1):
        spec = ExperimentSpec(
            family=config.search_family, strategy="rule_swing", params=params,
            universe=config.universe, eval_end=config.train_end, cost_bps=config.cost_bps, cash=config.cash,
            engine=config.base_engine | engine, hypothesis=describe(params, engine),
        )
        row = {"candidate": i, "description": describe(params, engine),
               "params": params, "engine": engine}
        try:
            out = run_experiment(spec, registry, ohlc=_until(ohlc, config.train_end))
            m = out.metrics
            row |= {"run_id": out.run_id, "sharpe": float(m["sharpe"]), "cagr": float(m["cagr"]),
                    "max_drawdown": float(m["max_drawdown"]),
                    "trades": int(m.get("trade_trades", 0) or 0), "status": "ok"}
        except Exception as exc:  # una candidata che fallisce resta registrata come tentativo
            row |= {"run_id": None, "sharpe": np.nan, "cagr": np.nan, "max_drawdown": np.nan,
                    "trades": 0, "status": f"errore: {exc}"}
        rows.append(row)
        say(f"[{i}/{len(candidates)}] {row['description']}: Sharpe {row['sharpe']:.2f}")

    search = pd.DataFrame(rows)
    eligible = search[(search["status"] == "ok") & (search["trades"] >= config.min_trades)]
    top = eligible.sort_values("sharpe", ascending=False).head(config.top_k)
    search["selected"] = search["candidate"].isin(top["candidate"])

    # --- Fase 2: conferma su dati mai usati per scegliere ------------------------
    benchmark = ExperimentSpec(family="benchmark", strategy="buy_and_hold",
                               params={"rebalance": "M"}, universe=config.universe,
                               cost_bps=config.cost_bps)
    confirm_specs = [
        ExperimentSpec(
            family=config.confirm_family, strategy="rule_swing", params=cand["params"],
            universe=config.universe, eval_start=config.confirm_start, eval_end=config.confirm_end,
            cost_bps=config.cost_bps, cash=config.cash, engine=config.base_engine | cand["engine"],
            hypothesis=cand["description"],
        )
        for _, cand in top.iterrows()
    ]
    # Prima si registrano TUTTE le K candidate, poi si valutano: così il Deflated
    # Sharpe di ognuna conta gli stessi K tentativi, non 1, 2, 3… in ordine di arrivo.
    for spec in confirm_specs:
        run_experiment(spec, registry, ohlc=_until(ohlc, config.confirm_end))

    confirmed, reports = [], []
    for spec, (_, cand) in zip(confirm_specs, top.iterrows()):
        report = validate_candidate(spec, benchmark, registry, criteria=criteria,
                                    ohlc=_until(ohlc, config.confirm_end))
        registry.log_validation(report.details["run_id"], config.confirm_family, report.verdict,
                                [asdict(c) for c in report.checks])
        m = report.details["metrics"]
        confirmed.append({
            "candidate": cand["candidate"], "description": cand["description"],
            "train_sharpe": cand["sharpe"], "confirm_sharpe": float(m["sharpe"]),
            "confirm_cagr": float(m["cagr"]), "confirm_max_drawdown": float(m["max_drawdown"]),
            "confirm_trades": int(m.get("trade_trades", 0) or 0),
            "checks_passed": sum(c.passed for c in report.checks), "checks_total": len(report.checks),
            "verdict": report.verdict, "run_id": report.details["run_id"],
        })
        reports.append(report)
        say(f"conferma: {cand['description']} → {report.verdict}")

    return DiscoveryResult(config, search, pd.DataFrame(confirmed), reports)


def _until(ohlc, end):
    if ohlc is None:
        return None
    return {t: d.loc[:end] for t, d in ohlc.items()}
