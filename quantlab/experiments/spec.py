"""Descrizione dichiarativa di un esperimento.

Uno `ExperimentSpec` contiene tutto ciò che serve per rieseguire un backtest:
strategia e parametri, universo, finestra di valutazione, costi, ritardo.

`family` raggruppa le varianti della stessa ipotesi (es. momentum su ETF con
lookback diversi). È l'unità su cui contiamo i tentativi: se in una famiglia
provi 20 varianti e tieni la migliore, il suo Sharpe va giudicato sapendo che
è il massimo di 20 prove, non una prova singola.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from quantlab.config import DEFAULT_COST_BPS, RESEARCH_END


@dataclass(frozen=True)
class ExperimentSpec:
    family: str
    strategy: str
    params: dict = field(default_factory=dict)
    universe: tuple[str, ...] = ("SPY",)
    eval_start: str | None = None      # None = subito dopo il warm-up
    eval_end: str = RESEARCH_END
    cost_bps: float = DEFAULT_COST_BPS
    execution_lag: int = 1
    hypothesis: str = ""
    final_evaluation: bool = False     # True solo per la valutazione sull'holdout
    engine: dict = field(default_factory=dict)  # solo swing: max_positions, sizing, stop a tempo…

    def __post_init__(self):
        if not self.family.strip():
            raise ValueError("ogni esperimento deve appartenere a una famiglia")
        object.__setattr__(self, "universe", tuple(t.upper() for t in self.universe))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["universe"] = list(self.universe)
        return d

    def config_hash(self) -> str:
        """Impronta della configurazione: due spec con lo stesso hash sono lo stesso trial.

        Normalizza ciò che non cambia il risultato: testo dell'ipotesi escluso,
        parametri completati con i default, ordine dei ticker ignorato.
        """
        from quantlab.strategies.catalog import CATALOG, SWING_CATALOG, full_params

        d = self.to_dict()
        d.pop("hypothesis")
        if self.strategy in CATALOG:
            d["params"] = full_params(self.strategy, self.params)
        if self.strategy in SWING_CATALOG:
            from dataclasses import asdict as _asdict

            from quantlab.swing import SwingConfig
            d["engine"] = _asdict(SwingConfig(**self.engine)) | {"cost_bps": None}
        elif not self.engine:
            d.pop("engine")  # i run di portafoglio registrati prima del campo restano identici
        d["universe"] = sorted(d["universe"])
        d["cost_bps"] = float(d["cost_bps"])
        payload = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]
