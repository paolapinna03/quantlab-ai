"""Esperimento 02: il momentum supera i criteri di promozione?

Candidato: time-series momentum 12 mesi su SPY, QQQ, TLT, GLD (configurazione
della letteratura, la stessa dell'esperimento 01).
Benchmark: pesi uguali sugli stessi ETF, ribilanciati ogni mese. È il confronto
giusto: isola il contributo del *segnale* da quello della diversificazione.

I criteri (quantlab/validation/gates.py → Criteria) sono fissati prima di
eseguire questo script. Se la strategia viene respinta, NON si abbassano le
soglie: si torna alle ipotesi.

Esecuzione:
    python examples/02_validate_momentum.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.experiments import ExperimentRegistry, ExperimentSpec
from quantlab.validation import Criteria, validate_candidate

UNIVERSE = ("SPY", "QQQ", "TLT", "GLD")
CANDIDATE = ExperimentSpec(
    family="tsmom_etf", strategy="time_series_momentum", params={"lookback": 252},
    universe=UNIVERSE,
    hypothesis="Il rendimento total-return degli ultimi 12 mesi di un ETF liquido predice "
               "positivamente il suo rendimento del mese successivo (Moskowitz, Ooi & Pedersen, 2012).",
)
BENCHMARK = ExperimentSpec(family="benchmark", strategy="buy_and_hold",
                           params={"rebalance": "M"}, universe=UNIVERSE)
# Per il walk-forward: ogni anno si sceglie il lookback migliore usando solo il passato.
GRID = [{"lookback": lb} for lb in (63, 126, 189, 252, 378)]


def main() -> None:
    criteria = Criteria()
    report = validate_candidate(CANDIDATE, BENCHMARK, ExperimentRegistry(), GRID, criteria)

    m, b = report.details["metrics"], report.details["benchmark"]
    print(f"\nCandidato: {CANDIDATE.strategy} {CANDIDATE.params} su {', '.join(UNIVERSE)}")
    print(f"Finestra: {m['start'].date()} → {m['end'].date()} · run {report.details['run_id']}")
    print(f"Sharpe {m['sharpe']:.2f} (benchmark {b['sharpe']:.2f}) · "
          f"CAGR {m['cagr']:.1%} (benchmark {b['cagr']:.1%})\n")

    t = report.table()
    out = pd.DataFrame({
        "valore": t["value"].map(lambda v: f"{v:.3f}"),
        "soglia": t["threshold"],
        "esito": t["passed"].map({True: "✓ superato", False: "✗ fallito"}),
    })
    print(out.to_string())
    print()
    for name, row in t.iterrows():
        print(f"- {name}: {row['meaning']}")

    wf = report.details.get("walk_forward")
    if wf is not None:
        print("\nWalk-forward: lookback scelto ogni anno (solo con dati passati)")
        f = wf.folds.assign(lookback=wf.folds["params"].map(lambda p: p["lookback"]))
        print(f[["test_start", "test_end", "lookback", "train_sharpe", "test_sharpe_chosen"]]
              .to_string(index=False, float_format="{:.2f}".format))

    print(f"\nVERDETTO: {report.verdict}")
    print("Promossa non significa profittevole in futuro; respinta non significa che l'idea "
          "sia sbagliata, ma che questa evidenza non basta.")


if __name__ == "__main__":
    main()
