"""Esperimento 01: time-series momentum contro buy-and-hold.

Domanda: una regola di trend following a 12 mesi (Moskowitz, Ooi & Pedersen,
2012) applicata a ETF liquidi migliora il rapporto rischio/rendimento rispetto
al semplice buy-and-hold, al netto dei costi?

Regole dell'esperimento:
- solo dati fino al 2022-12-31: il periodo dal 2023 è l'holdout finale
  (il runner rifiuta qualsiasi tentativo di usarlo);
- parametri fissati in anticipo dalla letteratura (lookback 12 mesi,
  ribilanciamento mensile), nessuna ottimizzazione;
- stessa finestra di valutazione per strategia e benchmark;
- ogni configurazione del momentum viene registrata nella famiglia "tsmom_etf".

Esecuzione:
    python examples/01_momentum_vs_buy_and_hold.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment

FAMILY = "tsmom_etf"
HYPOTHESIS = (
    "Il rendimento total-return degli ultimi 12 mesi di un ETF liquido predice "
    "positivamente il suo rendimento del mese successivo (Moskowitz, Ooi & Pedersen, 2012)."
)


def compare(title: str, universe: tuple[str, ...], registry: ExperimentRegistry) -> None:
    # La strategia viene registrata: è un tentativo di verificare l'ipotesi.
    mom = run_experiment(
        ExperimentSpec(family=FAMILY, strategy="time_series_momentum",
                       params={"lookback": 252}, universe=universe, hypothesis=HYPOTHESIS),
        registry,
    )
    start = str(mom.eval_start.date())
    # I benchmark non sono ipotesi: non vengono registrati né contati come tentativi.
    bench = {
        "Buy & hold SPY": run_experiment(ExperimentSpec(
            family="benchmark", strategy="buy_and_hold", universe=("SPY",), eval_start=start)),
    }
    if len(universe) > 1:
        bench["Pesi uguali (ribil. mensile)"] = run_experiment(ExperimentSpec(
            family="benchmark", strategy="buy_and_hold", params={"rebalance": "M"},
            universe=universe, eval_start=start))

    rows = {name: o.metrics for name, o in bench.items()} | {"TSMOM 12m": mom.metrics}
    df = pd.DataFrame(rows).T
    print(f"\n{title}")
    print(f"Finestra: {mom.eval_start.date()} → {mom.eval_end.date()} · run {mom.run_id} · "
          f"famiglia {FAMILY}: {mom.trials_in_family} configurazioni provate finora\n")
    print(pd.DataFrame({
        "CAGR": df["cagr"].map("{:.1%}".format),
        "Vol": df["volatility"].map("{:.1%}".format),
        "Sharpe ± SE": [f"{s:.2f} ± {e:.2f}" for s, e in zip(df["sharpe"], df["sharpe_se"])],
        "Max DD": df["max_drawdown"].map("{:.1%}".format),
        "Turnover/anno": df["turnover_yr"].map("{:.2f}".format),
        "Costi/anno": df["cost_yr"].map("{:.2%}".format),
    }).to_string())


def main() -> None:
    registry = ExperimentRegistry()
    compare("A) SPY: momentum 12 mesi contro buy-and-hold", ("SPY",), registry)
    compare("B) SPY+QQQ+TLT+GLD: momentum contro pesi uguali", ("SPY", "QQQ", "TLT", "GLD"), registry)
    print(
        "\nNota: risultati storici in-sample, costi ipotizzati (5 bp per lato), "
        "liquidità a rendimento zero. Non sono una previsione di rendimenti futuri."
        "\nRegistro: python -m quantlab.experiments"
    )


if __name__ == "__main__":
    main()
