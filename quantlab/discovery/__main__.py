"""Avvia una sessione di Strategy Discovery.

Esempi:
    python -m quantlab.discovery                          # 60 candidate, 16 ETF
    python -m quantlab.discovery --budget 120 --top 8 --name etf_v2 --seed 11

Attenzione: ogni sessione aggiunge tentativi al registro. Una nuova sessione
con un nuovo seme non è "un'altra possibilità": è altra ricerca, e va
dichiarata come tale (nome diverso).
"""

from __future__ import annotations

import argparse

import pandas as pd

from quantlab.discovery import DiscoveryConfig, full_space, run_discovery
from quantlab.experiments import ExperimentRegistry

ETF_UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD",
                "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU")


def main() -> None:
    parser = argparse.ArgumentParser(description="Strategy Discovery su ETF liquidi.")
    parser.add_argument("--name", default="etf_v1", help="nome della sessione (famiglia nel registro)")
    parser.add_argument("--budget", type=int, default=60, help="candidate da testare")
    parser.add_argument("--top", type=int, default=5, help="candidate da confermare")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = DiscoveryConfig(name=args.name, universe=ETF_UNIVERSE, budget=args.budget,
                          top_k=args.top, seed=args.seed)
    print(f"Spazio di ricerca: {len(full_space())} combinazioni · ne provo {cfg.budget} "
          f"(seme {cfg.seed}) · training fino al {cfg.train_end} · conferma "
          f"{cfg.confirm_start} → {cfg.confirm_end}\n")
    res = run_discovery(cfg, ExperimentRegistry(), progress=print)

    pd.set_option("display.width", 220)
    print("\nCONFERMA (dati 2017-2022, mai usati per scegliere):")
    c = res.confirmed
    print(pd.DataFrame({
        "strategia": c["description"],
        "Sharpe training": c["train_sharpe"].map("{:.2f}".format),
        "Sharpe conferma": c["confirm_sharpe"].map("{:.2f}".format),
        "Max DD conferma": c["confirm_max_drawdown"].map("{:.1%}".format),
        "operazioni": c["confirm_trades"],
        "criteri": [f"{p}/{t}" for p, t in zip(c["checks_passed"], c["checks_total"])],
        "verdetto": c["verdict"],
    }).to_string(index=False))
    promoted = (c["verdict"] == "PROMOSSA").sum()
    print(f"\n{len(res.search)} candidate → {len(c)} confermate → {promoted} promosse.")
    print("Dettagli nella dashboard: pagina Strategy Discovery.")


if __name__ == "__main__":
    main()
