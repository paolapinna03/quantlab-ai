"""Riga di comando per consultare il registro.

Esempi:
    python -m quantlab.experiments                 # ultimi 20 run
    python -m quantlab.experiments --family tsmom_etf
    python -m quantlab.experiments --reproduce <id>
"""

from __future__ import annotations

import argparse

import pandas as pd

from quantlab.experiments import ExperimentRegistry, reproduce


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta il registro degli esperimenti.")
    parser.add_argument("--family", help="mostra solo questa famiglia")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--reproduce", metavar="ID", help="riesegue un run e confronta")
    args = parser.parse_args()

    reg = ExperimentRegistry()
    if args.reproduce:
        print(reproduce(args.reproduce, reg).to_string())
        return

    df = reg.runs(family=args.family, limit=args.limit)
    if df.empty:
        print("Registro vuoto.")
        return
    pd.set_option("display.width", 200)
    print(df.drop(columns=["created_at"]).to_string(index=False, float_format="{:.3f}".format))
    print()
    for fam in reg.families():
        print(f"famiglia {fam}: {reg.count_trials(fam)} configurazioni distinte provate")


if __name__ == "__main__":
    main()
