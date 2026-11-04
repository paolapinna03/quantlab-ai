"""Esperimento 04: allocazione basata sul rischio contro pesi uguali.

Ipotesi, candidati e criteri sono pre-registrati in
docs/preregistrazioni/allocazione_etf.md (salvato su Git prima di questo test).

Esecuzione:
    python examples/04_risk_allocation.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.config import DEFAULT_CASH
from quantlab.experiments import ExperimentRegistry, ExperimentSpec
from quantlab.validation import Criteria, validate_candidate

FAMILY = "allocazione_etf"
UNIVERSE = ("SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "GLD", "DBC", "VNQ")
HYPOTHESIS = ("Un'allocazione basata sul rischio migliora il rapporto rischio/rendimento "
              "rispetto ai pesi uguali, fuori campione e al netto dei costi.")
BASE = {"lookback": 252, "max_weight": 0.4, "shrinkage": True}

CANDIDATES = {
    "Inverse volatility": {"method": "inverse_vol"},
    "Minima varianza (tetto 40%)": {"method": "min_variance"},
    "Risk parity (ERC)": {"method": "risk_parity"},
    "Hierarchical Risk Parity": {"method": "hrp"},
    "Risk parity + momentum 12m": {"method": "risk_parity", "momentum_filter": 252},
    "Media-varianza (didattico)": {"method": "mean_variance"},
}
BENCHMARK = ExperimentSpec(cash=DEFAULT_CASH, family="benchmark", strategy="buy_and_hold",
                           params={"rebalance": "M"}, universe=UNIVERSE)


def main() -> None:
    registry = ExperimentRegistry()
    rows, details = [], {}
    for name, extra in CANDIDATES.items():
        params = BASE | extra
        spec = ExperimentSpec(cash=DEFAULT_CASH, family=FAMILY, strategy="risk_allocation",
                              params=params, universe=UNIVERSE, hypothesis=HYPOTHESIS)
        grid = [params | {"lookback": lb} for lb in (126, 252, 504)]
        report = validate_candidate(spec, BENCHMARK, registry, grid, Criteria())
        m, b = report.details["metrics"], report.details["benchmark"]
        t = report.table()
        rows.append({
            "strategia": name, "CAGR": m["cagr"], "Vol": m["volatility"], "Sharpe": m["sharpe"],
            "Max DD": m["max_drawdown"], "Turnover/anno": m["turnover_yr"],
            "p vs 1/N": t.loc["Sharpe > benchmark", "value"],
            "DSR": t.loc["Deflated Sharpe", "value"],
            "criteri": f"{int(t['passed'].sum())}/{len(t)}", "verdetto": report.verdict,
        })
        details[name] = t
    bench = report.details["benchmark"]
    rows.append({"strategia": "Pesi uguali 1/N (benchmark)", "CAGR": bench["cagr"],
                 "Vol": bench["volatility"], "Sharpe": bench["sharpe"],
                 "Max DD": bench["max_drawdown"], "Turnover/anno": bench["turnover_yr"],
                 "p vs 1/N": None, "DSR": None, "criteri": "—", "verdetto": "—"})

    df = pd.DataFrame(rows).set_index("strategia")
    fmt = {"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "Max DD": "{:.1%}",
           "Turnover/anno": "{:.2f}", "p vs 1/N": "{:.3f}", "DSR": "{:.3f}"}
    out = df.copy()
    for c, f in fmt.items():
        out[c] = df[c].map(lambda v, f=f: "—" if v is None or pd.isna(v) else f.format(v))
    pd.set_option("display.width", 200)
    print(f"\nFinestra: {bench['start'].date()} → {bench['end'].date()} · {len(UNIVERSE)} ETF · "
          f"famiglia {FAMILY}: {registry.count_trials(FAMILY)} configurazioni registrate\n")
    print(out.to_string())
    print("\nCriteri falliti:")
    for name, t in details.items():
        failed = list(t.index[~t["passed"]])
        print(f"- {name}: {', '.join(failed) if failed else 'nessuno'}")
    print("\nRisultati storici in-sample (dati fino al 2022), costi ipotizzati, liquidità al T-bill.")


if __name__ == "__main__":
    main()
