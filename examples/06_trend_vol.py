"""Esperimento 06: trend su più orizzonti e volatility targeting.

Ipotesi, candidati, criteri e regola per il paper trading sono pre-registrati in
docs/preregistrazioni/trend_vol_etf.md (salvato su Git prima di questo test).

Esecuzione:
    python examples/06_trend_vol.py
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quantlab.backtest import excess
from quantlab.config import DEFAULT_CASH
from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.validation import Criteria, validate_candidate
from quantlab.validation.stats import sharpe_difference_test

FAMILY = "trend_vol_etf"
UNIVERSE = ("SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "GLD", "DBC", "VNQ")
HYPOTHESIS = ("Trend su più orizzonti e volatility targeting migliorano il controllo del rischio "
              "del time-series momentum 12 mesi a parità di rendimento.")
MULTI = (21, 63, 126, 252)
TARGETS = (0.06, 0.08, 0.10)

CANDIDATES = {
    "Trend 12m + vol target 8%": (
        {"horizons": (252,), "vol_target": 0.08, "vol_lookback": 63},
        lambda p: [p | {"vol_target": t} for t in TARGETS]),
    "Trend multi-orizzonte": (
        {"horizons": MULTI},
        lambda p: [p | {"horizons": h} for h in (MULTI, (63, 126, 252), (126, 252))]),
    "Multi-orizzonte + vol target 8%": (
        {"horizons": MULTI, "vol_target": 0.08, "vol_lookback": 63},
        lambda p: [p | {"vol_target": t} for t in TARGETS]),
}
BENCHMARK = ExperimentSpec(cash=DEFAULT_CASH, family="benchmark", strategy="buy_and_hold",
                           params={"rebalance": "M"}, universe=UNIVERSE)
BASELINE = ExperimentSpec(cash=DEFAULT_CASH, family="tsmom_etf", strategy="time_series_momentum",
                          params={"lookback": 252, "long_only": True}, universe=UNIVERSE)


def main() -> None:
    registry = ExperimentRegistry()
    # Tutti i candidati vengono registrati prima di guardare i risultati: il Deflated Sharpe
    # di ognuno conta fin da subito tutti e tre i tentativi.
    specs = {name: ExperimentSpec(cash=DEFAULT_CASH, family=FAMILY, strategy="trend_signals",
                                  params=params, universe=UNIVERSE, hypothesis=HYPOTHESIS)
             for name, (params, _) in CANDIDATES.items()}
    for spec in specs.values():
        run_experiment(spec, registry)

    rows, details = [], {}
    for name, (params, grid) in CANDIDATES.items():
        spec = specs[name]
        report = validate_candidate(spec, BENCHMARK, registry, grid(params), Criteria())
        m, t = report.details["metrics"], report.table()
        start, end = m["start"], m["end"]
        cand = run_experiment(spec)
        base = run_experiment(replace(BASELINE, eval_start=str(start.date())))
        vs_base = sharpe_difference_test(excess(cand.result, start, end), excess(base.result, start, end))
        rows.append({
            "strategia": name, "CAGR": m["cagr"], "Vol": m["volatility"], "Sharpe": m["sharpe"],
            "Max DD": m["max_drawdown"], "Turnover/anno": m["turnover_yr"],
            "p vs 1/N": t.loc["Sharpe > benchmark", "value"],
            "p vs tsmom 12m": vs_base.p_value, "DSR": t.loc["Deflated Sharpe", "value"],
            "criteri": f"{int(t['passed'].sum())}/{len(t)}", "verdetto": report.verdict,
        })
        details[name] = t
    bm, bb = report.details["benchmark"], base.metrics
    for label, x in (("Momentum 12m (tsmom_9etf)", bb), ("Pesi uguali 1/N (benchmark)", bm)):
        rows.append({"strategia": label, "CAGR": x["cagr"], "Vol": x["volatility"],
                     "Sharpe": x["sharpe"], "Max DD": x["max_drawdown"],
                     "Turnover/anno": x["turnover_yr"], "p vs 1/N": None, "p vs tsmom 12m": None,
                     "DSR": None, "criteri": "—", "verdetto": "—"})

    df = pd.DataFrame(rows).set_index("strategia")
    fmt = {"CAGR": "{:.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "Max DD": "{:.1%}",
           "Turnover/anno": "{:.2f}", "p vs 1/N": "{:.3f}", "p vs tsmom 12m": "{:.3f}",
           "DSR": "{:.3f}"}
    out = df.copy()
    for c, f in fmt.items():
        out[c] = df[c].map(lambda v, f=f: "—" if v is None or pd.isna(v) else f.format(v))
    pd.set_option("display.width", 220)
    print(f"\nFinestra: {start.date()} → {end.date()} · {len(UNIVERSE)} ETF · "
          f"famiglia {FAMILY}: {registry.count_trials(FAMILY)} configurazioni registrate\n")
    print(out.to_string())
    print("\nCriteri falliti:")
    for name, t in details.items():
        failed = list(t.index[~t["passed"]])
        print(f"- {name}: {', '.join(failed) if failed else 'nessuno'}")
    print("\nRisultati storici in-sample (dati fino al 2022), costi ipotizzati, liquidità al T-bill.")


if __name__ == "__main__":
    main()
