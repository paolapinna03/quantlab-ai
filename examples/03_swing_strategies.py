"""Esperimento 03: due strategie swing classiche su ETF liquidi.

Candidati (regole divulgate da trader, non peer-reviewed, parametri di default
dei libri, NESSUNA ottimizzazione):
- RSI(2) mean reversion: compra le discese brevi dentro un trend rialzista;
- Donchian breakout: compra le rotture dei massimi a 55 giorni, stop a 2 ATR.

Universo: 16 ETF USA liquidi (indici, settori, obbligazioni, oro), al massimo
5 posizioni da 20% del capitale. Costi 5 bp + slippage 5 bp per lato.
Benchmark: pesi uguali sugli stessi ETF, ribilanciati ogni mese.
Ogni strategia è una famiglia di ipotesi separata nel registro.

Esecuzione:
    python examples/03_swing_strategies.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.experiments import ExperimentRegistry, ExperimentSpec
from quantlab.validation import Criteria, validate_candidate

UNIVERSE = ("SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD",
            "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU")
ENGINE = {"max_positions": 5, "position_fraction": 0.2}

CANDIDATES = {
    "RSI(2) mean reversion": ExperimentSpec(
        family="swing_rsi2_etf", strategy="rsi2_mean_reversion", universe=UNIVERSE, engine=ENGINE,
        hypothesis="Dopo forti cali di 1-2 giorni in un trend rialzista, gli ETF liquidi tendono "
                   "a rimbalzare nei giorni successivi (inversione di breve periodo).",
    ),
    "Donchian breakout 55/20": ExperimentSpec(
        family="swing_donchian_etf", strategy="donchian_breakout", universe=UNIVERSE, engine=ENGINE,
        hypothesis="Le rotture dei massimi a 55 giorni segnalano trend che tendono a proseguire "
                   "(trend following su orizzonte di settimane).",
    ),
}
BENCHMARK = ExperimentSpec(family="benchmark", strategy="buy_and_hold",
                           params={"rebalance": "M"}, universe=UNIVERSE)


def main() -> None:
    registry = ExperimentRegistry()
    for name, spec in CANDIDATES.items():
        report = validate_candidate(spec, BENCHMARK, registry, criteria=Criteria())
        m, b = report.details["metrics"], report.details["benchmark"]
        print(f"\n=== {name} ===  (run {report.details['run_id']})")
        print(f"Finestra {m['start'].date()} → {m['end'].date()}")
        print(f"Portafoglio: CAGR {m['cagr']:.1%} · Sharpe {m['sharpe']:.2f} · "
              f"Max DD {m['max_drawdown']:.1%} · esposizione media {m['invested']:.0%}")
        print(f"Benchmark pesi uguali: CAGR {b['cagr']:.1%} · Sharpe {b['sharpe']:.2f} · "
              f"Max DD {b['max_drawdown']:.1%}")
        print(f"Operazioni: {m['trade_trades']} · win rate {m['trade_win_rate']:.0%} · "
              f"guadagno medio {m['trade_avg_win']:+.2%} · perdita media {m['trade_avg_loss']:+.2%} · "
              f"profit factor {m['trade_profit_factor']:.2f} · "
              f"durata media {m['trade_avg_bars_held']:.1f} giorni")
        t = report.table()
        print(pd.DataFrame({
            "valore": t["value"].map(lambda v: f"{v:.3f}"),
            "soglia": t["threshold"],
            "esito": t["passed"].map({True: "✓", False: "✗"}),
        }).to_string())
        print(f"VERDETTO: {report.verdict}")

    print("\nRisultati storici in-sample (2005-2022), costi ipotizzati, azioni frazionarie, "
          "solo posizioni lunghe. Non sono una previsione di rendimenti futuri.")


if __name__ == "__main__":
    main()
