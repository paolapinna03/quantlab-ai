"""Esperimento 01: time-series momentum contro buy-and-hold.

Domanda: una regola di trend following a 12 mesi (Moskowitz, Ooi & Pedersen,
2012) applicata a ETF liquidi migliora il rapporto rischio/rendimento rispetto
al semplice buy-and-hold, al netto dei costi?

Regole dell'esperimento:
- solo dati fino al 2022-12-31: il periodo dal 2023 è l'holdout finale;
- parametri fissati in anticipo dalla letteratura (lookback 12 mesi,
  ribilanciamento mensile), nessuna ottimizzazione;
- stessa finestra di valutazione per strategia e benchmark.

Esecuzione:
    python examples/01_momentum_vs_buy_and_hold.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.backtest import BacktestConfig, run_backtest
from quantlab.config import RESEARCH_END
from quantlab.data import load_prices
from quantlab.metrics import summarize
from quantlab.strategies import buy_and_hold, time_series_momentum

CONFIG = BacktestConfig()  # 5 bp per lato, esecuzione il giorno dopo


def load_closes(tickers: list[str]) -> pd.DataFrame:
    closes = {t: load_prices(t, end=RESEARCH_END)["close"] for t in tickers}
    return pd.DataFrame(closes)


def evaluate(result, start: str, end: str) -> pd.Series:
    window = slice(start, end)
    s = summarize(result.returns.loc[window])
    years = s["periods"] / 252
    s["turnover_yr"] = result.turnover.loc[window].sum() / years
    s["cost_yr"] = result.costs.loc[window].sum() / years
    s["invested"] = result.weights.loc[window].abs().sum(axis=1).mean()
    return s


def report(title: str, rows: dict[str, pd.Series]) -> None:
    df = pd.DataFrame(rows).T
    first = df.iloc[0]
    print(f"\n{title}")
    print(f"Finestra di valutazione: {first['start'].date()} → {first['end'].date()} "
          f"({first['periods']} giorni)\n")
    table = pd.DataFrame({
        "CAGR": df["cagr"].map("{:.1%}".format),
        "Vol": df["volatility"].map("{:.1%}".format),
        "Sharpe ± SE": [f"{s:.2f} ± {e:.2f}" for s, e in zip(df["sharpe"], df["sharpe_se"])],
        "Sortino": df["sortino"].map("{:.2f}".format),
        "Max DD": df["max_drawdown"].map("{:.1%}".format),
        "Calmar": df["calmar"].map("{:.2f}".format),
        "Turnover/anno": df["turnover_yr"].map("{:.2f}".format),
        "Costi/anno": df["cost_yr"].map("{:.2%}".format),
        "Investito": df["invested"].map("{:.0%}".format),
    })
    print(table.to_string())


def main() -> None:
    # A) Un solo asset: SPY. Il warm-up di 12 mesi finisce nel 1994.
    spy = load_closes(["SPY"])
    start, end = "1995-01-01", RESEARCH_END
    report("A) SPY: momentum 12 mesi (lungo/liquidità) contro buy-and-hold", {
        "Buy & hold SPY": evaluate(run_backtest(spy, buy_and_hold(spy), CONFIG), start, end),
        "TSMOM 12m": evaluate(
            run_backtest(spy, time_series_momentum(spy), CONFIG), start, end
        ),
    })

    # B) Multi-asset: azioni USA, Nasdaq, Treasury lunghi, oro.
    # GLD è quotato da fine 2004: warm-up fino a fine 2005.
    tickers = ["SPY", "QQQ", "TLT", "GLD"]
    multi = load_closes(tickers)
    start = "2006-01-01"
    report("B) SPY+QQQ+TLT+GLD: momentum contro pesi uguali", {
        "Buy & hold SPY": evaluate(
            run_backtest(multi[["SPY"]], buy_and_hold(multi[["SPY"]]), CONFIG), start, end
        ),
        "Pesi uguali (ribil. mensile)": evaluate(
            run_backtest(multi, buy_and_hold(multi, rebalance="M"), CONFIG), start, end
        ),
        "TSMOM 12m": evaluate(
            run_backtest(multi, time_series_momentum(multi), CONFIG), start, end
        ),
    })

    print(
        "\nNota: risultati storici in-sample, costi ipotizzati (5 bp per lato), "
        "liquidità a rendimento zero. Non sono una previsione di rendimenti futuri."
    )


if __name__ == "__main__":
    main()
