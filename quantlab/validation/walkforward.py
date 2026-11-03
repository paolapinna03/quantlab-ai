"""Walk-forward analysis: scegli i parametri sul passato, misurali sul futuro.

Per ogni "fold":
1. finestra di training: si calcola lo Sharpe di ogni combinazione di parametri
   usando SOLO i rendimenti di quella finestra;
2. si sceglie la combinazione migliore;
3. la si applica alla finestra di test successiva, mai vista durante la scelta.

Le finestre di test, messe in fila, formano una serie di rendimenti
"pseudo out-of-sample": ogni giorno è stato tradato con parametri scelti senza
conoscerlo. Se una strategia funziona solo con il senno di poi, qui crolla.

"Pseudo" perché i dati sono comunque quelli su cui abbiamo fatto ricerca:
il vero out-of-sample resta l'holdout dal 2023.

Dettaglio importante: si esegue UN solo backtest sui pesi target "cuciti"
fold per fold, così i costi dei cambi di parametro tra un fold e l'altro
vengono pagati come nella realtà.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from quantlab.backtest import BacktestConfig, BacktestResult, run_backtest, warmup_end
from quantlab.metrics import sharpe_ratio


@dataclass
class WalkForwardResult:
    returns: pd.Series          # rendimenti pseudo-OOS (solo finestre di test)
    folds: pd.DataFrame         # una riga per fold: finestre, parametri scelti, Sharpe
    backtest: BacktestResult    # backtest completo sui pesi cuciti
    param_grid: list[dict]


def walk_forward(
    prices: pd.DataFrame,
    strategy: Callable[..., pd.DataFrame],
    param_grid: list[dict],
    train_years: int = 5,
    test_years: int = 1,
    anchored: bool = True,
    config: BacktestConfig | None = None,
) -> WalkForwardResult:
    """Walk-forward con finestre annuali.

    Args:
        anchored: True = il training parte sempre dall'inizio dei dati e si allunga;
            False = finestra mobile di `train_years` anni.
    """
    if not param_grid:
        raise ValueError("param_grid vuota")
    config = config or BacktestConfig()

    # Ogni combinazione genera i suoi pesi: le strategie sono causali, quindi
    # calcolarle su tutta la storia non usa informazioni future.
    targets = {i: strategy(prices, **p) for i, p in enumerate(param_grid)}
    runs = {i: run_backtest(prices, t, config) for i, t in targets.items()}

    max_lookback = max(int(p.get("lookback", 0)) for p in param_grid)
    data_start = warmup_end(prices, max_lookback)
    first_test = data_start + pd.DateOffset(years=train_years)
    last_date = prices.index[-1]

    stitched = pd.DataFrame(float("nan"), index=prices.index, columns=prices.columns)
    rows = []
    test_start = first_test
    while test_start < last_date:
        test_end = min(test_start + pd.DateOffset(years=test_years), last_date + pd.Timedelta(days=1))
        train_start = data_start if anchored else test_start - pd.DateOffset(years=train_years)
        train = slice(train_start, test_start - pd.Timedelta(days=1))
        test = slice(test_start, test_end - pd.Timedelta(days=1))

        train_sharpes = {i: sharpe_ratio(r.returns.loc[train]) for i, r in runs.items()}
        best = max(train_sharpes, key=lambda i: (pd.notna(train_sharpes[i]), train_sharpes[i]))
        stitched.loc[test] = targets[best].loc[test]

        rows.append({
            "train_start": train_start.date(), "train_end": (test_start - pd.Timedelta(days=1)).date(),
            "test_start": test_start.date(), "test_end": (test_end - pd.Timedelta(days=1)).date(),
            "params": param_grid[best], "train_sharpe": train_sharpes[best],
            "test_sharpe_chosen": sharpe_ratio(runs[best].returns.loc[test]),
        })
        test_start = test_end

    if not rows:
        raise ValueError("storia troppo corta per anche un solo fold")

    bt = run_backtest(prices, stitched, config)
    oos = bt.returns.loc[first_test:]
    return WalkForwardResult(returns=oos, folds=pd.DataFrame(rows), backtest=bt,
                             param_grid=param_grid)
