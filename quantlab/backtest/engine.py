"""Motore di backtest su pesi target, con deriva del portafoglio e costi.

Il contratto tra strategia e motore
-----------------------------------
Una strategia produce un DataFrame `targets` (righe = date, colonne = asset)
con i **pesi desiderati** calcolati usando solo dati fino alla chiusura di
quella data.

- Una riga con almeno un valore = "ribilancia a questi pesi".
  I NaN dentro una riga attiva valgono 0 (es. asset non ancora quotato).
- Una riga tutta NaN = "nessuna operazione": i pesi derivano con i prezzi.
- Peso 1.0 = 100% del capitale; la parte non investita resta in liquidità
  (rendimento 0, ipotesi semplificativa).

Tempistica (il punto più delicato di ogni backtest)
---------------------------------------------------
Con `execution_lag = 1` (default) un segnale calcolato sulla chiusura del
giorno t viene eseguito sulla chiusura del giorno t+1 e inizia a produrre
rendimento dal giorno t+2. È prudente: lascia un giorno intero per calcolare
il segnale e inviare l'ordine.

`execution_lag = 0` significa eseguire alla stessa chiusura usata per il
segnale: possibile solo in teoria (ordini "market on close" calcolati
qualche minuto prima della chiusura), quindi leggermente ottimistico.

Ogni giorno, nell'ordine:
1. i pesi detenuti producono il rendimento lordo del giorno;
2. i pesi derivano (un asset che sale pesa di più);
3. alla chiusura si ribilancia ai nuovi target, se ci sono, pagando i costi.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from quantlab.config import DEFAULT_COST_BPS

_EPS = 1e-12


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: float = DEFAULT_COST_BPS
    execution_lag: int = 1
    max_gross_exposure: float = 1.0
    initial_capital: float = 100_000.0

    def __post_init__(self):
        if self.execution_lag < 0:
            raise ValueError("execution_lag non può essere negativo")
        if self.cost_bps < 0:
            raise ValueError("cost_bps non può essere negativo")


@dataclass
class BacktestResult:
    returns: pd.Series          # rendimenti netti giornalieri (dopo i costi)
    gross_returns: pd.Series    # rendimenti prima dei costi
    equity: pd.Series           # valore del portafoglio
    weights: pd.DataFrame       # pesi detenuti a fine giornata (dopo il ribilanciamento)
    turnover: pd.Series         # somma |variazioni di peso| per giorno
    costs: pd.Series            # costi pagati, in frazione del capitale
    trades: pd.DataFrame        # una riga per ogni variazione di peso
    config: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BacktestResult({self.returns.index[0].date()} → {self.returns.index[-1].date()}, "
            f"equity finale {self.equity.iloc[-1]:,.0f}, {len(self.trades)} operazioni)"
        )


def run_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Simula una strategia a partire dai prezzi di chiusura e dai pesi target.

    Args:
        prices: prezzi di chiusura aggiustati, una colonna per asset.
        targets: pesi desiderati (vedi docstring del modulo).
        config: costi, ritardo di esecuzione, leva massima, capitale iniziale.
    """
    config = config or BacktestConfig()
    _check_inputs(prices, targets)

    prices = prices.sort_index()
    assets = list(prices.columns)
    # Allinea i target al calendario dei prezzi, poi applica il ritardo.
    targets = targets.reindex(index=prices.index, columns=assets)
    executed = targets.shift(config.execution_lag)

    asset_returns = prices.pct_change().to_numpy()
    price_ok = prices.notna().to_numpy()
    tgt = executed.to_numpy()
    cost_rate = config.cost_bps / 10_000

    n, k = prices.shape
    w = np.zeros(k)
    out_gross = np.zeros(n)
    out_net = np.zeros(n)
    out_turnover = np.zeros(n)
    out_cost = np.zeros(n)
    out_weights = np.zeros((n, k))
    trades = []

    for i in range(n):
        r = np.nan_to_num(asset_returns[i], nan=0.0)

        # 1. Rendimento lordo con i pesi detenuti da ieri.
        gross = float(w @ r)
        if 1 + gross <= 0:
            raise RuntimeError(f"capitale azzerato il {prices.index[i].date()}")

        # 2. Deriva dei pesi.
        w_pre = w * (1 + r) / (1 + gross)

        # 3. Ribilanciamento, se c'è un target per oggi.
        row = tgt[i]
        if np.isnan(row).all():
            w_new = w_pre
            turnover = 0.0
        else:
            w_new = np.nan_to_num(row, nan=0.0)
            _check_row(w_new, price_ok[i], assets, prices.index[i], config)
            delta = w_new - w_pre
            turnover = float(np.abs(delta).sum())
            for j in np.flatnonzero(np.abs(delta) > 1e-9):
                trades.append(
                    (prices.index[i], assets[j], w_pre[j], w_new[j], delta[j])
                )

        cost = turnover * cost_rate
        out_gross[i] = gross
        out_net[i] = (1 + gross) * (1 - cost) - 1
        out_turnover[i] = turnover
        out_cost[i] = cost
        out_weights[i] = w_new
        w = w_new

    idx = prices.index
    net = pd.Series(out_net, index=idx, name="returns")
    return BacktestResult(
        returns=net,
        gross_returns=pd.Series(out_gross, index=idx, name="gross_returns"),
        equity=config.initial_capital * (1 + net).cumprod().rename("equity"),
        weights=pd.DataFrame(out_weights, index=idx, columns=assets),
        turnover=pd.Series(out_turnover, index=idx, name="turnover"),
        costs=pd.Series(out_cost, index=idx, name="costs"),
        trades=pd.DataFrame(
            trades, columns=["date", "asset", "weight_before", "weight_after", "change"]
        ),
        config=asdict(config),
    )


def _check_inputs(prices: pd.DataFrame, targets: pd.DataFrame) -> None:
    if prices.empty:
        raise ValueError("nessun prezzo")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("l'indice dei prezzi deve essere di date")
    unknown = set(targets.columns) - set(prices.columns)
    if unknown:
        raise ValueError(f"target per asset senza prezzi: {sorted(unknown)}")
    outside = targets.index.difference(prices.index)
    if len(outside):
        raise ValueError(f"{len(outside)} date dei target non presenti nei prezzi")


def _check_row(w, price_ok, assets, date, config: BacktestConfig) -> None:
    gross = np.abs(w).sum()
    if gross > config.max_gross_exposure + 1e-9:
        raise ValueError(
            f"{date.date()}: esposizione lorda {gross:.2f} oltre il limite "
            f"{config.max_gross_exposure:.2f}"
        )
    bad = (np.abs(w) > _EPS) & ~price_ok
    if bad.any():
        names = [a for a, b in zip(assets, bad) if b]
        raise ValueError(f"{date.date()}: posizione su asset senza prezzo: {names}")
