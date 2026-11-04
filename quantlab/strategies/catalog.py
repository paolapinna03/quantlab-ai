"""Catalogo delle strategie disponibili, per nome.

Un esperimento descrive la strategia con un nome e dei parametri (dati
semplici, salvabili in JSON) invece che con del codice. In futuro un agente
AI potrà proporre esperimenti componendo voci di questo catalogo: solo codice
già scritto, testato e revisionato verrà eseguito.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import pandas as pd

from quantlab.strategies.allocation import risk_allocation
from quantlab.strategies.buy_and_hold import buy_and_hold
from quantlab.strategies.momentum import time_series_momentum
from quantlab.strategies.rules import rule_swing, rule_swing_warmup
from quantlab.strategies.swing import WARMUP as SWING_WARMUP
from quantlab.strategies.swing import donchian_breakout, rsi2_mean_reversion
from quantlab.strategies.trend import trend_signals

Strategy = Callable[..., pd.DataFrame]

# Strategie di portafoglio: prezzi di chiusura → pesi target (motore quantlab.backtest).
PORTFOLIO_CATALOG: dict[str, Strategy] = {
    "buy_and_hold": buy_and_hold,
    "time_series_momentum": time_series_momentum,
    "risk_allocation": risk_allocation,
    "trend_signals": trend_signals,
}

# Strategie swing: dati OHLC → segnali di ingresso/uscita (motore quantlab.swing).
SWING_CATALOG: dict[str, Callable] = {
    "rsi2_mean_reversion": rsi2_mean_reversion,
    "donchian_breakout": donchian_breakout,
    "rule_swing": rule_swing,
}

CATALOG: dict[str, Callable] = PORTFOLIO_CATALOG | SWING_CATALOG


def get_strategy(name: str) -> Callable:
    try:
        return CATALOG[name]
    except KeyError:
        raise ValueError(f"strategia sconosciuta: {name!r}. Disponibili: {sorted(CATALOG)}") from None


def is_swing(name: str) -> bool:
    return name in SWING_CATALOG


def warmup_days(name: str, params: dict) -> int:
    """Giorni di storia necessari prima che i segnali della strategia siano validi."""
    if is_swing(name):
        warm = SWING_WARMUP | {"rule_swing": rule_swing_warmup}
        return int(warm[name](full_params(name, params)))
    p = full_params(name, params)
    vol = p.get("vol_lookback", 0) if p.get("vol_target", 0) > 0 else 0
    return int(max(p.get("lookback", 0), p.get("momentum_filter", 0),
                   max(p.get("horizons", (0,))), vol))


def _strategy_params(name: str) -> list[inspect.Parameter]:
    """Parametri configurabili: tutti tranne il primo (i dati)."""
    return list(inspect.signature(get_strategy(name)).parameters.values())[1:]


def full_params(name: str, params: dict) -> dict:
    """Parametri completi: quelli passati + i default della strategia.

    Così {"lookback": 252} e {"lookback": 252, "long_only": True} risultano
    la stessa configurazione (e lo stesso tentativo).
    """
    defaults = {
        p.name: p.default for p in _strategy_params(name)
        if p.default is not inspect.Parameter.empty
    }
    return defaults | dict(params)


def check_params(name: str, params: dict) -> None:
    """Verifica che i parametri esistano nella firma della strategia."""
    allowed = [p.name for p in _strategy_params(name)]
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(f"parametri non validi per {name}: {sorted(unknown)}. Ammessi: {allowed}")
