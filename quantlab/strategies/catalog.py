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

from quantlab.strategies.buy_and_hold import buy_and_hold
from quantlab.strategies.momentum import time_series_momentum

Strategy = Callable[..., pd.DataFrame]

CATALOG: dict[str, Strategy] = {
    "buy_and_hold": buy_and_hold,
    "time_series_momentum": time_series_momentum,
}


def get_strategy(name: str) -> Strategy:
    try:
        return CATALOG[name]
    except KeyError:
        raise ValueError(f"strategia sconosciuta: {name!r}. Disponibili: {sorted(CATALOG)}") from None


def full_params(name: str, params: dict) -> dict:
    """Parametri completi: quelli passati + i default della strategia.

    Così {"lookback": 252} e {"lookback": 252, "long_only": True} risultano
    la stessa configurazione (e lo stesso tentativo).
    """
    sig = inspect.signature(get_strategy(name))
    defaults = {
        p.name: p.default for p in sig.parameters.values()
        if p.name != "prices" and p.default is not inspect.Parameter.empty
    }
    return defaults | dict(params)


def check_params(name: str, params: dict) -> None:
    """Verifica che i parametri esistano nella firma della strategia."""
    sig = inspect.signature(get_strategy(name))
    allowed = [p for p in sig.parameters if p != "prices"]
    unknown = set(params) - set(allowed)
    if unknown:
        raise ValueError(f"parametri non validi per {name}: {sorted(unknown)}. Ammessi: {allowed}")
