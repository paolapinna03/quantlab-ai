"""Strategie: funzioni da prezzi a pesi target."""

from quantlab.strategies.buy_and_hold import buy_and_hold
from quantlab.strategies.momentum import time_series_momentum

__all__ = ["buy_and_hold", "time_series_momentum"]
