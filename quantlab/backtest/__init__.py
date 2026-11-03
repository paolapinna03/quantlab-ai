"""Motore di backtest (indipendente da dati, strategie e interfaccia)."""

from quantlab.backtest.engine import BacktestConfig, BacktestResult, run_backtest

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]
