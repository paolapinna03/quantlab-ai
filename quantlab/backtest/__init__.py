"""Motore di backtest (indipendente da dati, strategie e interfaccia)."""

from quantlab.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from quantlab.backtest.report import evaluate, excess, warmup_end

__all__ = ["BacktestConfig", "BacktestResult", "evaluate", "excess", "run_backtest", "warmup_end"]
