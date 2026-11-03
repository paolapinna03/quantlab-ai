"""Swing trading: motore a operazioni (ingresso, stop, target, uscita)."""

from quantlab.swing.engine import (
    SIGNAL_COLUMNS,
    SwingConfig,
    SwingResult,
    run_swing_backtest,
    trade_stats,
)

__all__ = ["SIGNAL_COLUMNS", "SwingConfig", "SwingResult", "run_swing_backtest", "trade_stats"]
