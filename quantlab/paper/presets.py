"""Strategie pronte per il paper trading (usate da dashboard e terminale)."""

from __future__ import annotations

from quantlab.config import DEFAULT_CASH, DEFAULT_COST_BPS

PRESETS = {
    "rp_momentum": {
        "spec": {
            "strategy": "risk_allocation",
            "params": {"method": "risk_parity", "lookback": 252, "max_weight": 0.4,
                       "shrinkage": True, "momentum_filter": 252},
            "universe": ["SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
            "cost_bps": DEFAULT_COST_BPS, "cash": DEFAULT_CASH,
        },
        "note": "Risk parity + filtro momentum 12m, famiglia allocazione_etf (non promossa: "
                "5/6 criteri, non significativamente migliore di 1/N). Track record di prova.",
    },
}

PRESET_LABELS = {"rp_momentum": "Risk parity + momentum 12 mesi (9 ETF multi-asset)"}
