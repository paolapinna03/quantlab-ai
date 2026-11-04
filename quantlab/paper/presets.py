"""Strategie pronte per il paper trading (usate da dashboard e terminale).

Le due strategie usano lo stesso universo, gli stessi costi e la stessa
liquidità: così il confronto dal vivo isola la differenza tra i metodi.
"""

from __future__ import annotations

from quantlab.config import DEFAULT_CASH, DEFAULT_COST_BPS

ETF_9 = ["SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "GLD", "DBC", "VNQ"]

PRESETS = {
    "rp_momentum": {
        "spec": {
            "strategy": "risk_allocation",
            "params": {"method": "risk_parity", "lookback": 252, "max_weight": 0.4,
                       "shrinkage": True, "momentum_filter": 252},
            "universe": ETF_9, "cost_bps": DEFAULT_COST_BPS, "cash": DEFAULT_CASH,
        },
        "note": "Risk parity + filtro momentum 12m, famiglia allocazione_etf (non promossa: "
                "5/6 criteri, non significativamente migliore di 1/N). Track record di prova.",
    },
    "tsmom_9etf": {
        "spec": {
            "strategy": "time_series_momentum",
            "params": {"lookback": 252, "long_only": True},
            "universe": ETF_9, "cost_bps": DEFAULT_COST_BPS, "cash": DEFAULT_CASH,
        },
        "note": "Time-series momentum 12m, pesi 1/N, famiglia tsmom_etf (non promossa). Stessi 9 ETF "
                "di rp_momentum: nel backtest 2007-2022 rendimento simile, rischio più alto "
                "(correlazione 0,81, differenza non significativa).",
    },
    "ai_analyst": {
        "spec": {
            "strategy": "ai_ideas", "params": {},
            "universe": ETF_9, "cost_bps": DEFAULT_COST_BPS, "cash": DEFAULT_CASH,
        },
        "note": "Idee proprie dell'analista AI (Claude), registrate per misurarne nel tempo il valore. "
                "Non è una strategia validata: è un esperimento.",
    },
}

PRESET_LABELS = {
    "rp_momentum": "Risk parity + momentum 12 mesi (9 ETF multi-asset)",
    "tsmom_9etf": "Momentum 12 mesi, pesi uguali (stessi 9 ETF)",
    "ai_analyst": "Idee dell'analista AI (esperimento, stessi 9 ETF)",
}
