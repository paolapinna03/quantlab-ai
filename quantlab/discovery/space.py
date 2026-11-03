"""Spazio di ricerca: tutte le combinazioni che la Discovery può provare.

Lo spazio è dichiarato per intero PRIMA della ricerca. La sua dimensione
viene registrata: è un'informazione necessaria per giudicare quanto sia
sorprendente la migliore candidata.
"""

from __future__ import annotations

import itertools
import random

# Per ogni tipo di ingresso: valori di lookback e threshold sensati.
ENTRY_GRID = {
    "rsi_dip": {"lookback": [2, 3, 5], "threshold": [10.0, 20.0, 30.0]},
    "breakout": {"lookback": [20, 55, 100], "threshold": [0.0]},
    "ma_cross": {"lookback": [10, 20, 50], "threshold": [0.0]},
    "momentum": {"lookback": [20, 60, 120], "threshold": [0.0, 0.05]},
    "high_52w": {"lookback": [252], "threshold": [0.02, 0.05]},
}
TREND_FILTERS = [0, 200]
EXIT_RULES = ["opposite", "none"]
STOPS_ATR = [0.0, 2.0, 3.0]
TARGETS_ATR = [0.0, 3.0]
MAX_HOLD = [None, 10, 40]


def full_space() -> list[tuple[dict, dict]]:
    """Tutte le combinazioni valide come (params della strategia, impostazioni del motore)."""
    combos = []
    for entry, grid in ENTRY_GRID.items():
        for lb, th, tf, ex, st, tg, mh in itertools.product(
            grid["lookback"], grid["threshold"], TREND_FILTERS, EXIT_RULES,
            STOPS_ATR, TARGETS_ATR, MAX_HOLD,
        ):
            # Una posizione deve poter uscire in qualche modo.
            if ex == "none" and st == 0 and tg == 0 and mh is None:
                continue
            params = {"entry": entry, "lookback": lb, "threshold": th, "trend_filter": tf,
                      "exit_rule": ex, "stop_atr": st, "target_atr": tg}
            engine = {"max_holding_days": mh} if mh else {}
            combos.append((params, engine))
    return combos


def sample_space(n: int, seed: int) -> list[tuple[dict, dict]]:
    """Campione casuale ma riproducibile di `n` combinazioni (senza ripetizioni)."""
    space = full_space()
    if n >= len(space):
        return space
    return random.Random(seed).sample(space, n)
