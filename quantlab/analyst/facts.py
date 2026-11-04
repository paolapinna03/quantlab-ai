"""Fatti del giorno, calcolati in modo deterministico.

Sono l'unica fonte di numeri del resoconto: l'AI riceve questi fatti e non
deve introdurne altri. Gli ordini da eseguire sono scritti dal codice, non
dall'AI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.paper.presets import ETF_9, PRESET_LABELS
from quantlab.paper.store import PaperStore

ETF_NAMES = {
    "SPY": "azioni USA", "EFA": "azioni sviluppate ex-USA", "EEM": "azioni emergenti",
    "IWM": "small cap USA", "TLT": "Treasury USA 20+ anni", "IEF": "Treasury USA 7-10 anni",
    "GLD": "oro", "DBC": "materie prime", "VNQ": "immobiliare USA",
}


def last_weekday_of_month(day: pd.Timestamp) -> pd.Timestamp:
    """Ultimo giorno feriale del mese (approssimazione: festività di borsa escluse)."""
    end = (day + pd.offsets.MonthEnd(0)).normalize()
    while end.weekday() >= 5:
        end -= pd.Timedelta(days=1)
    return end


def signal_table(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Per ogni ETF: rendimento a 12 mesi (il segnale), a 1 giorno e a 1 mese, volatilità a 3 mesi."""
    last = prices.iloc[-1]
    rows = {}
    for t in prices.columns:
        s = prices[t].dropna()
        if len(s) <= lookback:
            continue
        r = s.pct_change()
        rows[t] = {
            "nome": ETF_NAMES.get(t, t),
            "rend_12m": float(s.iloc[-1] / s.iloc[-1 - lookback] - 1),
            "rend_1g": float(r.iloc[-1]),
            "rend_1m": float(s.iloc[-1] / s.iloc[-22] - 1),
            "vol_3m": float(r.iloc[-63:].std() * np.sqrt(252)),
            "trend": "positivo" if s.iloc[-1] > s.iloc[-1 - lookback] else "negativo",
            "prezzo": float(last[t]),
        }
    return pd.DataFrame(rows).T


def account_facts(store: PaperStore, name: str) -> dict | None:
    acc = store.account(name)
    last = store.last_day(name)
    if last is None:
        return None
    eq = last["equity"]
    pend = store.pending(name)
    orders = []
    if pend:
        current = {a: v / eq for a, v in last["values"].items()}
        for a, w in sorted(pend["targets"].items()):
            delta = w - current.get(a, 0.0)
            if abs(delta) > 0.0025:
                orders.append({"etf": a, "azione": "compra" if delta > 0 else "vendi",
                               "peso_da": round(current.get(a, 0.0), 4), "peso_a": round(w, 4),
                               "controvalore": round(abs(delta) * eq, 2)})
    events = store.events(name)
    today_events = (events[events["date"] == str(last["date"].date())]
                    if len(events) else events)
    return {
        "conto": name,
        "strategia": PRESET_LABELS.get(name, acc["spec"]["strategy"]),
        "stato": acc["status"],
        "data": str(last["date"].date()),
        "valore": round(eq, 2),
        "rendimento_da_avvio": round(eq / acc["initial_capital"] - 1, 4),
        "drawdown": round(eq / last["peak"] - 1, 4),
        "liquidita": round(last["cash"] / eq, 4),
        "posizioni": {a: round(v / eq, 4) for a, v in last["values"].items() if v / eq > 0.0025},
        "ordini_prossima_chiusura": orders,
        "eventi_oggi": [f"{e['kind']}: {e['message']}" for _, e in today_events.iterrows()
                        if e["kind"] != "segnale"],
    }


def build_facts(store: PaperStore, prices: pd.DataFrame, tbill_pct: float | None) -> dict:
    """Tutti i fatti del giorno in un dizionario serializzabile."""
    sig = signal_table(prices[[c for c in ETF_9 if c in prices.columns]])
    day = prices.index[-1]
    next_rebal = last_weekday_of_month(day)
    if next_rebal <= day:
        next_rebal = last_weekday_of_month(day + pd.offsets.MonthBegin(1))
    accounts = [f for f in (account_facts(store, a["name"]) for a in store.accounts()) if f]
    return {
        "data": str(day.date()),
        "prossimo_ribilanciamento_strategie": str(next_rebal.date()),
        "tasso_tbill_annuo": round(tbill_pct / 100, 4) if tbill_pct is not None else None,
        "segnali": {t: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                    for t, row in sig.to_dict(orient="index").items()},
        "conti": accounts,
    }
