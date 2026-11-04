"""Tasso privo di rischio: rendimento giornaliero della liquidità.

Fonte: ^IRX su Yahoo Finance, il rendimento dei T-bill USA a 13 settimane
(in percentuale annua, quotazione "discount yield"). Storia dal 1960.

Uso in QuantLab:
- la parte di capitale non investita rende questo tasso;
- Sharpe, Sortino e i test statistici usano il rendimento *in eccesso*
  rispetto a questo tasso (definizione originale dello Sharpe ratio).

Approssimazioni dichiarate:
- il discount yield non è esattamente il rendimento di un conto liquidità:
  differenza tipicamente di pochi centesimi di punto;
- in alcuni giorni del 2020 il dato è leggermente negativo (fino a -0,1%):
  lo portiamo a zero, perché un conto liquidità al dettaglio non ha
  rendimento negativo;
- un investitore europeo in euro avrebbe un tasso diverso (€STR/Euribor):
  da aggiungere quando passeremo agli ETF UCITS in euro.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.loader import DEFAULT_CACHE_DIR, drop_incomplete_bar

TBILL_SYMBOL = "^IRX"
TRADING_DAYS = 252


def fetch_tbill_yield() -> pd.Series:
    """Rendimento annuo dei T-bill a 3 mesi, in percentuale, dall'inizio della storia."""
    import yfinance as yf

    raw = yf.download(TBILL_SYMBOL, period="max", interval="1d", auto_adjust=False,
                      progress=False, multi_level_index=False, threads=False)
    s = raw["Close"].astype("float64")
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.dropna().sort_index().rename("yield_pct")


def load_tbill_yield(
    end: str | None = None,
    *,
    refresh: bool = False,
    cache_dir: Path | str | None = None,
    fetcher: Callable[[], pd.Series] = fetch_tbill_yield,
) -> pd.Series:
    """Serie dei rendimenti annui (%) con cache locale, come per i prezzi."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    path = cache_dir / "TBILL_IRX.parquet"
    if refresh or not path.exists():
        s = fetcher()
        s = drop_incomplete_bar(s.to_frame()).iloc[:, 0]
        if s.empty or s.index.has_duplicates or not s.index.is_monotonic_increasing:
            raise ValueError("serie dei tassi non valida")
        if (s.abs() > 25).any():
            raise ValueError("tassi fuori scala: attesi valori in percentuale (es. 4.5)")
        cache_dir.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(path)
        (cache_dir / "TBILL_IRX.meta.json").write_text(json.dumps({
            "symbol": TBILL_SYMBOL,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first_date": str(s.index[0].date()), "last_date": str(s.index[-1].date()),
        }, indent=2))
    else:
        s = pd.read_parquet(path).iloc[:, 0]
    return s.loc[:end]


def daily_cash_returns(yield_pct: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Rendimento della liquidità per ogni giorno di `index`.

    Il tasso noto alla chiusura del giorno precedente si applica al giorno
    corrente (nessuna informazione futura). Giorni senza dato: ultimo valore
    noto. Prima dell'inizio della serie: zero.
    """
    y = yield_pct.clip(lower=0.0) / 100
    y = y.reindex(y.index.union(index)).ffill().shift(1).reindex(index)
    daily = (1 + y) ** (1 / TRADING_DAYS) - 1
    return daily.fillna(0.0).rename("cash_return")


def zero_cash(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.zeros(len(index)), index=index, name="cash_return")
