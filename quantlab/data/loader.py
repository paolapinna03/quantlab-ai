"""Scaricamento e cache dei prezzi giornalieri.

Scelte di progetto:

1. **Prezzi aggiustati** (`auto_adjust=True`): dividendi e frazionamenti sono
   incorporati nei prezzi, quindi i rendimenti calcolati sono *total return*.
   Senza questo, un ETF che stacca un dividendo sembrerebbe "perdere" valore.

2. **Cache = istantanea (snapshot)**: Yahoo ricalcola retroattivamente i prezzi
   aggiustati a ogni nuovo dividendo, quindi lo stesso download fatto in giorni
   diversi può dare numeri leggermente diversi. Salvando una copia locale con la
   data di download, ogni esperimento sa *esattamente* su quali dati è girato.

3. **Fonte sostituibile** (`fetcher`): il resto del codice non sa che i dati
   arrivano da yfinance. Nei test usiamo un finto fetcher (niente Internet),
   e in futuro potremo passare a una fonte professionale senza riscrivere nulla.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.data.validation import OHLCV_COLUMNS, validate_ohlcv

Fetcher = Callable[[str], pd.DataFrame]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = Path(
    os.environ.get("QUANTLAB_CACHE_DIR", _PROJECT_ROOT / "data" / "cache")
)


def fetch_yfinance(ticker: str) -> pd.DataFrame:
    """Scarica l'intera storia giornaliera di un ticker da Yahoo Finance.

    Nota: yfinance è una libreria *non ufficiale*. Va bene per la ricerca
    iniziale, non per decisioni con denaro reale.
    """
    import yfinance as yf  # import locale: i test non ne hanno bisogno

    raw = yf.download(
        ticker,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
        threads=False,
    )
    return normalize_ohlcv(raw)


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Porta un DataFrame OHLCV al formato standard di QuantLab.

    Colonne minuscole in ordine fisso, indice di date senza fuso orario
    chiamato "date", righe ordinate e senza barre completamente vuote.
    """
    df = raw.rename(columns=str.lower)
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "date"
    df = df.dropna(how="all").sort_index()
    df = df.astype({c: "float64" for c in df.columns})
    return df


# La borsa USA chiude alle 16:00 di New York; aspettiamo un margine prima di
# considerare definitiva la barra del giorno.
_MARKET_TZ = "America/New_York"
_BAR_FINAL_AFTER = pd.Timedelta(hours=16, minutes=30)


def drop_incomplete_bar(df: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    """Elimina la barra di oggi se la borsa USA non ha ancora chiuso.

    Durante la seduta, Yahoo restituisce una barra "di oggi" il cui close è
    solo l'ultimo prezzo scambiato. Salvarla in cache significherebbe usare
    un prezzo di chiusura che non è mai esistito.
    """
    now_ny = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert(_MARKET_TZ)
    today = now_ny.normalize().tz_localize(None)
    bar_is_final = now_ny - now_ny.normalize() >= _BAR_FINAL_AFTER
    if bar_is_final:
        return df[df.index <= today]
    return df[df.index < today]


def _cache_paths(ticker: str, cache_dir: Path) -> tuple[Path, Path]:
    safe = ticker.upper().replace("/", "_").replace("^", "IDX_")
    return cache_dir / f"{safe}.parquet", cache_dir / f"{safe}.meta.json"


def load_prices(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    refresh: bool = False,
    cache_dir: Path | str | None = None,
    fetcher: Fetcher = fetch_yfinance,
) -> pd.DataFrame:
    """Restituisce i prezzi OHLCV giornalieri di `ticker`, usando la cache.

    Args:
        ticker: simbolo, es. "SPY".
        start, end: date "AAAA-MM-GG", entrambe *incluse*. None = nessun limite.
        refresh: se True, riscarica i dati ignorando la cache.
        cache_dir: cartella della cache (default: data/cache del progetto).
        fetcher: funzione che scarica la storia completa di un ticker.

    Raises:
        DataValidationError: se i dati non superano i controlli di qualità.
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    data_path, meta_path = _cache_paths(ticker, cache_dir)

    if refresh or not data_path.exists():
        df = drop_incomplete_bar(fetcher(ticker))
        report = validate_ohlcv(df)
        report.raise_if_errors(ticker)
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path)
        meta = {
            "ticker": ticker.upper(),
            "source": getattr(fetcher, "__name__", "unknown"),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "first_date": str(df.index[0].date()),
            "last_date": str(df.index[-1].date()),
            "rows": len(df),
            "warnings": report.warnings,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    else:
        df = pd.read_parquet(data_path)

    return df.loc[start:end]


def cache_info(ticker: str, cache_dir: Path | str | None = None) -> dict | None:
    """Metadati della cache di `ticker` (data di download, intervallo, avvisi)."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    _, meta_path = _cache_paths(ticker, cache_dir)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())
