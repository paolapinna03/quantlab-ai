"""Test del modulo dati.

I test normali usano dati sintetici e un finto fetcher: girano offline, in
pochi millisecondi, e danno sempre lo stesso risultato. Il test marcato
`network` scarica dati veri e si lancia solo a richiesta:
    pytest -m network
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from quantlab.data.loader import drop_incomplete_bar
from quantlab.data import (
    DataValidationError,
    cache_info,
    load_prices,
    normalize_ohlcv,
    validate_ohlcv,
)


def make_ohlcv(n: int = 30, start: str = "2024-01-01", seed: int = 0) -> pd.DataFrame:
    """Serie OHLCV sintetica ma coerente (high >= open/close >= low)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * 1.005
    low = np.minimum(open_, close) * 0.995
    volume = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class CountingFetcher:
    """Finto fetcher che conta quante volte viene chiamato."""

    __name__ = "fake_fetcher"

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.calls = 0

    def __call__(self, ticker: str) -> pd.DataFrame:
        self.calls += 1
        return self.df.copy()


# --- Validazione ----------------------------------------------------------

def test_clean_data_passes():
    report = validate_ohlcv(make_ohlcv())
    assert report.ok
    assert report.warnings == []


def test_empty_data_is_error():
    assert not validate_ohlcv(pd.DataFrame()).ok


def test_duplicate_dates_are_error():
    df = make_ohlcv()
    df = pd.concat([df, df.iloc[[5]]]).sort_index()
    report = validate_ohlcv(df)
    assert any("duplicate" in e for e in report.errors)


def test_unsorted_dates_are_error():
    df = make_ohlcv().iloc[::-1]
    assert any("ordine" in e for e in validate_ohlcv(df).errors)


def test_non_positive_price_is_error():
    df = make_ohlcv()
    df.iloc[3, df.columns.get_loc("close")] = 0.0
    assert any("<= 0" in e for e in validate_ohlcv(df).errors)


def test_missing_close_is_error():
    df = make_ohlcv()
    df.iloc[3, df.columns.get_loc("close")] = np.nan
    assert any("mancanti" in e for e in validate_ohlcv(df).errors)


def test_extreme_return_is_warning_not_error():
    df = make_ohlcv()
    df.iloc[10:, :4] *= 0.5  # crollo del 50% in un giorno
    report = validate_ohlcv(df)
    assert report.ok
    assert any("estremo" in w for w in report.warnings)


def test_incoherent_high_low_is_warning():
    df = make_ohlcv()
    df.iloc[4, df.columns.get_loc("high")] = df["low"].iloc[4] * 0.5
    report = validate_ohlcv(df)
    assert any("incoerenti" in w for w in report.warnings)


# --- Normalizzazione -------------------------------------------------------

def test_normalize_yfinance_style_frame():
    raw = make_ohlcv().rename(columns=str.capitalize)
    raw.index = raw.index.tz_localize("America/New_York")
    raw.index.name = "Date"
    df = normalize_ohlcv(raw[["Close", "High", "Low", "Open", "Volume"]])
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is None
    assert df.index.name == "date"


# --- Barra incompleta del giorno ---------------------------------------------

def _two_bars():
    return make_ohlcv(n=2, start="2024-03-14")  # gio 14 e ven 15 marzo


def test_todays_bar_dropped_during_session():
    # Venerdì 15 marzo 2024, ore 12:00 a New York (16:00 UTC): borsa aperta.
    now = datetime(2024, 3, 15, 16, 0, tzinfo=timezone.utc)
    df = drop_incomplete_bar(_two_bars(), now=now)
    assert df.index[-1] == pd.Timestamp("2024-03-14")


def test_todays_bar_kept_after_close():
    # Venerdì 15 marzo 2024, ore 17:00 a New York (21:00 UTC): borsa chiusa.
    now = datetime(2024, 3, 15, 21, 0, tzinfo=timezone.utc)
    df = drop_incomplete_bar(_two_bars(), now=now)
    assert df.index[-1] == pd.Timestamp("2024-03-15")


# --- Cache -----------------------------------------------------------------

def test_cache_avoids_second_download(tmp_path):
    fetcher = CountingFetcher(make_ohlcv())
    first = load_prices("TEST", cache_dir=tmp_path, fetcher=fetcher)
    second = load_prices("TEST", cache_dir=tmp_path, fetcher=fetcher)
    assert fetcher.calls == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_refresh_forces_download(tmp_path):
    fetcher = CountingFetcher(make_ohlcv())
    load_prices("TEST", cache_dir=tmp_path, fetcher=fetcher)
    load_prices("TEST", cache_dir=tmp_path, fetcher=fetcher, refresh=True)
    assert fetcher.calls == 2


def test_date_range_is_inclusive(tmp_path):
    fetcher = CountingFetcher(make_ohlcv(n=30, start="2024-01-01"))
    df = load_prices("TEST", "2024-01-03", "2024-01-05", cache_dir=tmp_path, fetcher=fetcher)
    assert df.index[0] == pd.Timestamp("2024-01-03")
    assert df.index[-1] == pd.Timestamp("2024-01-05")
    assert len(df) == 3


def test_metadata_is_written(tmp_path):
    fetcher = CountingFetcher(make_ohlcv())
    load_prices("TEST", cache_dir=tmp_path, fetcher=fetcher)
    info = cache_info("TEST", cache_dir=tmp_path)
    assert info["source"] == "fake_fetcher"
    assert info["rows"] == 30
    assert "fetched_at" in info


def test_bad_data_is_never_cached(tmp_path):
    bad = make_ohlcv()
    bad.iloc[2, bad.columns.get_loc("close")] = -1.0
    with pytest.raises(DataValidationError):
        load_prices("BAD", cache_dir=tmp_path, fetcher=CountingFetcher(bad))
    assert cache_info("BAD", cache_dir=tmp_path) is None


# --- Dati reali (solo con: pytest -m network) -------------------------------

@pytest.mark.network
def test_real_spy_download(tmp_path):
    df = load_prices("SPY", "2020-01-01", "2020-12-31", cache_dir=tmp_path)
    assert validate_ohlcv(df).ok
    assert 250 <= len(df) <= 254  # circa 252 giorni di borsa in un anno
