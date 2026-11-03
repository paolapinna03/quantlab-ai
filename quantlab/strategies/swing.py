"""Strategie swing: da dati OHLC a segnali di ingresso/uscita.

Ogni strategia restituisce, per ogni asset, un DataFrame con le colonne
descritte in `quantlab.swing.engine` (entry, exit, stop_distance,
target_distance, score), calcolate solo con dati fino alla chiusura di t.

Provenienza delle regole (importante distinguere):
- Le due strategie qui sotto sono regole note tra i trader, pubblicate in
  libri divulgativi, NON in riviste peer-reviewed. Le usiamo come candidati
  da sottoporre al laboratorio, non come strategie "dimostrate".
- Il fenomeno su cui si appoggiano è invece documentato in letteratura:
  inversione di breve periodo dei rendimenti (Jegadeesh 1990, Journal of
  Finance; Lehmann 1990, Quarterly Journal of Economics) e trend following
  (Hurst, Ooi & Pedersen 2017, AQR).
"""

from __future__ import annotations

import pandas as pd

from quantlab.indicators import atr, donchian_high_prior, donchian_low_prior, rsi, sma


def rsi2_mean_reversion(
    data: dict[str, pd.DataFrame],
    rsi_period: int = 2,
    entry_rsi: float = 10.0,
    trend_sma: int = 200,
    exit_sma: int = 5,
    stop_atr: float | None = None,
) -> dict[str, pd.DataFrame]:
    """Compra le discese brevi dentro un trend rialzista.

    Ingresso: chiusura sopra la media a `trend_sma` giorni (trend rialzista)
    e RSI a `rsi_period` giorni sotto `entry_rsi` (forte calo recente).
    Uscita: chiusura sopra la media a `exit_sma` giorni (il prezzo è rimbalzato).
    Priorità: i titoli più ipervenduti per primi.

    Origine: regola divulgata da L. Connors e C. Alvarez, "Short Term Trading
    Strategies That Work" (2009), libro non peer-reviewed.
    """
    out = {}
    for asset, df in data.items():
        c = df["close"]
        r = rsi(c, rsi_period)
        sig = pd.DataFrame(index=df.index)
        sig["entry"] = (c > sma(c, trend_sma)) & (r < entry_rsi)
        sig["exit"] = c > sma(c, exit_sma)
        sig["stop_distance"] = stop_atr * atr(df["high"], df["low"], c, 14) if stop_atr else float("nan")
        sig["target_distance"] = float("nan")
        sig["score"] = -r
        out[asset] = sig
    return out


def donchian_breakout(
    data: dict[str, pd.DataFrame],
    entry_window: int = 55,
    exit_window: int = 20,
    atr_window: int = 20,
    stop_atr: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Compra le rotture dei massimi, esce sulla rottura dei minimi.

    Ingresso: chiusura sopra il massimo degli `entry_window` giorni precedenti.
    Uscita: chiusura sotto il minimo degli `exit_window` giorni precedenti.
    Stop iniziale: `stop_atr` × ATR sotto il prezzo d'ingresso.
    Priorità: le rotture più ampie (in unità di ATR) per prime.

    Origine: regole dei "Turtle Traders" (anni '80), descritte da C. Faith,
    "Way of the Turtle" (2007), libro non peer-reviewed.
    """
    out = {}
    for asset, df in data.items():
        c = df["close"]
        a = atr(df["high"], df["low"], c, atr_window)
        upper = donchian_high_prior(df["high"], entry_window)
        sig = pd.DataFrame(index=df.index)
        sig["entry"] = c > upper
        sig["exit"] = c < donchian_low_prior(df["low"], exit_window)
        sig["stop_distance"] = stop_atr * a
        sig["target_distance"] = float("nan")
        sig["score"] = (c - upper) / a
        out[asset] = sig
    return out


# Giorni di storia necessari prima che i segnali siano validi (warm-up).
WARMUP = {
    "rsi2_mean_reversion": lambda p: max(p.get("trend_sma", 200), 14),
    "donchian_breakout": lambda p: max(p.get("entry_window", 55), p.get("atr_window", 20)),
}
