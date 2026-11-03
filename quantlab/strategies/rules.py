"""Strategia swing componibile: ingresso + filtro + uscita + stop/target.

È il "mattoncino" usato dalla Strategy Discovery. Ogni combinazione di
parametri descrive una strategia completa, salvabile in JSON e quindi
registrabile, riproducibile e contabile come tentativo.

Ingressi disponibili (e loro uscita "opposta"):

| entry      | Ingresso                                       | Uscita opposta                   | Fondamento |
|------------|------------------------------------------------|----------------------------------|------------|
| rsi_dip    | RSI(`lookback`) < `threshold`                 | chiusura > media a 5 giorni      | inversione di breve (Jegadeesh 1990; Lehmann 1990) |
| breakout   | chiusura > massimo dei `lookback` gg prec.    | chiusura < minimo di lookback/2  | trend following (Hurst, Ooi & Pedersen 2017) |
| ma_cross   | media `lookback` incrocia al rialzo la 4×     | la media veloce scende sotto     | trend following, regola divulgativa |
| momentum   | rendimento a `lookback` gg > `threshold`      | rendimento a `lookback` gg < 0   | time-series momentum (Moskowitz, Ooi & Pedersen 2012) |
| high_52w   | chiusura entro `threshold` dal massimo a 1 anno | chiusura < media a 50 giorni    | 52-week high (George & Hwang 2004, Journal of Finance) |

Attenzione: i fondamenti accademici riguardano il fenomeno (spesso su
azioni singole o futures), non questa precisa regola su ETF. Proprio per
questo ogni combinazione passa dal laboratorio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.indicators import atr, donchian_high_prior, donchian_low_prior, rsi, sma

ENTRIES = ("rsi_dip", "breakout", "ma_cross", "momentum", "high_52w")
EXITS = ("opposite", "none")


def rule_swing(
    data: dict[str, pd.DataFrame],
    entry: str = "breakout",
    lookback: int = 55,
    threshold: float = 0.0,
    trend_filter: int = 0,
    exit_rule: str = "opposite",
    stop_atr: float = 0.0,
    target_atr: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Segnali per ogni asset. `trend_filter` > 0: entra solo sopra la media a N giorni.

    `stop_atr` / `target_atr` = 0 significa nessuno stop / nessun target.
    """
    if entry not in ENTRIES:
        raise ValueError(f"entry sconosciuto: {entry!r}. Disponibili: {ENTRIES}")
    if exit_rule not in EXITS:
        raise ValueError(f"exit_rule sconosciuta: {exit_rule!r}. Disponibili: {EXITS}")
    if lookback < 2:
        raise ValueError("lookback deve essere almeno 2")

    out = {}
    for asset, df in data.items():
        c, h, l = df["close"], df["high"], df["low"]
        a = atr(h, l, c, 14)
        if entry == "rsi_dip":
            r = rsi(c, lookback)
            enter, leave, score = r < threshold, c > sma(c, 5), -r
        elif entry == "breakout":
            upper = donchian_high_prior(h, lookback)
            enter = c > upper
            leave = c < donchian_low_prior(l, max(lookback // 2, 2))
            score = (c - upper) / a
        elif entry == "ma_cross":
            fast, slow = sma(c, lookback), sma(c, 4 * lookback)
            above = fast > slow
            enter = above & ~above.shift(1, fill_value=False)
            leave, score = ~above, fast / slow - 1
        elif entry == "momentum":
            ret = c / c.shift(lookback) - 1
            enter, leave, score = ret > threshold, ret < 0, ret
        else:  # high_52w
            high_252 = donchian_high_prior(h, 252)
            enter = c >= (1 - threshold) * high_252
            leave, score = c < sma(c, 50), c / high_252

        if trend_filter > 0:
            enter = enter & (c > sma(c, trend_filter))

        sig = pd.DataFrame(index=df.index)
        sig["entry"] = enter.fillna(False).astype(bool)
        sig["exit"] = leave.fillna(False).astype(bool) if exit_rule == "opposite" else False
        sig["stop_distance"] = stop_atr * a if stop_atr > 0 else np.nan
        sig["target_distance"] = target_atr * a if target_atr > 0 else np.nan
        sig["score"] = score
        out[asset] = sig
    return out


def rule_swing_warmup(p: dict) -> int:
    lb = int(p.get("lookback", 55))
    need = {"rsi_dip": lb + 14, "breakout": lb, "ma_cross": 4 * lb,
            "momentum": lb, "high_52w": 252}[p.get("entry", "breakout")]
    return max(need, int(p.get("trend_filter", 0)), 14)


def describe(params: dict, engine: dict | None = None) -> str:
    """Descrizione leggibile di una combinazione, per tabelle e rapporti."""
    p = params
    e = p.get("entry", "breakout")
    lb, th = p.get("lookback", 55), p.get("threshold", 0.0)
    text = {
        "rsi_dip": f"RSI({lb}) < {th:g}",
        "breakout": f"Breakout {lb}g",
        "ma_cross": f"Incrocio medie {lb}/{4 * lb}",
        "momentum": f"Momentum {lb}g > {th:.0%}",
        "high_52w": f"Entro {th:.0%} dal max 52 sett.",
    }[e]
    parts = [text]
    if p.get("trend_filter"):
        parts.append(f"filtro SMA{p['trend_filter']}")
    parts.append("uscita opposta" if p.get("exit_rule", "opposite") == "opposite" else "senza uscita di segnale")
    if p.get("stop_atr"):
        parts.append(f"stop {p['stop_atr']:g} ATR")
    if p.get("target_atr"):
        parts.append(f"target {p['target_atr']:g} ATR")
    if engine and engine.get("max_holding_days"):
        parts.append(f"max {engine['max_holding_days']}g")
    return " · ".join(parts)
