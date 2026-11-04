"""Strategie disponibili nella pagina Backtesting.

Ogni voce descrive: famiglia nel registro (per contare i tentativi), ipotesi,
configurazione di riferimento (letteratura o pre-registrazione), controlli
della barra laterale e testo della metodologia. La pagina è generica: per
aggiungere una strategia all'interfaccia basta aggiungere una voce qui.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

import streamlit as st

from quantlab.paper.presets import PRESETS as PAPER_PRESETS

UNIVERSE = {
    "SPY": "Azioni USA large cap", "QQQ": "Nasdaq 100", "IWM": "Azioni USA small cap",
    "EFA": "Azioni sviluppati ex-USA", "EEM": "Azioni emergenti",
    "TLT": "Treasury USA 20+ anni", "IEF": "Treasury USA 7-10 anni", "GLD": "Oro",
    "DBC": "Materie prime", "VNQ": "Immobiliare USA (REIT)",
    "XLE": "Settore energia", "XLF": "Settore finanziario", "XLK": "Settore tecnologia",
    "XLV": "Settore salute", "XLY": "Consumi discrezionali", "XLP": "Beni di prima necessità",
    "XLI": "Settore industriale", "XLU": "Utility",
}
SECTOR_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD",
                   "XLE", "XLF", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU"]


@dataclass(frozen=True)
class Preset:
    key: str
    label: str                      # nel menu
    short: str                      # nelle tabelle e nei grafici
    family: str
    strategy: str
    hypothesis: str
    default_universe: list[str]
    controls: Callable[[], tuple[dict, dict]]   # disegna i controlli → (params, engine)
    reference: list[dict]           # configurazioni dichiarate in anticipo (params)
    method_md: Callable[[dict, dict], str]
    grid: Callable[[dict], list[dict]] | None = None   # walk-forward (solo portafoglio)
    swing: bool = False
    extra: dict = field(default_factory=dict)


# --- Momentum ------------------------------------------------------------------------

def _momentum_controls():
    st.html('<div class="ql-section">Segnale momentum</div>')
    lb = st.select_slider("Finestra", [63, 126, 189, 252, 378], value=252,
                          format_func=lambda d: f"{d // 21} mesi", key="mom_lb")
    side = st.segmented_control("Posizioni", ["Lungo / liquidità", "Lungo / corto"],
                                default="Lungo / liquidità", key="mom_side")
    return {"lookback": lb, "long_only": side != "Lungo / corto"}, {}


def _momentum_md(p, e):
    return f"""
**Strategia.** Time-series momentum semplificato da Moskowitz, Ooi & Pedersen (2012), *Time Series
Momentum*, Journal of Financial Economics. Ogni ETF riceve 1/N del capitale: lo si detiene se il suo
rendimento total-return degli ultimi {p['lookback'] // 21} mesi è positivo, altrimenti quella quota
resta in liquidità{"" if p['long_only'] else " (o va corta)"}. Ribilanciamento a fine mese."""


# --- Allocazione per rischio ------------------------------------------------------------

METHOD_LABELS = {"risk_parity": "Risk parity (ERC)", "min_variance": "Minima varianza",
                 "inverse_vol": "Inverse volatility", "hrp": "Hierarchical Risk Parity",
                 "mean_variance": "Media-varianza (didattico)"}


def _alloc_controls(defaults: dict | None = None, prefix: str = "al"):
    """Controlli dell'allocazione; `defaults` e `prefix` permettono due voci di menu indipendenti."""
    d = {"method": "risk_parity", "lookback": 252, "max_weight": 0.4, "momentum_filter": 0} | (defaults or {})
    st.html('<div class="ql-section">Allocazione</div>')
    methods = list(METHOD_LABELS)
    method = st.selectbox("Metodo", methods, index=methods.index(d["method"]),
                          format_func=METHOD_LABELS.get, key=f"{prefix}_m")
    lb = st.select_slider("Finestra di stima del rischio", [126, 252, 504], value=d["lookback"],
                          format_func=lambda x: f"{x // 21} mesi", key=f"{prefix}_lb")
    cap = st.slider("Peso massimo per ETF", 0.2, 1.0, float(d["max_weight"]), 0.05, key=f"{prefix}_cap",
                    help="Usato da minima varianza e media-varianza.")
    mom = st.toggle("Filtro momentum 12 mesi", value=bool(d["momentum_filter"]), key=f"{prefix}_mom",
                    help="Gli ETF con rendimento a 12 mesi negativo vanno in liquidità.")
    return {"method": method, "lookback": lb, "max_weight": cap, "shrinkage": True,
            "momentum_filter": 252 if mom else 0}, {}


def _alloc_md(p, e):
    m = METHOD_LABELS[p["method"]]
    return f"""
**Strategia.** {m}: a ogni fine mese si stima la covarianza degli ultimi {p['lookback'] // 21} mesi
(con shrinkage di Ledoit-Wolf) e si calcolano i pesi in base al rischio, senza usare previsioni di
rendimento{" (tranne la media-varianza, inclusa come confronto)" if p['method'] == 'mean_variance' else ""}.
{"Filtro momentum: gli ETF con rendimento negativo a 12 mesi vanno in liquidità. " if p['momentum_filter'] else ""}
Fonti: Maillard, Roncalli & Teïletche (2010); López de Prado (2016); Ledoit & Wolf (2004);
DeMiguel, Garlappi & Uppal (2009) per il confronto con 1/N.
Pre-registrazione: `docs/preregistrazioni/allocazione_etf.md`."""


_ALLOC_REF = [{"method": m, "lookback": 252, "max_weight": 0.4, "shrinkage": True, "momentum_filter": f}
              for m, f in [("inverse_vol", 0), ("min_variance", 0), ("risk_parity", 0), ("hrp", 0),
                           ("risk_parity", 252), ("mean_variance", 0)]]


# --- Trend su più orizzonti / volatility targeting ---------------------------------

_MULTI = (21, 63, 126, 252)
_HORIZONS = {"Solo 12 mesi": (252,), "1-3-6-12 mesi": _MULTI, "3-6-12 mesi": (63, 126, 252),
             "6-12 mesi": (126, 252)}


def _trend_controls():
    st.html('<div class="ql-section">Segnali di trend</div>')
    h = st.selectbox("Orizzonti", list(_HORIZONS), index=1, key="tr_h")
    vt = st.select_slider("Volatility target", [0.0, 0.06, 0.08, 0.10], value=0.0, key="tr_v",
                          format_func=lambda v: "nessuno" if v == 0 else f"{v:.0%} annuo")
    params = {"horizons": _HORIZONS[h]}
    if vt:
        params |= {"vol_target": vt, "vol_lookback": 63}
    return params, {}


def _trend_md(p, e):
    hs = ", ".join(f"{h // 21}" for h in p["horizons"])
    vt = p.get("vol_target", 0)
    return f"""
**Strategia.** Per ogni ETF si guarda il segno del rendimento passato a {hs} mesi e si fa la media:
l'ETF riceve quella frazione della sua quota 1/N (per esempio 3 segnali positivi su 4 → 75%).
Il resto va in liquidità. {f"Se la volatilità stimata del portafoglio (ultimi 63 giorni) supera il "
f"{vt:.0%} annuo, tutte le posizioni vengono ridotte in proporzione, mai con leva. " if vt else ""}
Ribilanciamento mensile, esecuzione il giorno dopo.
Fonti: Hurst, Ooi & Pedersen (2017); Moreira & Muir (2017).
Pre-registrazione ed esito: `docs/preregistrazioni/trend_vol_etf.md`."""


_TREND_REF = [{"horizons": (252,), "vol_target": 0.08, "vol_lookback": 63}, {"horizons": _MULTI},
              {"horizons": _MULTI, "vol_target": 0.08, "vol_lookback": 63}]


def _trend_grid(p):
    if p.get("vol_target"):
        return [p | {"vol_target": t} for t in (0.06, 0.08, 0.10)]
    return [p | {"horizons": h} for h in (_MULTI, (63, 126, 252), (126, 252))]


# --- Swing -------------------------------------------------------------------------

def _swing_engine():
    st.html('<div class="ql-section">Portafoglio swing</div>')
    n = st.slider("Posizioni contemporanee massime", 1, 10, 5, key="sw_n")
    return {"max_positions": n, "position_fraction": round(1 / n, 4)}


def _rsi_controls():
    st.html('<div class="ql-section">Segnale RSI(2)</div>')
    entry = st.select_slider("Ingresso se RSI(2) sotto", [5.0, 10.0, 20.0], value=10.0, key="rsi_e")
    trend = st.select_slider("Filtro di trend (media mobile)", [100, 200], value=200,
                             format_func=lambda d: f"{d} giorni", key="rsi_t")
    return {"entry_rsi": entry, "trend_sma": trend}, _swing_engine()


def _rsi_md(p, e):
    return f"""
**Strategia.** Compra le discese brevi dentro un trend rialzista: ingresso quando la chiusura è sopra
la media a {p['trend_sma']} giorni e l'RSI a 2 giorni è sotto {p['entry_rsi']:g}; uscita quando la
chiusura supera la media a 5 giorni. Al massimo {e['max_positions']} posizioni da
{e['position_fraction']:.0%} del capitale. Regola divulgata da Connors & Alvarez (2009), libro non
peer-reviewed; il fenomeno di inversione di breve periodo è documentato da Jegadeesh (1990) e
Lehmann (1990). Ordini all'apertura del giorno dopo, stop e target controllati durante la giornata."""


def _donchian_controls():
    st.html('<div class="ql-section">Segnale breakout</div>')
    entry = st.select_slider("Ingresso: massimo dei giorni precedenti", [20, 55, 100], value=55, key="dc_e")
    exit_ = st.select_slider("Uscita: minimo dei giorni precedenti", [10, 20, 50], value=20, key="dc_x")
    stop = st.select_slider("Stop iniziale (multipli di ATR)", [1.5, 2.0, 3.0], value=2.0, key="dc_s")
    return {"entry_window": entry, "exit_window": exit_, "stop_atr": stop}, _swing_engine()


def _donchian_md(p, e):
    return f"""
**Strategia.** Compra quando la chiusura supera il massimo dei {p['entry_window']} giorni precedenti,
esce quando scende sotto il minimo dei {p['exit_window']} giorni precedenti; stop iniziale a
{p['stop_atr']:g} ATR. Al massimo {e['max_positions']} posizioni da {e['position_fraction']:.0%}.
Regole dei "Turtle Traders" descritte da Faith (2007), libro non peer-reviewed; il trend following è
documentato da Hurst, Ooi & Pedersen (2017)."""


_PAPER = PAPER_PRESETS["rp_momentum"]["spec"]

PRESETS = {
    "paper": Preset(
        key="paper", label="★ Risk parity + momentum (in paper trading)", short="RP + momentum",
        family="allocazione_etf", strategy=_PAPER["strategy"],
        hypothesis="Un'allocazione basata sul rischio migliora il rapporto rischio/rendimento "
                   "rispetto ai pesi uguali, fuori campione e al netto dei costi.",
        default_universe=list(_PAPER["universe"]),
        controls=lambda: _alloc_controls(_PAPER["params"], prefix="pp"),
        reference=_ALLOC_REF, method_md=_alloc_md,
        grid=lambda p: [p | {"lookback": lb} for lb in (126, 252, 504)],
    ),
    "momentum": Preset(
        key="momentum", label="Momentum 12 mesi (time-series)", short="Momentum",
        family="tsmom_etf", strategy="time_series_momentum",
        hypothesis="Il rendimento total-return degli ultimi 12 mesi di un ETF liquido predice "
                   "positivamente il suo rendimento del mese successivo (Moskowitz, Ooi & Pedersen, 2012).",
        default_universe=["SPY", "QQQ", "TLT", "GLD"], controls=_momentum_controls,
        reference=[{"lookback": 252, "long_only": True}], method_md=_momentum_md,
        grid=lambda p: [p | {"lookback": lb} for lb in (63, 126, 189, 252, 378)],
    ),
    "allocation": Preset(
        key="allocation", label="Allocazione per rischio", short="Allocazione",
        family="allocazione_etf", strategy="risk_allocation",
        hypothesis="Un'allocazione basata sul rischio migliora il rapporto rischio/rendimento "
                   "rispetto ai pesi uguali, fuori campione e al netto dei costi.",
        default_universe=["SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
        controls=_alloc_controls, reference=_ALLOC_REF, method_md=_alloc_md,
        grid=lambda p: [p | {"lookback": lb} for lb in (126, 252, 504)],
    ),
    "trend": Preset(
        key="trend", label="Trend su più orizzonti / volatility targeting", short="Trend multi",
        family="trend_vol_etf", strategy="trend_signals",
        hypothesis="Trend su più orizzonti e volatility targeting migliorano il controllo del rischio "
                   "del time-series momentum 12 mesi a parità di rendimento.",
        default_universe=["SPY", "EFA", "EEM", "IWM", "TLT", "IEF", "GLD", "DBC", "VNQ"],
        controls=_trend_controls, reference=_TREND_REF, method_md=_trend_md, grid=_trend_grid,
    ),
    "rsi2": Preset(
        key="rsi2", label="Swing · RSI(2) mean reversion", short="RSI(2)",
        family="swing_rsi2_etf", strategy="rsi2_mean_reversion",
        hypothesis="Dopo forti cali di 1-2 giorni in un trend rialzista, gli ETF liquidi tendono "
                   "a rimbalzare nei giorni successivi (inversione di breve periodo).",
        default_universe=SECTOR_UNIVERSE, controls=_rsi_controls,
        reference=[{"entry_rsi": 10.0, "trend_sma": 200}], method_md=_rsi_md, swing=True,
    ),
    "donchian": Preset(
        key="donchian", label="Swing · Donchian breakout", short="Breakout",
        family="swing_donchian_etf", strategy="donchian_breakout",
        hypothesis="Le rotture dei massimi a 55 giorni segnalano trend che tendono a proseguire "
                   "(trend following su orizzonte di settimane).",
        default_universe=SECTOR_UNIVERSE, controls=_donchian_controls,
        reference=[{"entry_window": 55, "exit_window": 20, "stop_atr": 2.0}],
        method_md=_donchian_md, swing=True,
    ),
}
