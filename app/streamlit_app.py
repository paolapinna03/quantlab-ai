"""QuantLab AI: dashboard di backtesting.

Avvio:
    streamlit run app/streamlit_app.py

L'interfaccia non contiene logica quantitativa: chiama solo funzioni di
`quantlab/`. Ogni numero mostrato viene da un backtest eseguito davvero.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import color_map, drawdown_chart, equity_chart, weights_chart
from quantlab.backtest import BacktestConfig, evaluate, run_backtest, warmup_end
from quantlab.config import DEFAULT_COST_BPS, HOLDOUT_START, RESEARCH_END
from quantlab.data import load_prices
from quantlab.metrics import drawdown_series, equity_curve
from quantlab.strategies import buy_and_hold, time_series_momentum

# Universo di ETF liquidi disponibili nella dashboard.
UNIVERSE = {
    "SPY": "Azioni USA large cap",
    "QQQ": "Nasdaq 100",
    "IWM": "Azioni USA small cap",
    "EFA": "Azioni sviluppati ex-USA",
    "EEM": "Azioni emergenti",
    "TLT": "Treasury USA 20+ anni",
    "IEF": "Treasury USA 7-10 anni",
    "GLD": "Oro",
    "DBC": "Materie prime",
    "VNQ": "Immobiliare USA (REIT)",
}
LITERATURE_LOOKBACK = 252
STRATEGIES = ["Buy & hold SPY", "Pesi uguali", "Momentum 12m"]

st.set_page_config(page_title="QuantLab AI", page_icon="📈", layout="wide")


# --- Dati e calcoli (in cache: si ricalcolano solo se cambiano gli input) ------

@st.cache_data(show_spinner="Carico i prezzi…")
def load_closes(tickers: tuple[str, ...]) -> pd.DataFrame:
    # Si caricano solo dati fino a RESEARCH_END: l'holdout non entra mai in memoria.
    return pd.DataFrame({t: load_prices(t, end=RESEARCH_END)["close"] for t in tickers})


@st.cache_data(show_spinner="Eseguo i backtest…")
def run_experiment(tickers: tuple[str, ...], lookback: int, long_only: bool,
                   cost_bps: float, lag: int):
    closes = load_closes(tickers)
    spy = load_closes(("SPY",))
    cfg = BacktestConfig(cost_bps=cost_bps, execution_lag=lag)
    results = {
        STRATEGIES[0]: run_backtest(spy, buy_and_hold(spy), cfg),
        STRATEGIES[1]: run_backtest(closes, buy_and_hold(closes, rebalance="M"), cfg),
        STRATEGIES[2]: run_backtest(
            closes, time_series_momentum(closes, lookback=lookback, long_only=long_only), cfg
        ),
    }
    start = warmup_end(closes, lookback)
    return results, start


# --- Barra laterale ------------------------------------------------------------

with st.sidebar:
    st.markdown("## QuantLab AI")
    st.caption("Backtesting · Fase 2")

    tickers = st.multiselect(
        "Universo",
        options=list(UNIVERSE),
        default=["SPY", "QQQ", "TLT", "GLD"],
        format_func=lambda t: f"{t} · {UNIVERSE[t]}",
        max_selections=8,
    )
    st.markdown("**Momentum**")
    lookback = st.select_slider(
        "Finestra del segnale",
        options=[63, 126, 189, 252, 378],
        value=LITERATURE_LOOKBACK,
        format_func=lambda d: f"{d // 21} mesi",
    )
    long_only = st.toggle("Solo lungo (altrimenti lungo/corto)", value=True)

    st.markdown("**Esecuzione**")
    cost_bps = st.number_input("Costi per lato (bp)", 0.0, 100.0, DEFAULT_COST_BPS, 1.0)
    lag = st.radio("Esecuzione", [1, 0, 2], horizontal=True,
                   format_func=lambda x: {0: "stessa chiusura", 1: "+1 giorno", 2: "+2 giorni"}[x])
    log_scale = st.toggle("Scala logaritmica", value=True)

    st.divider()
    st.caption(f"🔒 Holdout dal {HOLDOUT_START}: escluso da tutte le analisi.")

if not tickers:
    st.info("Scegli almeno un ETF nella barra laterale.")
    st.stop()

try:
    results, start = run_experiment(tuple(tickers), lookback, long_only, cost_bps, lag)
except Exception as exc:  # errori dei dati o del motore: mostrali, non nasconderli
    st.error(f"Impossibile eseguire il backtest: {exc}")
    st.stop()

end = pd.Timestamp(RESEARCH_END)
returns = {name: r.returns.loc[start:end] for name, r in results.items()}
stats = pd.DataFrame({name: evaluate(r, start, end) for name, r in results.items()}).T
colors = color_map(STRATEGIES)


# --- Intestazione ------------------------------------------------------------

st.markdown("### Momentum contro buy-and-hold")
st.caption(
    f"Finestra di valutazione {start.date()} → {end.date()} · universo {', '.join(tickers)} · "
    f"costi {cost_bps:g} bp per lato · stessa finestra per tutte le strategie"
)

if lookback != LITERATURE_LOOKBACK or not long_only:
    st.warning(
        "Hai cambiato i parametri rispetto alla configurazione di riferimento. "
        "Ogni variante provata dopo aver visto i risultati aumenta il rischio di "
        "overfitting: considera questi numeri esplorativi, non una scoperta.",
        icon="⚠️",
    )

mom, bench = stats.loc[STRATEGIES[2]], stats.loc[STRATEGIES[0]]
k1, k2, k3, k4 = st.columns(4)
def pp(x: float) -> str:
    """Differenza tra due percentuali, in punti percentuali."""
    return f"{x * 100:+.1f} pp vs SPY"


k1.metric("CAGR momentum", f"{mom['cagr']:.1%}", pp(mom["cagr"] - bench["cagr"]))
k2.metric(f"Sharpe momentum (± {mom['sharpe_se']:.2f})", f"{mom['sharpe']:.2f}",
          f"{mom['sharpe'] - bench['sharpe']:+.2f} vs SPY",
          help="± = errore standard (Lo, 2002). Differenze inferiori a ~2 SE "
               "non sono statisticamente distinguibili.")
k3.metric("Max drawdown", f"{mom['max_drawdown']:.1%}",
          pp(mom["max_drawdown"] - bench["max_drawdown"]))
k4.metric("Volatilità", f"{mom['volatility']:.1%}",
          pp(mom["volatility"] - bench["volatility"]), delta_color="inverse")


# --- Schede -------------------------------------------------------------------

tab_perf, tab_table, tab_weights, tab_trades, tab_method = st.tabs(
    ["Performance", "Metriche", "Pesi", "Operazioni", "Metodologia"]
)

with tab_perf:
    st.markdown("**Valore di 100 investito**")
    curves = {n: equity_curve(r, start_value=100) for n, r in returns.items()}
    st.plotly_chart(equity_chart(curves, colors, log_scale), use_container_width=True)
    st.markdown("**Drawdown** · distanza dal massimo precedente")
    dds = {n: drawdown_series(r) for n, r in returns.items()}
    st.plotly_chart(drawdown_chart(dds, colors), use_container_width=True)

with tab_table:
    table = pd.DataFrame({
        "CAGR": stats["cagr"].map("{:.1%}".format),
        "Volatilità": stats["volatility"].map("{:.1%}".format),
        "Sharpe": stats["sharpe"].map("{:.2f}".format),
        "± SE": stats["sharpe_se"].map("{:.2f}".format),
        "Sortino": stats["sortino"].map("{:.2f}".format),
        "Max DD": stats["max_drawdown"].map("{:.1%}".format),
        "Durata max DD (gg)": stats["max_dd_duration"].astype(int),
        "Calmar": stats["calmar"].map("{:.2f}".format),
        "Turnover/anno": stats["turnover_yr"].map("{:.2f}".format),
        "Costi/anno": stats["cost_yr"].map("{:.2%}".format),
        "Investito": stats["invested"].map("{:.0%}".format),
    })
    st.dataframe(table, use_container_width=True)
    st.caption(
        "SE = errore standard dello Sharpe (Lo, 2002). Differenze di Sharpe più piccole "
        "di circa 2 × SE non sono statisticamente distinguibili. Sharpe calcolato con "
        "tasso privo di rischio a zero."
    )

with tab_weights:
    # Colori nell'ordine validato della palette (coppie adiacenti distinguibili
    # anche per chi è daltonico), assegnati agli ETF selezionati.
    asset_colors = color_map(tickers)
    w = results[STRATEGIES[2]].weights.loc[start:end]
    st.markdown("**Pesi del momentum** · lo spazio vuoto sopra le aree è liquidità")
    st.plotly_chart(weights_chart(w, asset_colors), use_container_width=True)

with tab_trades:
    trades = results[STRATEGIES[2]].trades
    trades = trades[(trades["date"] >= start) & (trades["date"] <= end)]
    st.markdown(f"**Operazioni del momentum** · {len(trades)} variazioni di peso")
    st.dataframe(
        trades.sort_values("date", ascending=False).style.format({
            "date": lambda d: d.date(), "weight_before": "{:.1%}",
            "weight_after": "{:.1%}", "change": "{:+.1%}",
        }),
        use_container_width=True, hide_index=True,
    )

with tab_method:
    execution_text = {0: "alla stessa chiusura", 1: "alla chiusura del giorno dopo"}.get(
        lag, f"{lag} giorni dopo"
    )
    st.markdown(f"""
**Strategia.** Time-series momentum semplificato da Moskowitz, Ooi & Pedersen (2012),
*Time Series Momentum*, Journal of Financial Economics. Ogni ETF riceve 1/N del
capitale: lo si detiene se il suo rendimento total-return degli ultimi
{lookback // 21} mesi è positivo, altrimenti quella quota resta in liquidità
{"" if long_only else "(o va corta)"}. Ribilanciamento all'ultimo giorno di borsa del mese.

**Benchmark.** Buy-and-hold di SPY e portafoglio a pesi uguali ribilanciato
mensilmente sugli stessi ETF della strategia.

**Ipotesi del backtest.**
- Prezzi di chiusura aggiustati per dividendi e frazionamenti (Yahoo Finance, fonte non ufficiale).
- Segnale calcolato alla chiusura, eseguito {execution_text}.
- Costi: {cost_bps:g} bp per ogni unità di capitale scambiata (commissioni, spread, slippage).
- La liquidità rende 0%: penalizza leggermente il momentum, che resta spesso in liquidità.
- Nessuna leva, nessun costo di prestito titoli.

**Limiti.** Risultati storici e in-sample: i parametri vengono dalla letteratura, ma il
periodo analizzato si sovrappone in parte a quello in cui la strategia è stata
pubblicata. Universo scelto con il senno di poi (ETF che esistono ancora oggi).
I dati dal {HOLDOUT_START} sono riservati alla valutazione finale.
""")
