"""Pagina Backtesting: momentum contro buy-and-hold, con validazione.

L'interfaccia non contiene logica quantitativa: chiama solo funzioni di
`quantlab/`. Ogni numero mostrato viene da un backtest eseguito davvero.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import (
    PLOTLY_CONFIG, color_map, drawdown_chart, equity_chart, walk_forward_chart, weights_chart,
)
from styles import card_title, criteria_table, esc, header, section, verdict_block
from quantlab.config import DEFAULT_COST_BPS, HOLDOUT_START, RESEARCH_END
from quantlab.config import DEFAULT_CASH
from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.metrics import drawdown_series, equity_curve
from quantlab.validation import Criteria, validate_candidate

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
FAMILY = "tsmom_etf"
HYPOTHESIS = (
    "Il rendimento total-return degli ultimi 12 mesi di un ETF liquido predice "
    "positivamente il suo rendimento del mese successivo (Moskowitz, Ooi & Pedersen, 2012)."
)
BENCH, EQUAL, MOM = "SPY buy & hold", "Pesi uguali", "Momentum"
STRATEGIES = [BENCH, EQUAL, MOM]



# --- Dati e calcoli (in cache: si ricalcolano solo se cambiano gli input) ------

@st.cache_data(show_spinner="Eseguo i backtest…")
def run_all(tickers: tuple[str, ...], lookback: int, long_only: bool, cost_bps: float, lag: int):
    """Momentum (registrato come tentativo nella famiglia) + benchmark sulla stessa finestra.

    La cache fa sì che ogni configurazione venga registrata una volta per sessione;
    il conteggio dei tentativi usa comunque le configurazioni distinte.
    """
    common = dict(cost_bps=cost_bps, execution_lag=lag)
    mom = run_experiment(momentum_spec(tickers, lookback, long_only, cost_bps, lag),
                         ExperimentRegistry())
    start = str(mom.eval_start.date())
    bench = run_experiment(ExperimentSpec(
        cash=DEFAULT_CASH,
        family="benchmark", strategy="buy_and_hold", universe=("SPY",), eval_start=start, **common))
    equal = run_experiment(equal_weight_spec(tickers, cost_bps, lag, eval_start=start))
    return {BENCH: bench, EQUAL: equal, MOM: mom}


@st.cache_data(show_spinner="Verifico i criteri di promozione…")
def run_validation(tickers: tuple[str, ...], lookback: int, long_only: bool, cost_bps: float,
                   lag: int, trials: int):
    """Criteri di promozione. `trials` è nella chiave della cache: quando provi nuove
    configurazioni la soglia del Deflated Sharpe sale e la validazione va rifatta."""
    grid = [{"lookback": lb, "long_only": long_only} for lb in (63, 126, 189, 252, 378)]
    return validate_candidate(
        momentum_spec(tickers, lookback, long_only, cost_bps, lag),
        equal_weight_spec(tickers, cost_bps, lag),
        ExperimentRegistry(), grid, Criteria(),
    )


def momentum_spec(tickers, lookback, long_only, cost_bps, lag) -> ExperimentSpec:
    return ExperimentSpec(cash=DEFAULT_CASH, family=FAMILY, strategy="time_series_momentum",
                          params={"lookback": lookback, "long_only": long_only},
                          universe=tickers, hypothesis=HYPOTHESIS,
                          cost_bps=cost_bps, execution_lag=lag)


def equal_weight_spec(tickers, cost_bps, lag, eval_start=None) -> ExperimentSpec:
    return ExperimentSpec(cash=DEFAULT_CASH, family="benchmark", strategy="buy_and_hold", params={"rebalance": "M"},
                          universe=tickers, eval_start=eval_start,
                          cost_bps=cost_bps, execution_lag=lag)


# --- Pannello parametri ----------------------------------------------------------

with st.sidebar:
    st.html(section("Universo"))
    tickers = st.multiselect(
        "ETF", options=list(UNIVERSE), default=["SPY", "QQQ", "TLT", "GLD"],
        format_func=lambda t: f"{t} · {UNIVERSE[t]}", max_selections=8,
        label_visibility="collapsed",
    )

    st.html(section("Segnale momentum"))
    lookback = st.select_slider(
        "Finestra", options=[63, 126, 189, 252, 378], value=LITERATURE_LOOKBACK,
        format_func=lambda d: f"{d // 21} mesi",
    )
    direction = st.segmented_control(
        "Posizioni", ["Lungo / liquidità", "Lungo / corto"], default="Lungo / liquidità",
    )
    long_only = direction != "Lungo / corto"

    st.html(section("Esecuzione"))
    cost_bps = st.number_input("Costi per lato (bp)", 0.0, 100.0, DEFAULT_COST_BPS, 1.0)
    lag = st.segmented_control(
        "Esecuzione dell'ordine", [0, 1, 2], default=1,
        format_func=lambda x: {0: "Stessa chiusura", 1: "Giorno dopo", 2: "2 giorni dopo"}[x],
        help="Quando viene eseguito l'ordine rispetto alla chiusura su cui si calcola il segnale. "
             "\"Giorno dopo\" è l'ipotesi realistica; \"stessa chiusura\" è ottimistica.",
    )
    lag = 1 if lag is None else lag

    st.html(section("Visualizzazione"))
    log_scale = st.toggle("Scala logaritmica", value=True)

if not tickers:
    st.info("Scegli almeno un ETF nel pannello a sinistra.")
    st.stop()

try:
    outcomes = run_all(tuple(tickers), lookback, long_only, cost_bps, lag)
except Exception as exc:  # errori dei dati o del motore: mostrali, non nasconderli
    st.error(f"Impossibile eseguire il backtest: {exc}")
    st.stop()

results = {name: o.result for name, o in outcomes.items()}
start, end = outcomes[MOM].eval_start, outcomes[MOM].eval_end
returns = {name: r.returns.loc[start:end] for name, r in results.items()}
stats = pd.DataFrame({name: o.metrics for name, o in outcomes.items()}).T
trials = outcomes[MOM].trials_in_family
try:
    report = run_validation(tuple(tickers), lookback, long_only, cost_bps, lag, trials)
except Exception as exc:  # la validazione non deve nascondere i risultati del backtest
    report = None
    validation_error = str(exc)
colors = color_map(STRATEGIES)
modified = lookback != LITERATURE_LOOKBACK or not long_only


# --- Intestazione ------------------------------------------------------------

badges = [
    (f"{trials} configurazion{'e' if trials == 1 else 'i'} provat{'a' if trials == 1 else 'e'}",
     "warn" if trials > 1 else "",
     f"Varianti di questa ipotesi (famiglia {FAMILY}) già testate e registrate. Più sono, "
     "più alta diventa la soglia del Deflated Sharpe."),
    (f"Dati {HOLDOUT_START[:4]}+ riservati", "lock",
     f"I dati dal {HOLDOUT_START} non vengono usati qui: servono per la verifica finale."),
    ("Periodo di ricerca", "",
     "Risultati sul periodo usato per studiare la strategia (in-sample): "
     "tendono a essere più ottimistici del futuro."),
]
if modified:
    badges.insert(0, ("Parametri modificati", "warn",
                      "Diversi da quelli della letteratura: risultati da considerare esplorativi."))
if report is not None:
    badges.insert(0, (f"{'✓' if report.promoted else '✗'} {report.verdict}",
                      "pass" if report.promoted else "fail",
                      "Esito dei criteri di promozione: vedi la sezione più in basso."))
st.html(header(
    eyebrow="Backtesting · Esperimento 01",
    title="Time-series momentum contro buy-and-hold",
    meta=(f"<b>{start:%d %b %Y} → {end:%d %b %Y}</b> · {esc(', '.join(tickers))} · "
          f"costi <b>{cost_bps:g} bp</b> per lato · esecuzione <b>"
          f"{ {0: 'stessa chiusura', 1: 'giorno dopo', 2: '2 giorni dopo'}[lag] }</b>"),
    badges=badges,
))

if modified:
    st.warning(
        "Parametri diversi dalla configurazione della letteratura (12 mesi, lungo/liquidità). "
        "Ogni variante provata dopo aver visto i risultati aumenta il rischio di overfitting: "
        "considera questi numeri esplorativi.",
        icon=":material/warning:",
    )


# --- KPI ---------------------------------------------------------------------

with st.expander("Come leggere questa pagina"):
    st.markdown(f"""
1. **In alto** il verdetto e i numeri chiave della strategia momentum. Le frecce colorate
   confrontano con il **portafoglio a pesi uguali** sugli stessi ETF: è lo stesso benchmark
   usato dai criteri di promozione, così si isola l'effetto del segnale da quello della diversificazione.
2. **Grafici**: andamento di 100 € investiti, confronto tra strategie, quanto capitale il momentum
   tiene investito in ogni ETF.
3. **Criteri di promozione**: sei controlli statistici fissati prima di vedere i risultati.
   Basta un fallimento per respingere la strategia.
4. **Barra laterale**: cambiare i parametri è esplorazione, non scoperta. Ogni configurazione
   che guardi viene registrata e alza l'asticella per tutte le altre.

Tutti i numeri sono calcolati davvero, su dati dal 2005 al 2022. Il periodo dal {HOLDOUT_START[:4]}
è escluso: servirà per la verifica finale.
""")


def pp(x: float) -> str:
    return f"{x * 100:+.1f} pp vs pesi uguali"


mom, bench = stats.loc[MOM], stats.loc[EQUAL]
k = st.columns(5)
k[0].metric("CAGR", f"{mom['cagr']:.2%}", pp(mom["cagr"] - bench["cagr"]), border=True,
            help="Rendimento medio annuo composto, al netto dei costi.")
k[1].metric(f"Sharpe ± {mom['sharpe_se']:.2f}", f"{mom['sharpe']:.2f}",
            f"{mom['sharpe'] - bench['sharpe']:+.2f} vs pesi uguali", border=True,
            help="Rendimento per unità di rischio (volatilità), annualizzato. Sopra 0,5 è buono "
                 "per una strategia diversificata. ± = errore standard (Lo, 2002): differenze "
                 "inferiori a circa 2 errori standard non sono statisticamente distinguibili.")
k[2].metric("Max drawdown", f"{mom['max_drawdown']:.1%}",
            pp(mom["max_drawdown"] - bench["max_drawdown"]), border=True,
            help="Perdita massima dal picco più alto raggiunto in precedenza.")
k[3].metric("Volatilità", f"{mom['volatility']:.1%}",
            pp(mom["volatility"] - bench["volatility"]), delta_color="inverse", border=True,
            help="Quanto oscillano i rendimenti, annualizzato. Più bassa = percorso più tranquillo.")
k[4].metric("Turnover / anno", f"{mom['turnover_yr']:.2f}×",
            f"costi {mom['cost_yr']:.2%} / anno", delta_color="off", border=True,
            help="Quanto capitale viene scambiato in un anno: 1× = l'intero portafoglio. "
                 "Più turnover significa più costi e, in Italia, più tasse.")


# --- Equity + classifica -------------------------------------------------------

left, right = st.columns([2.3, 1], gap="medium")

with left, st.container(border=True):
    st.html(card_title("Valore di 100 investito", "scala log" if log_scale else "scala lineare"))
    curves = {n: equity_curve(r, start_value=100) for n, r in returns.items()}
    st.plotly_chart(equity_chart(curves, colors, log_scale), config=PLOTLY_CONFIG, width="stretch")

with right, st.container(border=True):
    st.html(card_title("Confronto strategie", "ordinate per Sharpe"))
    rows = "".join(
        f"""<tr>
          <td><span class="ql-dot" style="background:{colors[name]}"></span>{esc(name)}</td>
          <td>{s['sharpe']:.2f} <span class="se">±{s['sharpe_se']:.2f}</span></td>
          <td>{s['cagr']:.1%}</td>
          <td>{s['max_drawdown']:.1%}</td>
        </tr>"""
        for name, s in stats.sort_values("sharpe", ascending=False).iterrows()
    )
    st.html(f"""
      <table class="ql-table">
        <thead><tr><th>Strategia</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <div class="ql-note">Stessa finestra per tutte le strategie, al netto dei costi.
      Differenze di Sharpe inferiori a ~2 errori standard non sono significative.
      Sharpe calcolato sul rendimento in eccesso rispetto ai T-bill USA a 3 mesi.</div>""")

    st.html(card_title("Esposizione momentum", "vuoto = liquidità"))
    w = results[MOM].weights.loc[start:end]
    st.plotly_chart(weights_chart(w, color_map(tickers), height=215),
                    config=PLOTLY_CONFIG, width="stretch")


# --- Validazione --------------------------------------------------------------

v_left, v_right = st.columns([2.3, 1], gap="medium")
if report is None:
    st.error(f"Validazione non disponibile: {validation_error}")
else:
    passed = sum(c.passed for c in report.checks)
    with v_left, st.container(border=True):
        st.html(card_title("Criteri di promozione",
                           "fissati prima di vedere i risultati · benchmark: pesi uguali"))
        st.html(criteria_table(report.checks))
    with v_right, st.container(border=True):
        st.html(card_title("Verdetto", f"{passed}/{len(report.checks)} criteri superati"))
        failed = [c.name for c in report.checks if not c.passed]
        st.html(verdict_block(
            report.promoted,
            "Supera tutti i criteri: può passare alla valutazione sull'holdout."
            if report.promoted else
            f"Non supera: {', '.join(failed)}. Le soglie non si abbassano dopo aver visto "
            "l'esito: si torna alle ipotesi.",
        ))
        wf = report.details.get("walk_forward")
        if wf is not None:
            st.html(card_title("Walk-forward", "Sharpe per anno di test"))
            st.plotly_chart(walk_forward_chart(wf.folds), config=PLOTLY_CONFIG, width="stretch")
            st.html('<div class="ql-note">Ogni anno il lookback è scelto usando solo gli anni '
                    'precedenti (passa sopra una barra per vederlo).</div>')


# --- Drawdown -----------------------------------------------------------------

with st.container(border=True):
    st.html(card_title("Drawdown", "distanza dal massimo precedente"))
    dds = {n: drawdown_series(r) for n, r in returns.items()}
    st.plotly_chart(drawdown_chart(dds, colors), config=PLOTLY_CONFIG, width="stretch")


# --- Dettagli -----------------------------------------------------------------

tab_metrics, tab_trades, tab_log, tab_method = st.tabs(
    ["Tutte le metriche", "Operazioni", "Registro esperimenti", "Metodologia"]
)

with tab_metrics:
    PCT_COLS = ["cagr", "volatility", "max_drawdown", "cost_yr", "invested"]
    table = stats[["cagr", "volatility", "sharpe", "sharpe_se", "sortino", "max_drawdown",
                   "max_dd_duration", "calmar", "turnover_yr", "cost_yr", "invested"]].astype(float)
    table[PCT_COLS] = table[PCT_COLS] * 100  # in punti percentuali per un formato uniforme
    pct = lambda label, digits=1: st.column_config.NumberColumn(label, format=f"%.{digits}f%%")
    num = lambda label, fmt="%.2f": st.column_config.NumberColumn(label, format=fmt)
    st.dataframe(
        table, width="stretch",
        column_config={
            "cagr": pct("CAGR"), "volatility": pct("Volatilità"),
            "sharpe": num("Sharpe"), "sharpe_se": num("± SE"), "sortino": num("Sortino"),
            "max_drawdown": pct("Max DD"), "max_dd_duration": num("Durata max DD (gg)", "%d"),
            "calmar": num("Calmar"), "turnover_yr": num("Turnover/anno"),
            "cost_yr": pct("Costi/anno", 3), "invested": pct("Investito", 0),
        } | {
            # Spiegazioni al passaggio del mouse sull'intestazione
            k: st.column_config.NumberColumn(cfg_label, format=fmt, help=tip)
            for k, cfg_label, fmt, tip in [
                ("sharpe_se", "± SE", "%.2f", "Errore standard dello Sharpe: la sua incertezza."),
                ("sortino", "Sortino", "%.2f", "Come lo Sharpe, ma penalizza solo le oscillazioni al ribasso."),
                ("max_dd_duration", "Durata max DD (gg)", "%d",
                 "Giorni di borsa consecutivi passati sotto il massimo precedente."),
                ("calmar", "Calmar", "%.2f", "CAGR diviso per il max drawdown: rendimento per unità di perdita peggiore."),
                ("invested", "Investito", "%.0f%%", "Quota media del capitale investita (il resto è liquidità)."),
            ]
        },
    )

with tab_trades:
    trades = results[MOM].trades
    trades = trades[(trades["date"] >= start) & (trades["date"] <= end)]
    st.caption(f"{len(trades)} variazioni di peso del momentum nella finestra di valutazione")
    st.dataframe(
        trades.sort_values("date", ascending=False),
        width="stretch", hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Data", format="YYYY-MM-DD"),
            "asset": "ETF",
            "weight_before": st.column_config.NumberColumn("Peso prima", format="percent"),
            "weight_after": st.column_config.NumberColumn("Peso dopo", format="percent"),
            "change": st.column_config.NumberColumn("Variazione", format="percent"),
        },
    )

with tab_log:
    st.caption(
        f"Ogni configurazione del momentum che visualizzi viene registrata. La famiglia "
        f"**{FAMILY}** conta **{trials}** configurazioni distinte: più varianti si provano, "
        f"più è probabile che la migliore sembri buona per caso. "
        f"Run attuale: `{outcomes[MOM].run_id}`."
    )
    log = ExperimentRegistry().runs(family=FAMILY, limit=200)
    log[["cagr", "max_drawdown"]] = log[["cagr", "max_drawdown"]].astype(float) * 100
    st.dataframe(
        log.drop(columns=["family", "final"]), width="stretch", hide_index=True,
        column_config={
            "id": "Run", "created_at": "Data (UTC)", "strategy": None,
            "params": "Parametri", "universe": "Universo", "status": "Stato",
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "cagr": st.column_config.NumberColumn("CAGR", format="%.1f%%"),
            "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
            "commit": "Commit",
        },
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
- La liquidità rende il tasso dei T-bill USA a 3 mesi (^IRX, circa 1,2% medio annuo nel 2005-2022).
- Sharpe, Sortino e criteri di promozione usano il rendimento **in eccesso** su quel tasso:
  restare in liquidità non viene premiato come se fosse un vantaggio della strategia.
- Nessuna leva, nessun costo di prestito titoli.

**Limiti.** Risultati storici e in-sample: i parametri vengono dalla letteratura, ma il
periodo analizzato si sovrappone in parte a quello in cui la strategia è stata
pubblicata. Universo scelto con il senno di poi (ETF che esistono ancora oggi).
I dati dal {HOLDOUT_START} sono riservati alla valutazione finale.
""")
