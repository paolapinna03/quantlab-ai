"""Pagina Backtesting: qualsiasi strategia del catalogo contro i benchmark, con validazione.

L'interfaccia non contiene logica quantitativa: chiama solo funzioni di
`quantlab/`. Ogni numero mostrato viene da un backtest eseguito davvero.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import (
    PLOTLY_CONFIG, color_map, drawdown_chart, equity_chart, walk_forward_chart, weights_chart,
)
from presets import PRESETS, UNIVERSE
from styles import card_title, criteria_table, esc, header, section, verdict_block
from quantlab.config import DEFAULT_CASH, DEFAULT_COST_BPS, HOLDOUT_START
from quantlab.experiments import ExperimentRegistry, ExperimentSpec, run_experiment
from quantlab.metrics import drawdown_series, equity_curve
from quantlab.strategies.catalog import full_params
from quantlab.validation import Criteria, validate_candidate

BENCH, EQUAL = "SPY buy & hold", "Pesi uguali"


# --- Calcoli (in cache: si ricalcolano solo se cambiano gli input) ---------------------

def candidate_spec(key, tickers, params, engine, cost_bps, lag) -> ExperimentSpec:
    p = PRESETS[key]
    return ExperimentSpec(cash=DEFAULT_CASH, family=p.family, strategy=p.strategy, params=params,
                          engine=engine, universe=tickers, hypothesis=p.hypothesis,
                          cost_bps=cost_bps, execution_lag=lag)


def equal_weight_spec(tickers, cost_bps, lag, eval_start=None) -> ExperimentSpec:
    return ExperimentSpec(cash=DEFAULT_CASH, family="benchmark", strategy="buy_and_hold",
                          params={"rebalance": "M"}, universe=tickers, eval_start=eval_start,
                          cost_bps=cost_bps, execution_lag=lag)


@st.cache_data(show_spinner="Eseguo i backtest…")
def run_all(key, tickers, params, engine, cost_bps, lag):
    """Candidata (registrata come tentativo nella sua famiglia) + benchmark sulla stessa finestra."""
    cand = run_experiment(candidate_spec(key, tickers, params, engine, cost_bps, lag),
                          ExperimentRegistry())
    start = str(cand.eval_start.date())
    bench = run_experiment(ExperimentSpec(cash=DEFAULT_CASH, family="benchmark",
                                          strategy="buy_and_hold", universe=("SPY",),
                                          eval_start=start, cost_bps=cost_bps, execution_lag=1))
    equal = run_experiment(equal_weight_spec(tickers, cost_bps, 1, eval_start=start))
    return {BENCH: bench, EQUAL: equal, PRESETS[key].short: cand}


@st.cache_data(show_spinner="Verifico i criteri di promozione…")
def run_validation(key, tickers, params, engine, cost_bps, lag, trials):
    """`trials` è nella chiave della cache: con più tentativi la soglia del Deflated Sharpe sale."""
    p = PRESETS[key]
    grid = p.grid(params) if p.grid else None
    return validate_candidate(candidate_spec(key, tickers, params, engine, cost_bps, lag),
                              equal_weight_spec(tickers, cost_bps, 1), ExperimentRegistry(),
                              grid, Criteria())


# --- Barra laterale ---------------------------------------------------------------------

with st.sidebar:
    st.html(section("Strategia"))
    key = st.selectbox("Strategia", list(PRESETS), format_func=lambda k: PRESETS[k].label,
                       label_visibility="collapsed")
    preset = PRESETS[key]
    st.html(section("Universo"))
    tickers = st.multiselect(
        "ETF", options=list(UNIVERSE), default=preset.default_universe,
        format_func=lambda t: f"{t} · {UNIVERSE[t]}", label_visibility="collapsed",
        key=f"universe_{key}",
    )
    with st.container():
        params, engine = preset.controls()
    st.html(section("Esecuzione"))
    cost_bps = st.number_input("Costi per lato (bp)", 0.0, 100.0, DEFAULT_COST_BPS, 1.0)
    if preset.swing:
        lag = 1
        st.caption("Swing: ordini all'apertura del giorno dopo, stop e target durante la giornata.")
    else:
        lag = st.segmented_control(
            "Esecuzione dell'ordine", [0, 1, 2], default=1,
            format_func=lambda x: {0: "Stessa chiusura", 1: "Giorno dopo", 2: "2 giorni dopo"}[x],
            help="Quando viene eseguito l'ordine rispetto alla chiusura del segnale. "
                 "\"Giorno dopo\" è l'ipotesi realistica; \"stessa chiusura\" è ottimistica.",
        )
        lag = 1 if lag is None else lag
    st.html(section("Visualizzazione"))
    log_scale = st.toggle("Scala logaritmica", value=True)

if not tickers:
    st.info("Scegli almeno un ETF nel pannello a sinistra.")
    st.stop()

tickers = tuple(tickers)
CAND = preset.short
try:
    outcomes = run_all(key, tickers, params, engine, cost_bps, lag)
except Exception as exc:  # errori dei dati o del motore: mostrali, non nasconderli
    st.error(f"Impossibile eseguire il backtest: {exc}")
    st.stop()

results = {name: o.result for name, o in outcomes.items()}
start, end = outcomes[CAND].eval_start, outcomes[CAND].eval_end
returns = {name: r.returns.loc[start:end] for name, r in results.items()}
stats = pd.DataFrame({name: o.metrics for name, o in outcomes.items()}).T
trials = outcomes[CAND].trials_in_family
try:
    report = run_validation(key, tickers, params, engine, cost_bps, lag, trials)
    validation_error = ""
except Exception as exc:  # la validazione non deve nascondere i risultati del backtest
    report, validation_error = None, str(exc)
colors = color_map([BENCH, EQUAL, CAND])
reference = [full_params(preset.strategy, r) for r in preset.reference]
modified = (full_params(preset.strategy, params) not in reference
            or (preset.swing and engine.get("max_positions") != 5) or lag != 1)


# --- Intestazione ---------------------------------------------------------------------

badges = [
    (f"{trials} configurazion{'e' if trials == 1 else 'i'} provat{'a' if trials == 1 else 'e'}",
     "warn" if trials > len(reference) else "",
     f"Varianti di questa ipotesi (famiglia {preset.family}) già testate e registrate. Più sono, "
     "più alta diventa la soglia del Deflated Sharpe."),
    (f"Dati {HOLDOUT_START[:4]}+ riservati", "lock",
     f"I dati dal {HOLDOUT_START} non vengono usati qui: servono per la verifica finale."),
    ("Periodo di ricerca", "", "Risultati sul periodo usato per studiare la strategia (in-sample): "
                               "tendono a essere più ottimistici del futuro."),
]
if modified:
    badges.insert(0, ("Parametri modificati", "warn",
                      "Diversi dalla configurazione dichiarata in anticipo: risultati esplorativi."))
if report is not None:
    badges.insert(0, (f"{'✓' if report.promoted else '✗'} {report.verdict}",
                      "pass" if report.promoted else "fail",
                      "Esito dei criteri di promozione: vedi la sezione più in basso."))
st.html(header(
    eyebrow=f"Backtesting · famiglia {preset.family}",
    title=f"{preset.label} contro i benchmark",
    meta=(f"<b>{start:%d %b %Y} → {end:%d %b %Y}</b> · {len(tickers)} ETF · "
          f"costi <b>{cost_bps:g} bp</b> per lato · esecuzione <b>"
          f"{'apertura del giorno dopo' if preset.swing else {0: 'stessa chiusura', 1: 'giorno dopo', 2: '2 giorni dopo'}[lag]}</b>"),
    badges=badges,
))
if modified:
    st.warning("Configurazione diversa da quella dichiarata in anticipo. Ogni variante provata dopo aver "
               "visto i risultati aumenta il rischio di overfitting: considera questi numeri esplorativi.",
               icon=":material/warning:")

with st.expander("Come leggere questa pagina"):
    st.markdown(f"""
1. **Scegli la strategia** nella barra laterale: momentum, allocazione per rischio o swing.
   Ogni strategia ha la sua famiglia nel registro e la sua configurazione di riferimento.
2. **In alto** il verdetto e i numeri chiave. Le frecce confrontano con il **portafoglio a pesi uguali**
   sugli stessi ETF, lo stesso benchmark dei criteri di promozione.
3. **Criteri di promozione**: sei controlli statistici fissati prima di vedere i risultati.
   Basta un fallimento per respingere la strategia.
4. **Cambiare i parametri è esplorazione, non scoperta.** Ogni configurazione che guardi viene
   registrata e alza l'asticella per tutte le altre.

Dati fino al 2022; il periodo dal {HOLDOUT_START[:4]} è escluso e servirà per la verifica finale.
""")


# --- KPI ----------------------------------------------------------------------------

def pp(x: float) -> str:
    return f"{x * 100:+.1f} pp vs pesi uguali"


cand, bench = stats.loc[CAND], stats.loc[EQUAL]
k = st.columns(5)
k[0].metric("CAGR", f"{cand['cagr']:.2%}", pp(cand["cagr"] - bench["cagr"]), border=True,
            help="Rendimento medio annuo composto, al netto dei costi.")
k[1].metric(f"Sharpe ± {cand['sharpe_se']:.2f}", f"{cand['sharpe']:.2f}",
            f"{cand['sharpe'] - bench['sharpe']:+.2f} vs pesi uguali", border=True,
            help="Rendimento in eccesso sul T-bill per unità di rischio, annualizzato. ± = errore "
                 "standard (Lo, 2002): differenze sotto ~2 errori standard non sono distinguibili.")
k[2].metric("Max drawdown", f"{cand['max_drawdown']:.1%}",
            pp(cand["max_drawdown"] - bench["max_drawdown"]), border=True,
            help="Perdita massima dal picco più alto raggiunto in precedenza.")
k[3].metric("Volatilità", f"{cand['volatility']:.1%}",
            pp(cand["volatility"] - bench["volatility"]), delta_color="inverse", border=True,
            help="Quanto oscillano i rendimenti, annualizzato.")
if preset.swing:
    k[4].metric("Operazioni", f"{int(cand.get('trade_trades', 0) or 0)}",
                f"win rate {cand.get('trade_win_rate', 0):.0%}", delta_color="off", border=True,
                help="Operazioni chiuse nella finestra; win rate = quota di operazioni in guadagno.")
else:
    k[4].metric("Turnover / anno", f"{cand['turnover_yr']:.2f}×", f"costi {cand['cost_yr']:.2%} / anno",
                delta_color="off", border=True,
                help="Capitale scambiato in un anno (1× = l'intero portafoglio).")


# --- Equity + confronto -----------------------------------------------------------------

left, right = st.columns([2.3, 1], gap="medium")
with left, st.container(border=True):
    st.html(card_title("Valore di 100 investito", "scala log" if log_scale else "scala lineare"))
    curves = {n: equity_curve(r, start_value=100) for n, r in returns.items()}
    st.plotly_chart(equity_chart(curves, colors, log_scale), config=PLOTLY_CONFIG, width="stretch")

with right, st.container(border=True):
    st.html(card_title("Confronto strategie", "ordinate per Sharpe"))
    rows = "".join(
        f"""<tr><td><span class="ql-dot" style="background:{colors[name]}"></span>{esc(name)}</td>
          <td>{s['sharpe']:.2f} <span class="se">±{s['sharpe_se']:.2f}</span></td>
          <td>{s['cagr']:.1%}</td><td>{s['max_drawdown']:.1%}</td></tr>"""
        for name, s in stats.sort_values("sharpe", ascending=False).iterrows()
    )
    st.html(f"""<table class="ql-table">
        <thead><tr><th>Strategia</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th></tr></thead>
        <tbody>{rows}</tbody></table>
      <div class="ql-note">Stessa finestra per tutte, al netto dei costi. Sharpe sul rendimento in
      eccesso rispetto ai T-bill USA a 3 mesi.</div>""")

    st.html(card_title(f"Esposizione {CAND.lower()}", "vuoto = liquidità"))
    w = results[CAND].weights.loc[start:end]
    if w.shape[1] > 8:   # oltre 8 colori la palette non è più distinguibile: raggruppa
        top = w.mean().sort_values(ascending=False).index[:7]
        w = w[top].assign(Altri=w.drop(columns=top).sum(axis=1))
    st.plotly_chart(weights_chart(w, color_map(list(w.columns)), height=215),
                    config=PLOTLY_CONFIG, width="stretch")


# --- Validazione -----------------------------------------------------------------------

v_left, v_right = st.columns([2.3, 1], gap="medium")
if report is None:
    st.error(f"Validazione non disponibile: {validation_error}")
else:
    passed = sum(c.passed for c in report.checks)
    with v_left, st.container(border=True):
        st.html(card_title("Criteri di promozione", "fissati prima di vedere i risultati · benchmark: pesi uguali"))
        st.html(criteria_table(report.checks))
    with v_right, st.container(border=True):
        st.html(card_title("Verdetto", f"{passed}/{len(report.checks)} criteri superati"))
        failed = [c.name for c in report.checks if not c.passed]
        st.html(verdict_block(
            report.promoted,
            "Supera tutti i criteri: può passare alla valutazione sull'holdout." if report.promoted else
            f"Non supera: {', '.join(failed)}. Le soglie non si abbassano dopo aver visto l'esito: "
            "si torna alle ipotesi.",
        ))
        wf = report.details.get("walk_forward")
        if wf is not None:
            st.html(card_title("Walk-forward", "Sharpe per anno di test"))
            st.plotly_chart(walk_forward_chart(wf.folds), config=PLOTLY_CONFIG, width="stretch")
            st.html('<div class="ql-note">Ogni anno il parametro è scelto usando solo gli anni '
                    'precedenti (passa sopra una barra per vederlo).</div>')
        elif preset.swing:
            st.html('<div class="ql-note">Per le strategie swing il walk-forward non è ancora disponibile: '
                    'la verifica su dati nuovi passa dalla Strategy Discovery.</div>')


# --- Drawdown -----------------------------------------------------------------------

with st.container(border=True):
    st.html(card_title("Drawdown", "distanza dal massimo precedente"))
    st.plotly_chart(drawdown_chart({n: drawdown_series(r) for n, r in returns.items()}, colors),
                    config=PLOTLY_CONFIG, width="stretch")


# --- Dettagli -----------------------------------------------------------------------

tab_metrics, tab_trades, tab_log, tab_method = st.tabs(
    ["Tutte le metriche", "Operazioni", "Registro esperimenti", "Metodologia"])

with tab_metrics:
    pct_cols = ["cagr", "volatility", "max_drawdown", "cost_yr", "invested"]
    table = stats[["cagr", "volatility", "sharpe", "sharpe_se", "sortino", "max_drawdown",
                   "max_dd_duration", "calmar", "turnover_yr", "cost_yr", "invested"]].astype(float)
    table[pct_cols] = table[pct_cols] * 100
    pct = lambda label, digits=1, tip=None: st.column_config.NumberColumn(label, format=f"%.{digits}f%%", help=tip)
    num = lambda label, fmt="%.2f", tip=None: st.column_config.NumberColumn(label, format=fmt, help=tip)
    st.dataframe(table, width="stretch", column_config={
        "cagr": pct("CAGR"), "volatility": pct("Volatilità"), "sharpe": num("Sharpe"),
        "sharpe_se": num("± SE", tip="Errore standard dello Sharpe: la sua incertezza."),
        "sortino": num("Sortino", tip="Come lo Sharpe, ma penalizza solo le oscillazioni al ribasso."),
        "max_drawdown": pct("Max DD"),
        "max_dd_duration": num("Durata max DD (gg)", "%d", "Giorni di borsa sotto il massimo precedente."),
        "calmar": num("Calmar", tip="CAGR diviso per il max drawdown."),
        "turnover_yr": num("Turnover/anno"), "cost_yr": pct("Costi/anno", 3),
        "invested": pct("Investito", 0, "Quota media del capitale investita (il resto è liquidità)."),
    })

with tab_trades:
    trades = results[CAND].trades
    if preset.swing:
        t = trades[(trades["entry_date"] >= start) & (trades["exit_date"] <= end)]
        st.caption(f"{len(t)} operazioni chiuse nella finestra di valutazione")
        st.dataframe(t.sort_values("exit_date", ascending=False), width="stretch", hide_index=True,
                     column_config={
                         "asset": "ETF",
                         "entry_date": st.column_config.DateColumn("Ingresso", format="YYYY-MM-DD"),
                         "exit_date": st.column_config.DateColumn("Uscita", format="YYYY-MM-DD"),
                         "entry_price": st.column_config.NumberColumn("Prezzo ingresso", format="%.2f"),
                         "exit_price": st.column_config.NumberColumn("Prezzo uscita", format="%.2f"),
                         "shares": None, "pnl": st.column_config.NumberColumn("P&L", format="%.0f"),
                         "return": st.column_config.NumberColumn("Rendimento", format="percent"),
                         "bars_held": "Giorni", "exit_reason": "Motivo uscita"})
    else:
        t = trades[(trades["date"] >= start) & (trades["date"] <= end)]
        st.caption(f"{len(t)} variazioni di peso nella finestra di valutazione")
        st.dataframe(t.sort_values("date", ascending=False), width="stretch", hide_index=True,
                     column_config={
                         "date": st.column_config.DateColumn("Data", format="YYYY-MM-DD"), "asset": "ETF",
                         "weight_before": st.column_config.NumberColumn("Peso prima", format="percent"),
                         "weight_after": st.column_config.NumberColumn("Peso dopo", format="percent"),
                         "change": st.column_config.NumberColumn("Variazione", format="percent")})

with tab_log:
    st.caption(f"Ogni configurazione che visualizzi viene registrata. La famiglia **{preset.family}** "
               f"conta **{trials}** configurazioni distinte. Run attuale: `{outcomes[CAND].run_id}`. "
               "L'elenco completo di tutte le famiglie è nella pagina Dati e registro.")
    log = ExperimentRegistry().runs(family=preset.family, limit=200)
    log[["cagr", "max_drawdown"]] = log[["cagr", "max_drawdown"]].astype(float) * 100
    st.dataframe(log.drop(columns=["family", "final"]), width="stretch", hide_index=True,
                 column_config={"id": "Run", "created_at": "Data (UTC)", "strategy": None,
                                "params": "Parametri", "universe": "Universo", "status": "Stato",
                                "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                                "cagr": st.column_config.NumberColumn("CAGR", format="%.1f%%"),
                                "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
                                "commit": "Commit"})

with tab_method:
    st.markdown(preset.method_md(params, engine) + f"""

**Benchmark.** Buy-and-hold di SPY e portafoglio a pesi uguali ribilanciato mensilmente sugli stessi ETF.

**Ipotesi del backtest.**
- Prezzi aggiustati per dividendi e frazionamenti (Yahoo Finance, fonte non ufficiale).
- Costi: {cost_bps:g} bp per lato{" più 5 bp di slippage" if preset.swing else ""}.
- La liquidità rende il tasso dei T-bill USA a 3 mesi; Sharpe e criteri usano il rendimento in eccesso.
- Nessuna leva, nessuna posizione corta salvo dove indicato.

**Limiti.** Risultati storici e in-sample; universo scelto con il senno di poi (ETF che esistono
ancora oggi). I dati dal {HOLDOUT_START} sono riservati alla valutazione finale.
""")
