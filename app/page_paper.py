"""Pagina Paper Trading: crea conti simulati, elabora i giorni nuovi, ferma in emergenza."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from charts import PLOTLY_CONFIG, color_map, drawdown_chart, equity_chart
from styles import card_title, esc, header
import re

from quantlab.paper import PaperStore, RiskLimits, create_account, halt, resume, run_account
from quantlab.paper.presets import PRESET_LABELS
from quantlab.paper.presets import PRESETS as PAPER_PRESETS

store = PaperStore()
accounts = store.accounts()


def create_form(expanded: bool) -> None:
    with st.expander("➕ Nuovo conto simulato", expanded=expanded):
        with st.form("nuovo_conto"):
            c1, c2, c3 = st.columns([1.2, 2, 1])
            new = c1.text_input("Nome del conto", value="rp_momentum" if not accounts else "")
            preset = c2.selectbox("Strategia", list(PAPER_PRESETS),
                                  format_func=lambda k: PRESET_LABELS.get(k, k))
            capital = c3.number_input("Capitale simulato", 1_000.0, 1_000_000.0, 10_000.0, 1_000.0)
            go = st.form_submit_button("Crea il conto", type="primary")
        if not go:
            return
        if not re.fullmatch(r"[A-Za-z0-9_]{2,40}", new or ""):
            st.error("Il nome può contenere solo lettere, numeri e _ (da 2 a 40 caratteri).")
            return
        try:
            create_account(store, new, PAPER_PRESETS[preset]["spec"], capital, RiskLimits(),
                           PAPER_PRESETS[preset]["note"])
        except ValueError as exc:
            st.error(str(exc))
            return
        with st.spinner("Scarico i prezzi e avvio il primo giorno…"):
            run_account(store, new)
        st.session_state["paper_account"] = new
        st.rerun()


if not accounts:
    st.html(header("Paper Trading", "Nessun conto simulato", "Soldi simulati, prezzi reali.",
                   [("Soldi simulati", "lock")]))
    st.info("Crea il primo conto: la strategia girerà ogni giorno su prezzi reali e nuovi, "
            "con soldi simulati. Nessun ordine viene inviato a un broker.")
    create_form(expanded=True)
    st.stop()

with st.sidebar:
    names = [a["name"] for a in accounts]
    default = st.session_state.get("paper_account")
    name = st.selectbox("Conto", names, index=names.index(default) if default in names else 0)
    st.html('<div class="ql-section">Giornata</div>')
    if st.button("Elabora i giorni nuovi", type="primary", width="stretch",
                 help="Scarica i prezzi aggiornati, esegue gli ordini in attesa e calcola i segnali. "
                      "Da usare una volta al giorno dopo le 22:30 italiane; i giorni saltati vengono recuperati."):
        with st.spinner("Scarico i prezzi ed elaboro…"):
            summary = run_account(store, name)
        st.session_state["paper_summary"] = summary
        st.rerun()
acc = store.account(name)
spec, limits = acc["spec"], acc["limits"]
hist = store.history(name)
last = store.last_day(name)
pend = store.pending(name)
active = acc["status"] == "attivo"

# --- Pulsante di emergenza --------------------------------------------------------

with st.sidebar:
    st.html('<div class="ql-section">Pulsante di emergenza</div>')
    if active:
        sure = st.checkbox("Confermo di voler fermare il conto")
        if st.button("Ferma", disabled=not sure, width="stretch",
                     help="Nessun nuovo ordine; le posizioni restano."):
            halt(store, name, liquidate=False, reason="arresto dalla dashboard")
            st.rerun()
        if st.button("Ferma e vendi tutto", type="primary", disabled=not sure, width="stretch",
                     help="Nessun nuovo ordine e vendita di tutto alla prossima chiusura."):
            halt(store, name, liquidate=True, reason="arresto dalla dashboard")
            st.rerun()
    else:
        st.caption("Il conto è fermato: i segnali non generano ordini.")
        if st.button("Riattiva il conto"):
            resume(store, name)
            st.rerun()

# --- Intestazione ------------------------------------------------------------------

p = spec.get("params", {})
title = {"risk_parity": "Risk parity", "min_variance": "Minima varianza", "hrp": "HRP",
         "inverse_vol": "Inverse volatility", "mean_variance": "Media-varianza"}.get(
    p.get("method"), spec["strategy"])
if p.get("momentum_filter"):
    title += f" + momentum {p['momentum_filter'] // 21} mesi"
start = hist.index[0] if len(hist) else None
st.html(header(
    eyebrow=f"Paper Trading · conto {name}",
    title=title,
    meta=(f"{len(spec['universe'])} ETF · capitale simulato <b>{acc['initial_capital']:,.0f}</b> · "
          + (f"avvio <b>{start:%d %b %Y}</b> · ultimo giorno elaborato <b>{last['date']:%d %b %Y}</b>"
             if last else "non ancora avviato")),
    badges=[("● attivo" if active else "■ fermato", "pass" if active else "fail",
             "Attivo: i segnali generano ordini. Fermato: nessun nuovo ordine."),
            ("Soldi simulati", "lock", "Nessun ordine reale viene inviato a un broker."),
            ("Strategia non promossa", "warn", acc.get("note") or "")],
))

with st.expander("Come funziona"):
    st.markdown(f"""
- **Ogni giorno dopo le 22:30 italiane** (chiusura di New York) premi **"Elabora i giorni nuovi"**
  nella barra laterale. Se salti dei giorni, vengono recuperati in ordine; premerlo due volte non duplica nulla.
- **Tempistica identica al backtest**: i pesi decisi alla chiusura di un giorno vengono eseguiti alla
  chiusura successiva. Un test verifica che il paper trading e il backtest diano gli stessi rendimenti.
- **La strategia ribilancia a fine mese.** Negli altri giorni le posizioni si muovono solo con i prezzi.
- **Il risk manager** blocca gli ordini se un ETF supera il {limits['max_weight']:.0%}, se l'esposizione
  supera il {limits['max_gross']:.0%} o se i dati sono vecchi; con un drawdown oltre il
  {limits['halt_drawdown']:.0%} vende tutto e ferma il conto.
- I dati dal 2023 in poi sono **già serviti a questa strategia per calcolare i pesi**, ma il suo
  track record comincia dal giorno di avvio: ogni rendimento qui sotto è su dati che non esistevano
  quando la strategia è stata scelta.
""")

summary = st.session_state.pop("paper_summary", None)
if summary is not None:
    days = ", ".join(f"{d:%d %b}" for d in summary.processed) or "nessun giorno nuovo"
    st.success(f"Elaborazione completata · giorni: {days} · ordini eseguiti {summary.filled}, "
               f"bloccati {summary.blocked}")
    for m in summary.messages:
        st.info(m)

create_form(expanded=False)

if not last:
    st.info("Il conto esiste ma non è ancora stato elaborato: usa \"Elabora i giorni nuovi\".")
    st.stop()

# --- KPI ---------------------------------------------------------------------------

eq = hist["equity"]
ret = last["equity"] / acc["initial_capital"] - 1
dd = last["equity"] / last["peak"] - 1
k = st.columns(5)
k[0].metric("Valore", f"{last['equity']:,.2f}", border=True,
            help="Valore del conto simulato all'ultima chiusura elaborata.")
k[1].metric("Dall'avvio", f"{ret:+.2%}", border=True, help="Rendimento dal primo giorno, al netto dei costi.")
k[2].metric("Drawdown", f"{dd:.1%}", f"arresto a {limits['halt_drawdown']:.0%}", delta_color="off",
            border=True, help="Distanza dal valore massimo raggiunto.")
k[3].metric("Liquidità", f"{last['cash'] / last['equity']:.0%}", border=True,
            help="Quota non investita; rende il tasso dei T-bill USA.")
k[4].metric("Giorni di track record", f"{len(hist)}", border=True,
            help="Giorni di borsa elaborati. Per giudicare servono mesi, non giorni.")

# --- Andamento e posizioni -----------------------------------------------------------

left, right = st.columns([1.6, 1], gap="medium")
with left, st.container(border=True):
    st.html(card_title("Valore del conto", "soldi simulati"))
    if len(hist) >= 2:
        st.plotly_chart(equity_chart({"Conto": eq}, color_map(["Conto"]), log=False),
                        config=PLOTLY_CONFIG, width="stretch")
        st.plotly_chart(drawdown_chart({"Conto": eq.pct_change().fillna(0.0).pipe(
            lambda r: (1 + r).cumprod() / (1 + r).cumprod().cummax() - 1)}, color_map(["Conto"])),
            config=PLOTLY_CONFIG, width="stretch")
    else:
        st.html('<div class="ql-note">Il grafico compare dal secondo giorno elaborato.</div>')

with right, st.container(border=True):
    st.html(card_title("Posizioni", "peso attuale → in attesa"))
    current = pd.Series(last["values"]) / last["equity"]
    target = pd.Series(pend["targets"]) if pend else pd.Series(dtype=float)
    rows = ""
    for a in spec["universe"]:
        now = current.get(a, 0.0)
        nxt = target.get(a) if pend else None
        arrow = f"→ {nxt:.0%}" if nxt is not None and abs(nxt - now) > 1e-4 else ""
        rows += f"<tr><td>{esc(a)}</td><td>{now:.1%}</td><td>{arrow}</td></tr>"
    cash_now = last["cash"] / last["equity"]
    cash_next = f"→ {1 - target.sum():.0%}" if pend else ""
    rows += f"<tr><td>Liquidità</td><td>{cash_now:.1%}</td><td>{cash_next}</td></tr>"
    st.html(f"""<table class="ql-table"><thead><tr><th>ETF</th><th>Ora</th><th>In attesa</th></tr></thead>
      <tbody>{rows}</tbody></table>
      <div class="ql-note">{"Ordini decisi il " + f"{pend['signal_date']:%d %b %Y}" +
       ": eseguiti alla prossima chiusura." if pend else "Nessun ordine in attesa."}</div>""")

# --- Registro ------------------------------------------------------------------------

tab_o, tab_e, tab_l = st.tabs(["Ordini", "Eventi", "Limiti di rischio"])
with tab_o:
    o = store.orders(name)
    if o.empty:
        st.caption("Nessun ordine ancora.")
    else:
        st.dataframe(o.drop(columns=["id", "account"]).iloc[::-1], width="stretch", hide_index=True,
                     column_config={
                         "signal_date": "Segnale", "exec_date": "Esecuzione", "asset": "ETF",
                         "weight_before": st.column_config.NumberColumn("Peso prima", format="percent"),
                         "weight_after": st.column_config.NumberColumn("Peso dopo", format="percent"),
                         "value": st.column_config.NumberColumn("Controvalore", format="%.2f"),
                         "cost": st.column_config.NumberColumn("Costo", format="%.2f"),
                         "status": "Stato", "reason": "Motivo"})
with tab_e:
    e = store.events(name)
    st.dataframe(e[["date", "kind", "message"]].iloc[::-1] if not e.empty else e, width="stretch",
                 hide_index=True, column_config={"date": "Data", "kind": "Tipo", "message": "Dettaglio"})
with tab_l:
    st.markdown(f"""
| Limite | Valore |
|---|---|
| Peso massimo per ETF | {limits['max_weight']:.0%} |
| Esposizione massima | {limits['max_gross']:.0%} (niente leva) |
| Turnover massimo per ribilanciamento | {limits['max_turnover']:.1f} |
| Arresto automatico per drawdown | {limits['halt_drawdown']:.0%} |
| Età massima dei dati | {limits['max_data_age_days']} giorni |
| Movimento giornaliero sospetto | {limits['max_daily_move']:.0%} |
""")
