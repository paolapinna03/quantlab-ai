"""Pagina Dati e registro: prezzi in cache, registro degli esperimenti, pre-registrazioni."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from styles import card_title, header
from quantlab.config import HOLDOUT_START
from quantlab.data import DataValidationError, cached_tickers, load_prices
from quantlab.data.rates import load_tbill_yield
from quantlab.experiments import ExperimentRegistry, reproduce

registry = ExperimentRegistry()
PREREG_DIR = Path(__file__).resolve().parents[1] / "docs" / "preregistrazioni"

st.html(header(
    eyebrow="Dati e registro",
    title="Dati di mercato ed esperimenti",
    meta="Prezzi scaricati, storico di tutti i tentativi registrati, ipotesi dichiarate in anticipo.",
    badges=[(f"Dati {HOLDOUT_START[:4]}+ riservati", "lock",
             "I backtest di ricerca non usano i dati dal 2023: servono per la verifica finale.")],
))

tab_data, tab_reg, tab_pre = st.tabs(["Dati di mercato", "Registro esperimenti", "Pre-registrazioni"])

# --- Dati -----------------------------------------------------------------------------

with tab_data:
    cached = cached_tickers()
    c1, c2 = st.columns([1, 2], gap="medium")
    with c1, st.container(border=True):
        st.html(card_title("Aggiorna", "fonte: Yahoo Finance"))
        if st.button("Aggiorna tutti i prezzi", type="primary", width="stretch",
                     help="Riscarica la storia completa di ogni ETF in cache e dei tassi T-bill. "
                          "Le date già usate nei backtest restano tracciate dall'impronta dei dati."):
            bar = st.progress(0.0)
            errors = []
            for i, t in enumerate(cached["ticker"], 1):
                try:
                    load_prices(t, refresh=True)
                except Exception as exc:
                    errors.append(f"{t}: {exc}")
                bar.progress(i / max(len(cached), 1), text=f"{t} aggiornato")
            try:
                load_tbill_yield(refresh=True)
            except Exception as exc:
                errors.append(f"tassi T-bill: {exc}")
            bar.empty()
            if errors:
                for e in errors:
                    st.error(e)
            else:
                st.success(f"Aggiornati {len(cached)} ETF e i tassi T-bill.")
        with st.form("add_ticker", border=False):
            new = st.text_input("Aggiungi un ETF (simbolo USA)", placeholder="es. XLB")
            add = st.form_submit_button("Scarica e controlla", width="stretch")
        if add:
            sym = (new or "").strip().upper()
            if not re.fullmatch(r"[A-Z^.\-]{1,10}", sym):
                st.error("Simbolo non valido.")
            else:
                try:
                    df = load_prices(sym, refresh=True)
                    st.success(f"{sym}: {len(df)} giorni dal {df.index[0]:%d %b %Y} al {df.index[-1]:%d %b %Y}.")
                except DataValidationError as exc:
                    st.error(f"Dati scartati dai controlli di qualità: {exc}")
                except Exception as exc:
                    st.error(f"Download non riuscito: {exc}")
        st.html('<div class="ql-note">yfinance è una fonte non ufficiale: adatta alla ricerca, '
                'non a decisioni con denaro reale.</div>')

    with c2, st.container(border=True):
        st.html(card_title("ETF in cache", f"{len(cached)} simboli"))
        st.dataframe(cached, width="stretch", hide_index=True, column_config={
            "ticker": "ETF", "first_date": "Dal", "last_date": "Al", "rows": "Giorni",
            "fetched_at": "Scaricato (UTC)",
            "warnings": st.column_config.NumberColumn(
                "Avvisi", help="Anomalie segnalate dai controlli (es. movimenti oltre il 25% in un giorno)."),
        })

# --- Registro -------------------------------------------------------------------------

with tab_reg:
    families = registry.families()
    if not families:
        st.info("Il registro è vuoto: ogni backtest eseguito nella pagina Backtesting o nella "
                "Strategy Discovery viene salvato qui.")
    else:
        summary = pd.DataFrame([
            {"famiglia": f, "configurazioni distinte": registry.count_trials(f),
             "run totali": len(registry.runs(family=f))} for f in families if f != "benchmark"
        ])
        st.html(card_title("Famiglie di ipotesi", "più configurazioni = soglia del Deflated Sharpe più alta"))
        st.dataframe(summary, width="stretch", hide_index=True)

        fam = st.selectbox("Famiglia", [f for f in families if f != "benchmark"])
        runs = registry.runs(family=fam, limit=500)
        runs[["cagr", "max_drawdown"]] = runs[["cagr", "max_drawdown"]].astype(float) * 100
        st.dataframe(runs.drop(columns=["family"]), width="stretch", hide_index=True, column_config={
            "id": "Run", "created_at": "Data (UTC)", "strategy": "Strategia", "params": "Parametri",
            "universe": "Universo", "status": "Stato", "final": "Finale",
            "sharpe": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "cagr": st.column_config.NumberColumn("CAGR", format="%.1f%%"),
            "max_drawdown": st.column_config.NumberColumn("Max DD", format="%.1f%%"),
            "commit": st.column_config.TextColumn("Commit", help="* = codice con modifiche non salvate"),
        })

        with st.container(border=True):
            st.html(card_title("Riproduci un run", "riesegue e confronta con il risultato registrato"))
            done = runs[runs["status"] == "completed"]
            if done.empty:
                st.caption("Nessun run completato in questa famiglia.")
            else:
                rid = st.selectbox("Run", done["id"],
                                   format_func=lambda r: f"{r} · {done.set_index('id').loc[r, 'params']}")
                if st.button("Riproduci"):
                    with st.spinner("Rieseguo il backtest…"):
                        cmp = reproduce(rid, registry)
                    same = (cmp["registrato"] - cmp["rieseguito"]).abs().max() < 1e-9
                    if cmp["data_changed"].any():
                        st.warning("I prezzi sono cambiati rispetto al run originale (es. Yahoo ha "
                                   "ricalcolato i prezzi aggiustati): piccole differenze sono attese.")
                    elif same:
                        st.success("Riprodotto esattamente: stessi dati, stesso risultato.")
                    else:
                        st.error("Stessi dati ma risultato diverso: il codice è cambiato dal run originale.")
                    st.dataframe(cmp, width="stretch")

# --- Pre-registrazioni -------------------------------------------------------------------

with tab_pre:
    docs = sorted(PREREG_DIR.glob("*.md")) if PREREG_DIR.exists() else []
    if not docs:
        st.info("Nessuna pre-registrazione salvata.")
    else:
        st.caption("Ipotesi, candidati e criteri scritti e salvati su Git prima di eseguire i test.")
        doc = st.selectbox("Documento", docs, format_func=lambda p: p.stem)
        st.markdown(doc.read_text())
