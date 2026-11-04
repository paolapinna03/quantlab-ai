"""Pagina Analista AI: i resoconti giornalieri e le idee dell'AI, con il loro track record."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from styles import card_title, header
from quantlab.analyst.briefing import AI_ACCOUNT, BRIEFING_DIR
from quantlab.paper.store import automation_info, default_paper_db

st.html(header(
    eyebrow="Analista AI",
    title="Resoconto giornaliero",
    meta="Ogni sera dopo la chiusura: operazioni delle strategie, segnali, commento e idee di Claude.",
    badges=[("Soldi simulati", "lock", "Nessun ordine reale viene inviato a un broker."),
            ("Idee AI = esperimento", "warn",
             "Le idee dell'AI vanno in un conto simulato separato per misurarne il valore nel tempo.")],
))

files = sorted(BRIEFING_DIR.glob("*.md"), reverse=True) if BRIEFING_DIR.exists() else []

with st.expander("Come funziona e come si configura", expanded=not files):
    st.markdown("""
**Ogni sera** (dopo l'elaborazione dei conti su GitHub) l'analista:
1. raccoglie i **fatti del giorno** calcolati dal codice: conti, ordini, segnali a 12 mesi, eventi di rischio;
2. chiede a **Claude** un commento e le sue **idee** di portafoglio sui 9 ETF;
3. **scarta** il commento se contiene numeri che non stanno nei fatti, e le idee se violano i limiti
   (solo posizioni lunghe, massimo 5 ETF, massimo 25% ciascuno);
4. imposta le idee valide come ordini del conto simulato **ai_analyst**, eseguiti alla chiusura successiva;
5. invia il messaggio su **Telegram** e lo salva qui.

**Configurazione (una volta, su GitHub → Settings → Secrets and variables → Actions):**
`ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Senza chiave API il resoconto arriva lo
stesso, senza commento e idee; senza Telegram viene solo salvato qui.
""")

if not files:
    st.info("Nessun resoconto ancora: il primo arriverà dopo la prossima esecuzione serale su GitHub "
            "(poi `git pull` e ricarica).")
    st.stop()

dates = [f.stem for f in files]
with st.sidebar:
    day = st.selectbox("Giorno", dates)

left, right = st.columns([1.5, 1], gap="medium")
with left, st.container(border=True):
    st.html(card_title("Resoconto", day))
    st.markdown((BRIEFING_DIR / f"{day}.md").read_text())

with right, st.container(border=True):
    st.html(card_title("Idee dell'AI nel tempo", "conto ai_analyst"))
    rows = []
    for f in sorted(BRIEFING_DIR.glob("*.json")):
        data = json.loads(f.read_text())
        ai = data.get("ai") or {}
        ideas = (ai.get("idee") or {}).get("targets")
        rows.append({"giorno": f.stem,
                     "idee": ", ".join(f"{e} {w:.0%}" for e, w in ideas.items()) if ideas else
                             ("liquidità" if ideas == {} else "—"),
                     "applicate": "sì" if data.get("idee_applicate") else "no",
                     "problemi": "; ".join(ai.get("problemi", [])) if ai else "AI non attiva"})
    st.dataframe(pd.DataFrame(rows).iloc[::-1], width="stretch", hide_index=True)

    if default_paper_db().exists() or not automation_info():
        from quantlab.paper import PaperStore
        store = PaperStore()
        names = {a["name"] for a in store.accounts()}
        if AI_ACCOUNT in names and store.last_day(AI_ACCOUNT):
            last = store.last_day(AI_ACCOUNT)
            init = store.account(AI_ACCOUNT)["initial_capital"]
            st.metric("Conto ai_analyst", f"{last['equity']:,.2f}", f"{last['equity'] / init - 1:+.2%} dall'avvio",
                      border=True, help="Il confronto con le strategie validate è nella pagina Paper Trading.")
    st.caption("Giudicare le idee dell'AI richiede mesi: poche settimane sono rumore.")
