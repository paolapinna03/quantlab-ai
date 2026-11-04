"""QuantLab AI: punto di ingresso della dashboard.

Avvio:
    streamlit run app/streamlit_app.py

Le pagine sono file separati in app/. L'interfaccia non contiene logica
quantitativa: chiama solo funzioni di `quantlab/`.
"""

from __future__ import annotations

import streamlit as st

from styles import CSS

st.set_page_config(page_title="QuantLab AI", page_icon="◆", layout="wide")
st.html(CSS)

with st.sidebar:
    st.html('<div class="ql-brand">QUANT<span>LAB</span> AI</div>'
            '<div class="ql-sub">Piattaforma di ricerca quantitativa</div>')

page = st.navigation(
    [
        st.Page("page_backtesting.py", title="Backtesting", icon=":material/show_chart:", default=True),
        st.Page("page_discovery.py", title="Strategy Discovery", icon=":material/travel_explore:"),
        st.Page("page_paper.py", title="Paper Trading", icon=":material/play_circle:"),
    ],
    position="top",
)
page.run()
