# QuantLab AI

Piattaforma di ricerca quantitativa assistita da AI per scoprire, testare e validare strategie di trading in modo rigoroso.

> **Stato:** Fase 1 — ambiente di sviluppo. Solo ricerca e simulazione: nessun trading con denaro reale.

## Avvio rapido (GitHub Codespaces)

1. **Code** → **Codespaces** → **Create codespace on main**
2. L'ambiente (Python 3.12 + librerie) si configura da solo tramite `.devcontainer/`.
3. Verifica:
   ```bash
   python -c "import pandas, yfinance, streamlit; print('OK')"
   ```

## Principi

- Il motore quantitativo (`quantlab/`) è indipendente dall'interfaccia (`app/`).
- Ogni esperimento viene registrato, anche quelli falliti.
- I dati out-of-sample finali non si usano mai per ottimizzare.
