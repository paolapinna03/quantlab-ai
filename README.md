# QuantLab AI

Piattaforma di ricerca quantitativa assistita da AI per scoprire, testare e validare strategie di trading in modo rigoroso.

> **Stato:** Fase 2 — dati, backtest e prima dashboard. Solo ricerca e simulazione: nessun trading con denaro reale.

## Avvio rapido (GitHub Codespaces)

1. **Code** → **Codespaces** → **Create codespace on main**
2. L'ambiente (Python 3.12 + librerie) si configura da solo tramite `.devcontainer/`.
3. Verifica:
   ```bash
   python -c "import pandas, yfinance, streamlit; print('OK')"
   ```

## Comandi principali

```bash
pytest                                      # test offline (veloci)
pytest -m network                           # test con dati reali
python -m quantlab.data SPY QQQ             # scarica e controlla i dati
python examples/01_momentum_vs_buy_and_hold.py
python examples/02_validate_momentum.py     # criteri di promozione
python examples/03_swing_strategies.py      # swing trading: RSI(2) e Donchian
python examples/04_risk_allocation.py       # allocazione basata sul rischio (pre-registrata)
python -m quantlab.discovery                # Strategy Discovery (sessione etf_tbill_v1, ~1 min)
python -m quantlab.experiments              # registro degli esperimenti
python -m quantlab.experiments --reproduce <id>
streamlit run app/streamlit_app.py          # dashboard: Backtesting + Strategy Discovery
```

## Principi

- Il motore quantitativo (`quantlab/`) è indipendente dall'interfaccia (`app/`).
- Ogni esperimento viene registrato (`data/experiments.sqlite`), anche quelli falliti.
- L'holdout (dal 2023) è bloccato dal codice e si usa una sola volta per famiglia.
- La liquidità rende il tasso dei T-bill USA; Sharpe e test statistici usano il rendimento in eccesso.
- I dati out-of-sample finali non si usano mai per ottimizzare.
- Una strategia è promossa solo se supera tutti i criteri fissati in anticipo
  (Deflated Sharpe, confronto con il benchmark, walk-forward, stabilità, costi, drawdown).
