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

## Come si usa

Tutto si fa dalla dashboard:

```bash
streamlit run app/streamlit_app.py
```

| Pagina | Cosa ci fai |
|---|---|
| Backtesting | scegli una strategia (momentum, allocazione per rischio, swing), l'universo e i parametri; vedi risultati, criteri di promozione e verdetto |
| Strategy Discovery | avvii una ricerca automatica di strategie e ne vedi la conferma su dati mai usati |
| Paper Trading | crei un conto simulato, elabori i giorni nuovi, fermi il conto in emergenza |
| Dati e registro | aggiorni i prezzi, aggiungi ETF, consulti e riproduci ogni esperimento, leggi le pre-registrazioni |

## Automazione (GitHub Actions)

- **Paper trading giornaliero** (`.github/workflows/paper-daily.yml`): lun-ven alle 22:15 UTC elabora
  i conti `rp_momentum` e `tsmom_9etf` e salva lo stato in `state/` con un commit. Il track record è
  quindi datato e verificabile nella storia di Git. In locale: `git pull` e la dashboard mostra
  l'ultima giornata. Per elaborare subito o fermare un conto: Actions → *Paper trading giornaliero* →
  *Run workflow*.
- **Test** (`.github/workflows/tests.yml`): `pytest` a ogni push.

I comandi da terminale restano disponibili:

```bash
pytest                                      # test offline (veloci)
pytest -m network                           # test con dati reali
python -m quantlab.paper run                # elaborazione giornaliera del paper trading
python -m quantlab.discovery --name etf_v2  # ricerca automatica
```

## Principi

- Il motore quantitativo (`quantlab/`) è indipendente dall'interfaccia (`app/`).
- Ogni esperimento viene registrato (`data/experiments.sqlite`), anche quelli falliti.
- L'holdout (dal 2023) è bloccato dal codice e si usa una sola volta per famiglia.
- La liquidità rende il tasso dei T-bill USA; Sharpe e test statistici usano il rendimento in eccesso.
- I dati out-of-sample finali non si usano mai per ottimizzare.
- Una strategia è promossa solo se supera tutti i criteri fissati in anticipo
  (Deflated Sharpe, confronto con il benchmark, walk-forward, stabilità, costi, drawdown).
