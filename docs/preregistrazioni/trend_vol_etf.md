# Pre-registrazione: famiglia `trend_vol_etf`

Data: 2026-11-04 · Scritta e salvata su Git **prima** di eseguire qualsiasi test sui dati reali.

## Ipotesi

Due estensioni del time-series momentum a 12 mesi (`tsmom_9etf`, già in paper trading)
migliorano il controllo del rischio a parità di rendimento:

1. **Trend su più orizzonti** (Hurst, Ooi & Pedersen 2017): media dei segnali a 1, 3, 6 e 12 mesi
   invece del solo 12 mesi. Reagisce prima alle inversioni e dipende meno da una singola finestra.
2. **Volatility targeting** (Moreira & Muir 2017): se la volatilità stimata del portafoglio supera
   il target, tutte le posizioni vengono ridotte in proporzione (mai leva).

Aspettativa dichiarata in anticipo: drawdown più basso di `tsmom_9etf`, Sharpe simile o di poco
superiore. Nessun aumento importante del rendimento. Una superiorità significativa rispetto a 1/N
è improbabile, come per tutte le famiglie precedenti.

## Universo e periodo

- ETF: SPY, EFA, EEM, IWM, TLT, IEF, GLD, DBC, VNQ (gli stessi del paper trading).
- Dati fino al 2022-12-31. L'holdout 2023–2026 è già stato usato per la verifica finale e non
  viene usato qui: la prova fuori campione sarà il paper trading.
- Pesi 1/N per asset moltiplicati per il segnale, ribilanciamento mensile, esecuzione il giorno
  dopo, costi 5 bp per lato, liquidità al T-bill, Sharpe sui rendimenti in eccesso.

## Candidati (esattamente questi, nessun altro)

Strategia `trend_signals` (`quantlab/strategies/trend.py`).

| # | Nome | Orizzonti (giorni) | Volatility target | Griglia walk-forward |
|---|---|---|---|---|
| 1 | Trend 12m + vol target | 252 | 8% annuo, stima su 63 giorni | target 6%, 8%, 10% |
| 2 | Trend multi-orizzonte | 21, 63, 126, 252 | nessuno | orizzonti {21,63,126,252}, {63,126,252}, {126,252} |
| 3 | Multi-orizzonte + vol target | 21, 63, 126, 252 | 8% annuo, stima su 63 giorni | target 6%, 8%, 10% |

Il target dell'8% è scelto prima del test come livello vicino alla volatilità tipica di un
portafoglio trend multi-asset: così la riduzione interviene soprattutto nei periodi turbolenti.

## Criteri di promozione (invariati, `quantlab.validation.Criteria`)

Deflated Sharpe ≥ 0,95 · Sharpe significativamente > 1/N (p ≤ 0,05) · walk-forward Sharpe > 0 ·
Sharpe > 0 in ognuno di 3 sotto-periodi · Sharpe > 0 con costi ×3 · max drawdown ≥ -35%.

## Confronto secondario (dichiarato ora, non è un criterio di promozione)

Ogni candidato viene confrontato con `time_series_momentum` 12 mesi sugli stessi 9 ETF e nella
stessa finestra (stesso test di Jobson-Korkie/Memmel usato contro 1/N), riportando Sharpe,
volatilità, max drawdown e p-value.

## Regola per il paper trading (decisa ora)

Un candidato entra in paper trading come nuovo conto simulato **solo se**:
- supera tutti i criteri, **oppure** fallisce soltanto il confronto con 1/N (come le strategie
  già in paper trading); **e**
- ha max drawdown migliore e Sharpe almeno uguale a `time_series_momentum` 12 mesi.

Se più candidati soddisfano la regola, entra solo quello con lo Sharpe più alto. Se nessuno la
soddisfa, nessun nuovo conto viene aperto e l'esito viene riportato com'è.
