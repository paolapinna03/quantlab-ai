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

---

## Esito (aggiunto dopo il test, pre-registrazione nel commit 8cdaef1)

Finestra di valutazione 2007-03-12 → 2022-12-30 (9 ETF), `python examples/06_trend_vol.py`.

| Strategia | CAGR | Vol | Sharpe | Max DD | Turnover/anno | p vs 1/N | p vs tsmom 12m | DSR | Criteri | Verdetto |
|---|---|---|---|---|---|---|---|---|---|---|
| Trend 12m + vol target 8% | 3,7% | 6,7% | 0,45 | -14,3% | 1,55 | 0,390 | 0,549 | 0,950* | 4/6 | Respinta |
| Trend multi-orizzonte | 4,2% | 6,8% | 0,52 | -9,8% | 2,52 | 0,249 | 0,309 | 0,973 | 5/6 | Respinta |
| Multi-orizzonte + vol target 8% | 3,8% | 6,0% | 0,51 | -9,8% | 2,42 | 0,265 | 0,322 | 0,971 | 5/6 | Respinta |
| Momentum 12m (tsmom_9etf) | 4,2% | 7,9% | 0,46 | -14,1% | 1,40 | — | — | — | — | — |
| Pesi uguali 1/N | 5,4% | 13,8% | 0,39 | -38,8% | 0,38 | — | — | — | — | — |

\* 0,9496 arrotondato: sotto la soglia di 0,95.

Lettura:
- **Volatility targeting**: nessun beneficio. Con ribilanciamento mensile la riduzione arriva in
  ritardo; drawdown e Sharpe restano quelli del momentum 12 mesi, con meno rendimento.
- **Trend multi-orizzonte**: drawdown da -14,1% a -9,8% e Sharpe da 0,46 a 0,52, stesso CAGR,
  in linea con l'aspettativa. La differenza di Sharpe non è significativa (p 0,31); il turnover
  è quasi doppio, già incluso nei costi e sopravvive ai costi ×3.
- Aggiungere il vol target al multi-orizzonte non migliora nulla.

**Regola per il paper trading:** candidati 2 e 3 falliscono solo il confronto con 1/N e battono
tsmom 12m su drawdown e Sharpe. Entra il candidato con lo Sharpe più alto: **Trend multi-orizzonte**
(conto `trend_multi`).
