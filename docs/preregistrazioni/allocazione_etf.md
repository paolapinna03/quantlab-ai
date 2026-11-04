# Pre-registrazione: famiglia `allocazione_etf`

Data: 2026-11-04 · Scritta e salvata su Git **prima** di eseguire qualsiasi test sui dati reali.

## Ipotesi

Su un paniere multi-asset di ETF, un'allocazione basata sul rischio migliora il rapporto
rischio/rendimento rispetto al portafoglio a pesi uguali (1/N), fuori campione e al netto dei costi.

Aspettativa dichiarata in anticipo (da DeMiguel, Garlappi & Uppal 2009): volatilità e drawdown
più bassi di 1/N; Sharpe simile o di poco superiore; difficilmente una superiorità
statisticamente significativa.

## Universo e periodo

- ETF: SPY, EFA, EEM, IWM, TLT, IEF, GLD, DBC, VNQ (azioni USA/estere/emergenti/small cap,
  Treasury lunghi e medi, oro, materie prime, immobiliare).
- Dati fino al 2022-12-31. Holdout dal 2023 non usato.
- Ribilanciamento mensile, esecuzione il giorno dopo, costi 5 bp per lato,
  liquidità remunerata al T-bill, Sharpe sui rendimenti in eccesso.

## Candidati (esattamente questi, nessun altro)

| # | Metodo | Parametri |
|---|---|---|
| 1 | Inverse volatility | covarianza su 252 giorni, Ledoit-Wolf |
| 2 | Minima varianza | 252 giorni, Ledoit-Wolf, tetto 40% per ETF |
| 3 | Risk parity (ERC) | 252 giorni, Ledoit-Wolf, senza leva |
| 4 | Hierarchical Risk Parity | 252 giorni, Ledoit-Wolf |
| 5 | Risk parity + filtro momentum 12 mesi | come il 3; ETF con rendimento a 12 mesi ≤ 0 in liquidità |
| 6 | Media-varianza (solo confronto didattico) | 252 giorni, tetto 40% |

Walk-forward: per ogni metodo, finestra di stima scelta ogni anno tra 126, 252 e 504 giorni
usando solo il passato.

## Criteri di promozione (invariati, `quantlab.validation.Criteria`)

Deflated Sharpe ≥ 0,95 · Sharpe significativamente > 1/N (p ≤ 0,05) · walk-forward Sharpe > 0 ·
Sharpe > 0 in ognuno di 3 sotto-periodi · Sharpe > 0 con costi ×3 · max drawdown ≥ -35%.

Il criterio sul benchmark è il più severo: un esito negativo è atteso e va riportato com'è.

---

## Esito (aggiunto dopo il test, pre-registrazione nel commit d755edc)

Finestra di valutazione 2007-03-12 → 2022-12-30 (9 ETF), `python examples/04_risk_allocation.py`.

| Strategia | CAGR | Vol | Sharpe | Max DD | Turnover/anno | p vs 1/N | Criteri | Verdetto |
|---|---|---|---|---|---|---|---|---|
| Inverse volatility | 4,8% | 10,1% | 0,44 | -26,4% | 0,44 | 0,234 | 5/6 | Respinta |
| Minima varianza (tetto 40%) | 4,6% | 6,8% | 0,59 | -19,2% | 0,91 | 0,175 | 5/6 | Respinta |
| Risk parity (ERC) | 4,5% | 8,0% | 0,49 | -20,7% | 0,49 | 0,239 | 5/6 | Respinta |
| Hierarchical Risk Parity | 3,4% | 6,8% | 0,41 | -21,8% | 1,03 | 0,478 | 4/6 | Respinta |
| Risk parity + momentum 12m | 4,0% | 5,9% | 0,56 | -10,9% | 1,33 | 0,278 | 5/6 | Respinta |
| Media-varianza (didattico) | 5,4% | 9,7% | 0,51 | -20,7% | 3,93 | 0,317 | 5/6 | Respinta |
| Pesi uguali 1/N | 5,4% | 13,8% | 0,39 | -38,8% | 0,38 | — | — | — |

Lettura: l'aspettativa dichiarata è confermata. Tutti i metodi riducono volatilità e drawdown
rispetto a 1/N e hanno Sharpe più alto, ma nessuna differenza di Sharpe è statisticamente
significativa (p da 0,18 a 0,48). Il criterio fallito è lo stesso per tutti; HRP fallisce anche
il Deflated Sharpe. La media-varianza non è stata disastrosa qui, ma ha un turnover circa 4–8 volte
più alto degli altri metodi, segno di pesi instabili.
