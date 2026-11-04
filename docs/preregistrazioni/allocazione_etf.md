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
