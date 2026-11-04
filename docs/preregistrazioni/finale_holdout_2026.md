# Pre-registrazione: valutazione finale sull'holdout (2023 → oggi)

Data: 2026-11-04 · Scritta e salvata su Git **prima** di eseguire la valutazione.
Esecuzione: `python examples/05_final_evaluation.py` (una sola volta; il codice rifiuta una ripetizione).

## Cosa si valuta

Le due strategie oggi in paper trading, con la configurazione esatta di `quantlab/paper/presets.py`
(stessi 9 ETF: SPY, EFA, EEM, IWM, TLT, IEF, GLD, DBC, VNQ; costi 5 bp per lato; liquidità al T-bill;
esecuzione alla chiusura del giorno dopo).

| Famiglia | Configurazione | Come è stata scelta |
|---|---|---|
| `allocazione_etf` | Risk parity (ERC) + filtro momentum 12 mesi, finestra 252 giorni, Ledoit-Wolf | Candidato #5 della pre-registrazione `allocazione_etf.md`; scelto per il paper trading per il drawdown più basso |
| `tsmom_etf` | Time-series momentum 12 mesi, pesi 1/N, solo lungo | Configurazione della letteratura; universo a 9 ETF scelto **dopo** il confronto alla pari con la risk parity |

Periodo: dal 2023-01-01 all'ultima chiusura disponibile al momento dell'esecuzione.

## Criteri (FinalCriteria in `quantlab/validation/final.py`)

Una strategia è **CONFERMATA** se supera tutti e tre:

1. **Sharpe in eccesso sul T-bill > 0** nell'holdout.
2. **Max drawdown ≥ -20%** nell'holdout (lo stesso limite dell'arresto automatico del paper trading).
3. **Nessun crollo rispetto al backtest**: Sharpe holdout ≥ Sharpe in-sample (2005/2007–2022) − 2 errori standard dell'holdout.

Riportati ma **non** criteri (circa 4 anni non danno potenza statistica sufficiente):
confronto con pesi uguali 1/N sugli stessi ETF e con buy-and-hold SPY (Sharpe, CAGR, drawdown, p-value).

## Cosa ho già visto dei dati 2023+ (dichiarato per trasparenza)

- Avviando il paper trading (2026-11-04) sono stati calcolati i pesi di fine ottobre 2026, che usano i
  rendimenti a 12 mesi da novembre 2025 a ottobre 2026, e ho stampato quei rendimenti a 12 mesi
  (es. DBC +53%, GLD -2%, TLT -6%).
- Controllando una fonte dati esterna ho visto i prezzi giornalieri di SPY da fine 2025 a novembre 2026.
- **Non** è mai stato calcolato alcun rendimento, Sharpe o drawdown delle strategie sul periodo 2023+.

## Aspettativa dichiarata

Incerta. La ricerca raccolta indica un 2023-2025 debole per il trend following dei CTA professionali;
la risk parity con filtro momentum ha avuto in-sample uno Sharpe di 0,56 su 2007-2022. Un esito negativo
è plausibile e verrà riportato com'è. L'esito non modifica i conti di paper trading già avviati.

---

## Esito (aggiunto dopo l'esecuzione, pre-registrazione nel commit 5877b40)

Holdout 2023-01-01 → 2026-11-03. Dati completi in `state/final_evaluations.json`.

| | CAGR | Vol | Sharpe | Max DD | Verdetto |
|---|---|---|---|---|---|
| **RP + momentum** in-sample (2007-2022) | 4,0% | 5,9% | 0,56 | -10,9% | |
| **RP + momentum** holdout | 8,4% | 6,7% | 0,58 | -7,4% | **CONFERMATA** (3/3) |
| **Momentum 9 ETF** in-sample (2007-2022) | 4,2% | 7,9% | 0,46 | -14,1% | |
| **Momentum 9 ETF** holdout | 9,9% | 8,4% | 0,65 | -8,9% | **CONFERMATA** (3/3) |
| Pesi uguali 1/N, holdout | 12,5% | 10,2% | 0,78 | -9,8% | — |
| SPY buy & hold, holdout | 20,9% | 14,8% | 1,06 | -18,8% | — |

Lettura:
- Su anni mai usati, entrambe le strategie si sono comportate **come nel backtest o meglio**: Sharpe
  positivo e simile (0,58 e 0,65 contro 0,56 e 0,46), drawdown contenuti (-7% e -9%).
- In assoluto hanno reso **meno** di 1/N e molto meno di SPY, in un periodo di forte rialzo azionario.
  È coerente con strategie difensive che tengono parte del capitale in liquidità; le differenze non sono
  statisticamente significative (p da 0,74 a 0,88).
- Il terzo criterio si è rivelato poco severo: con meno di 4 anni l'errore standard dello Sharpe è circa
  0,5, quindi la soglia era negativa. "Confermata" significa che la strategia non si è rotta fuori
  campione, non che sia migliore delle alternative.
