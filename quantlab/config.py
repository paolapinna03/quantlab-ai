"""Impostazioni di ricerca condivise da tutto il progetto.

HOLDOUT: i dati da questa data in poi sono riservati alla valutazione finale.
Non vanno usati per sviluppare, confrontare o ottimizzare strategie.
Ogni volta che li guardiamo "solo per curiosità", smettono di essere
out-of-sample. Il modulo di validazione farà rispettare questa regola.
"""

HOLDOUT_START = "2023-01-01"
RESEARCH_END = "2022-12-31"

# Costo di transazione di default, in punti base (1 bp = 0.01%) per unità di
# capitale scambiata, in una direzione. Include commissioni, metà dello spread
# e slippage. Per ETF molto liquidi (SPY, QQQ) è un'ipotesi prudente.
DEFAULT_COST_BPS = 5.0
