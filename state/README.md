# Stato del paper trading automatico

Questa cartella è scritta **solo** dal workflow GitHub Actions `paper-daily.yml`, ogni sera dopo la
chiusura della borsa USA. Ogni giornata elaborata è un commit: il track record è datato e verificabile.

- `paper.sqlite`: archivio completo (conti, valori giornalieri, ordini, eventi).
- `paper/<conto>_daily.csv`: valore del conto, liquidità e valore di ogni posizione, giorno per giorno.
- `paper/<conto>_orders.csv`: ordini eseguiti o bloccati dal risk manager, con il motivo.
- `paper/<conto>_events.csv`: segnali, avvisi, arresti.

Non modificare questi file a mano. Per fermare un conto: GitHub → Actions → "Paper trading giornaliero" →
Run workflow → azione `ferma` o `ferma_e_vendi`.
