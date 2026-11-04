"""Paper trading giornaliero: la stessa logica del backtest, applicata a dati nuovi.

Da eseguire una volta al giorno, dopo la chiusura della borsa USA. Per ogni
giorno di borsa non ancora elaborato:

1. le posizioni crescono con il rendimento total-return del giorno, la
   liquidità con il tasso T-bill;
2. alla chiusura si eseguono gli ordini decisi alla chiusura PRECEDENTE,
   se il risk manager li approva (stessa tempistica del backtest: segnale
   oggi, esecuzione alla prossima chiusura);
3. si controlla il drawdown: oltre il limite, liquidazione e arresto;
4. si calcolano i segnali di oggi, che diventano gli ordini di domani.

La contabilità è in valore (non in numero di quote): con prezzi aggiustati
per i dividendi, applicare il rendimento giornaliero include automaticamente
i dividendi incassati.

Se il programma non gira per qualche giorno, al successivo avvio recupera
tutti i giorni mancanti in ordine. Eseguirlo due volte nello stesso giorno
non duplica nulla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantlab.data import load_prices
from quantlab.data.rates import daily_cash_returns, load_tbill_yield
from quantlab.paper.risk import RiskLimits, check_data, check_targets, drawdown
from quantlab.paper.store import PaperStore
from quantlab.strategies.catalog import get_strategy, is_swing


@dataclass
class RunSummary:
    account: str
    processed: list = field(default_factory=list)   # giorni elaborati
    filled: int = 0
    blocked: int = 0
    messages: list = field(default_factory=list)
    equity: float | None = None
    status: str = ""


def today_new_york() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def next_weekday(ts: pd.Timestamp) -> pd.Timestamp:
    d = ts + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


# Strategie "esterne": i pesi non li calcola una regola del catalogo, li imposta ogni giorno
# un altro componente (l'analista AI) tramite gli ordini in attesa.
EXTERNAL_STRATEGIES = {"ai_ideas"}


def live_targets(spec: dict, prices: pd.DataFrame) -> pd.DataFrame:
    """Pesi target della strategia, calcolati come dal vivo.

    Si aggiunge una riga vuota per il prossimo giorno feriale: serve solo a
    capire se l'ultimo giorno disponibile è una fine mese (il calendario è
    noto in anticipo), non contiene alcun prezzo. Approssimazione dichiarata:
    le festività di borsa non sono considerate.
    """
    if spec["strategy"] in EXTERNAL_STRATEGIES:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    phantom = pd.DataFrame(np.nan, index=[next_weekday(prices.index[-1])], columns=prices.columns)
    extended = pd.concat([prices, phantom])
    return get_strategy(spec["strategy"])(extended, **spec.get("params", {})).iloc[:-1]


def create_account(store: PaperStore, name: str, spec: dict, capital: float = 10_000.0,
                   limits: RiskLimits | None = None, note: str = "") -> None:
    if spec["strategy"] not in EXTERNAL_STRATEGIES:
        if is_swing(spec["strategy"]):
            raise ValueError("per ora il paper trading supporta le strategie di portafoglio, non quelle swing")
        get_strategy(spec["strategy"])  # verifica che esista
    store.create_account(name, spec, capital, vars(limits or RiskLimits()), note)


def halt(store: PaperStore, name: str, liquidate: bool, reason: str = "arresto manuale") -> None:
    """Pulsante di emergenza: nessun nuovo ordine; con `liquidate`, vende tutto alla prossima chiusura."""
    acc = store.account(name)
    store.set_status(name, "fermato")
    last = store.last_day(name)
    date = last["date"] if last else today_new_york()
    if liquidate:
        store.set_pending(name, date, {a: 0.0 for a in acc["spec"]["universe"]}, forced=True)
    else:
        store.clear_pending(name)
    store.log_event(name, date, "arresto", f"{reason}{' con liquidazione' if liquidate else ''}")


def resume(store: PaperStore, name: str) -> None:
    store.set_status(name, "attivo")
    last = store.last_day(name)
    store.log_event(name, last["date"] if last else today_new_york(), "ripartenza",
                    "conto riattivato: i nuovi segnali torneranno a generare ordini")


def run_account(
    store: PaperStore,
    name: str,
    prices: pd.DataFrame | None = None,
    tbill_yield: pd.Series | None = None,
    today: pd.Timestamp | None = None,
) -> RunSummary:
    """Elabora tutti i giorni di borsa nuovi per il conto `name`.

    `prices` e `tbill_yield` servono ai test; se mancano vengono scaricati aggiornati.
    """
    acc = store.account(name)
    spec, limits = acc["spec"], RiskLimits(**acc["limits"])
    universe = list(spec["universe"])
    today = pd.Timestamp(today) if today is not None else today_new_york()
    summary = RunSummary(account=name, status=acc["status"])

    if prices is None:
        prices = pd.DataFrame({t: load_prices(t, refresh=True)["close"] for t in universe})
    prices = prices.loc[:today, universe].dropna(how="all")

    blocking, warnings = check_data(prices, today, limits)
    for w in warnings:
        store.log_event(name, prices.index[-1], "avviso_dati", w)
        summary.messages.append(w)
    if blocking:
        for b in blocking:
            store.log_event(name, today, "dati_bloccati", b)
        summary.messages += [f"nessuna operazione: {b}" for b in blocking]
        return summary

    if spec.get("cash") == "tbill":
        y = tbill_yield if tbill_yield is not None else load_tbill_yield(refresh=True)
        rc = daily_cash_returns(y, prices.index)
    else:
        rc = pd.Series(0.0, index=prices.index)
    targets = live_targets(spec, prices)
    rets = prices.pct_change()
    cost_rate = spec.get("cost_bps", 5.0) / 10_000
    status = acc["status"]

    last = store.last_day(name)
    if last is None:
        # Nuovo conto: parte dall'ultima chiusura disponibile, tutto in liquidità.
        start = prices.index[-1]
        dates, cash, values, peak = [start], acc["initial_capital"], {a: 0.0 for a in universe}, acc["initial_capital"]
        fresh = True
        past = targets.loc[:start].dropna(how="all")
        if len(past):
            # Data del segnale = oggi: il conto nasce dopo questa chiusura, quindi gli
            # ordini possono essere eseguiti solo alla chiusura successiva.
            store.set_pending(name, start, past.iloc[-1].fillna(0.0).to_dict())
            store.log_event(name, start, "allocazione_iniziale",
                            f"si parte con i pesi dell'ultimo ribilanciamento ({past.index[-1].date()}), "
                            "eseguiti alla prossima chiusura")
    else:
        dates = list(prices.index[prices.index > last["date"]])
        cash, values, peak = last["cash"], {a: last["values"].get(a, 0.0) for a in universe}, last["peak"]
        fresh = False
        if not dates:
            summary.messages.append(f"nessun giorno nuovo dopo il {last['date'].date()}")
            summary.equity = last["equity"]
            return summary

    for d in dates:
        # 1. Rendimenti del giorno (non il primo giorno di un conto nuovo).
        if not fresh:
            r = rets.loc[d].fillna(0.0)
            values = {a: v * (1 + r[a]) for a, v in values.items()}
            cash *= 1 + rc.loc[d]
        fresh = False
        equity_pre = cash + sum(values.values())

        # 2. Esecuzione degli ordini decisi a una chiusura precedente.
        pend = store.pending(name)
        if pend and pend["signal_date"] < d and (status == "attivo" or pend["forced"]):
            current = pd.Series(values) / equity_pre
            t = pd.Series(pend["targets"]).reindex(universe).fillna(0.0)
            problems = [] if pend["forced"] else check_targets(t, current, limits)
            delta = t - current.reindex(universe).fillna(0.0)
            if problems:
                for a in delta.index[delta.abs() > 1e-9]:
                    store.log_order(name, pend["signal_date"], d, a, current.get(a, 0.0), t[a],
                                    delta[a] * equity_pre, 0.0, "bloccato", "; ".join(problems))
                    summary.blocked += 1
                store.log_event(name, d, "ordini_bloccati", "; ".join(problems))
                summary.messages.append("ordini bloccati: " + "; ".join(problems))
            else:
                cost = delta.abs().sum() * cost_rate * equity_pre
                equity_post = equity_pre - cost
                for a in delta.index[delta.abs() > 1e-9]:
                    store.log_order(name, pend["signal_date"], d, a, current.get(a, 0.0), t[a],
                                    delta[a] * equity_pre, abs(delta[a]) * cost_rate * equity_pre,
                                    "eseguito")
                    summary.filled += 1
                # Il costo riduce in proporzione tutte le posizioni, come nel motore di backtest.
                values = {a: t[a] * equity_post for a in universe}
                cash = (1 - t.sum()) * equity_post
            store.clear_pending(name)

        # 3. Valutazione e controllo del drawdown.
        equity = cash + sum(values.values())
        peak = max(peak, equity)
        dd = drawdown(equity, peak)
        if status == "attivo" and dd <= limits.halt_drawdown:
            status = "fermato"
            store.set_status(name, status)
            store.set_pending(name, d, {a: 0.0 for a in universe}, forced=True)
            msg = (f"drawdown {dd:.1%} oltre il limite {limits.halt_drawdown:.0%}: "
                   "liquidazione alla prossima chiusura e arresto")
            store.log_event(name, d, "arresto_automatico", msg)
            summary.messages.append(msg)

        # 4. Segnali di oggi → ordini della prossima chiusura.
        if status == "attivo" and d in targets.index and targets.loc[d].notna().any():
            store.set_pending(name, d, targets.loc[d].fillna(0.0).to_dict())
            store.log_event(name, d, "segnale", "ribilanciamento deciso: esecuzione alla prossima chiusura")

        store.save_day(name, d, equity, cash, values, peak)
        summary.processed.append(d)

    summary.equity = equity
    summary.status = status
    return summary
