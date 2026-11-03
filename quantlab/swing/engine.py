"""Motore di backtest per lo swing trading: operazioni con ingresso e uscita.

A differenza del motore a pesi target (pensato per portafogli ribilanciati),
qui ogni operazione è un oggetto con un prezzo di ingresso, uno stop, un
target e un'uscita. Il motore elabora una giornata alla volta ("event-driven"),
come farebbe un bot reale.

Il contratto con la strategia
-----------------------------
La strategia riceve i dati OHLC e restituisce, per ogni asset, un DataFrame
di segnali calcolati con dati fino alla chiusura della data t:

- `entry` (bool): apri una posizione lunga;
- `exit` (bool): chiudi la posizione;
- `stop_distance` (float, facoltativo): distanza dello stop dal prezzo d'ingresso;
- `target_distance` (float, facoltativo): distanza del take profit;
- `score` (float, facoltativo): priorità quando ci sono più segnali che posti liberi.

Tempistica di ogni giornata t
-----------------------------
1. All'apertura: si chiudono le posizioni con un segnale di uscita del giorno
   prima, poi si aprono quelle con un segnale di ingresso del giorno prima.
2. Durante la giornata: se il minimo tocca lo stop o il massimo tocca il
   target, la posizione si chiude. Se l'apertura è già oltre lo stop (gap),
   si esce all'apertura, a un prezzo peggiore dello stop. Se nella stessa
   barra vengono toccati sia lo stop sia il target, non sappiamo quale sia
   venuto prima: assumiamo lo stop (ipotesi prudente).
3. Alla chiusura: si valuta il portafoglio e si leggono i segnali per domani.

Semplificazioni dichiarate: solo posizioni lunghe, azioni frazionarie,
liquidità a rendimento zero, nessuna leva.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = ["entry", "exit", "stop_distance", "target_distance", "score"]


@dataclass(frozen=True)
class SwingConfig:
    initial_capital: float = 100_000.0
    max_positions: int = 5
    sizing: str = "fixed_fraction"      # "fixed_fraction" oppure "risk"
    position_fraction: float = 0.2      # quota di capitale per posizione (tetto anche per "risk")
    risk_per_trade: float = 0.01        # con sizing "risk": perdita allo stop = 1% del capitale
    cost_bps: float = 5.0               # commissioni + metà spread, per lato
    slippage_bps: float = 5.0           # prezzo peggiore sugli ordini a mercato e sugli stop
    max_holding_days: int | None = None  # uscita a tempo

    def __post_init__(self):
        if self.sizing not in ("fixed_fraction", "risk"):
            raise ValueError("sizing deve essere 'fixed_fraction' o 'risk'")
        if not 0 < self.position_fraction <= 1:
            raise ValueError("position_fraction deve stare tra 0 e 1")
        if self.max_positions < 1:
            raise ValueError("max_positions deve essere almeno 1")


@dataclass
class _Position:
    asset: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    entry_value: float
    entry_cost: float
    stop: float | None
    target: float | None
    bars_held: int = 0


@dataclass
class SwingResult:
    returns: pd.Series          # rendimenti giornalieri del portafoglio
    equity: pd.Series
    weights: pd.DataFrame       # esposizione per asset a fine giornata
    turnover: pd.Series         # controvalore scambiato / capitale
    costs: pd.Series            # costi (commissioni) / capitale
    trades: pd.DataFrame        # una riga per operazione chiusa
    open_positions: pd.DataFrame
    config: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"SwingResult({self.equity.index[0].date()} → {self.equity.index[-1].date()}, "
                f"{len(self.trades)} operazioni, equity finale {self.equity.iloc[-1]:,.0f})")


def run_swing_backtest(
    data: dict[str, pd.DataFrame],
    signals: dict[str, pd.DataFrame],
    config: SwingConfig | None = None,
) -> SwingResult:
    """Simula una strategia swing su più asset.

    Args:
        data: per ogni asset, OHLC giornaliero (colonne open, high, low, close).
        signals: per ogni asset, segnali con le colonne di SIGNAL_COLUMNS.
    """
    config = config or SwingConfig()
    assets = sorted(data)
    if not assets:
        raise ValueError("nessun dato")
    missing = set(assets) - set(signals)
    if missing:
        raise ValueError(f"segnali mancanti per: {sorted(missing)}")

    dates = pd.DatetimeIndex(sorted(set().union(*(data[a].index for a in assets))))
    field_frames = {
        f: pd.DataFrame({a: data[a][f] for a in assets}).reindex(dates)
        for f in ("open", "high", "low", "close")
    }
    o, h, l, c = (field_frames[f].to_numpy() for f in ("open", "high", "low", "close"))
    c_ffill = field_frames["close"].ffill().to_numpy()
    sig = {col: _signal_matrix(signals, assets, dates, col) for col in SIGNAL_COLUMNS}

    cost_rate = config.cost_bps / 10_000
    slip = config.slippage_bps / 10_000
    cash = config.initial_capital
    equity_prev = config.initial_capital
    positions: dict[int, _Position] = {}
    pending_exit: set[int] = set()
    pending_entry: list[int] = []
    pending_levels: dict[int, tuple[float, float]] = {}

    n, k = len(dates), len(assets)
    equity = np.zeros(n)
    traded = np.zeros(n)
    paid = np.zeros(n)
    weights = np.zeros((n, k))
    trades: list[dict] = []

    def close_position(j: int, i: int, price: float, reason: str) -> None:
        nonlocal cash
        p = positions.pop(j)
        proceeds = p.shares * price
        fee = proceeds * cost_rate
        cash += proceeds - fee
        traded[i] += proceeds
        paid[i] += fee
        invested = p.entry_value + p.entry_cost
        pnl = proceeds - fee - invested
        trades.append({
            "asset": p.asset, "entry_date": p.entry_date, "entry_price": p.entry_price,
            "exit_date": dates[i], "exit_price": price, "shares": p.shares,
            "pnl": pnl, "return": pnl / invested, "bars_held": p.bars_held, "exit_reason": reason,
        })

    for i in range(n):
        has_bar = ~np.isnan(o[i])

        # 1a. Uscite all'apertura (segnali di ieri). Un ordine di uscita per una
        # posizione già chiusa (es. dallo stop) viene scartato: non deve colpire
        # una futura nuova posizione sullo stesso asset.
        pending_exit = {j for j in pending_exit if j in positions}
        for j in sorted(pending_exit):
            if j in positions and has_bar[j]:
                close_position(j, i, o[i, j] * (1 - slip), "segnale")
                pending_exit.discard(j)
        # 1b. Ingressi all'apertura, in ordine di priorità.
        for j in pending_entry:
            if len(positions) >= config.max_positions:
                break
            if j in positions or not has_bar[j]:
                continue
            fill = o[i, j] * (1 + slip)
            stop_d, target_d = pending_levels[j]
            stop = fill - stop_d if np.isfinite(stop_d) and stop_d > 0 else None
            target = fill + target_d if np.isfinite(target_d) and target_d > 0 else None
            value = config.position_fraction * equity_prev
            if config.sizing == "risk" and stop is not None:
                value = min(value, config.risk_per_trade * equity_prev / (fill - stop) * fill)
            value = min(value, cash / (1 + cost_rate))      # niente leva
            if value <= 0:
                continue
            fee = value * cost_rate
            cash -= value + fee
            traded[i] += value
            paid[i] += fee
            positions[j] = _Position(assets[j], dates[i], fill, value / fill, value, fee, stop, target)
        pending_entry, pending_levels = [], {}

        # 2. Stop e target durante la giornata.
        for j in list(positions):
            if not has_bar[j]:
                continue
            p = positions[j]
            hit_stop = p.stop is not None and l[i, j] <= p.stop
            hit_target = p.target is not None and h[i, j] >= p.target
            if hit_stop:                                  # prudente: lo stop vince sul target
                close_position(j, i, min(o[i, j], p.stop) * (1 - slip), "stop")
            elif hit_target:
                close_position(j, i, max(o[i, j], p.target), "target")

        # 3. Valutazione alla chiusura.
        held_value = np.zeros(k)
        for j, p in positions.items():
            held_value[j] = p.shares * c_ffill[i, j]
            if has_bar[j]:
                p.bars_held += 1
        equity[i] = cash + held_value.sum()
        weights[i] = held_value / equity[i]

        # 4. Segnali di oggi → ordini di domani.
        for j, p in positions.items():
            if not has_bar[j]:
                continue
            time_up = config.max_holding_days is not None and p.bars_held >= config.max_holding_days
            if sig["exit"][i, j] > 0 or time_up:
                pending_exit.add(j)
        candidates = [j for j in range(k)
                      if has_bar[j] and j not in positions and sig["entry"][i, j] > 0]
        candidates.sort(key=lambda j: (-np.nan_to_num(sig["score"][i, j], nan=-np.inf), assets[j]))
        pending_entry = candidates
        pending_levels = {j: (sig["stop_distance"][i, j], sig["target_distance"][i, j])
                          for j in candidates}

        traded[i] /= equity_prev
        paid[i] /= equity_prev
        equity_prev = equity[i]

    eq = pd.Series(equity, index=dates, name="equity")
    rets = eq.pct_change()
    rets.iloc[0] = equity[0] / config.initial_capital - 1
    open_pos = pd.DataFrame([
        {"asset": p.asset, "entry_date": p.entry_date, "entry_price": p.entry_price,
         "shares": p.shares, "stop": p.stop, "target": p.target}
        for p in positions.values()
    ])
    return SwingResult(
        returns=rets.rename("returns"),
        equity=eq,
        weights=pd.DataFrame(weights, index=dates, columns=assets),
        turnover=pd.Series(traded, index=dates, name="turnover"),
        costs=pd.Series(paid, index=dates, name="costs"),
        trades=pd.DataFrame(trades, columns=[
            "asset", "entry_date", "entry_price", "exit_date", "exit_price", "shares",
            "pnl", "return", "bars_held", "exit_reason"]),
        open_positions=open_pos,
        config=asdict(config),
    )


def _signal_matrix(signals, assets, dates, col) -> np.ndarray:
    out = np.full((len(dates), len(assets)), np.nan)
    for j, a in enumerate(assets):
        s = signals[a]
        if col in s.columns:
            out[:, j] = s[col].reindex(dates).astype(float).to_numpy()
    if col in ("entry", "exit"):
        out = np.nan_to_num(out, nan=0.0)
    return out


def trade_stats(trades: pd.DataFrame) -> pd.Series:
    """Statistiche per operazione (sui rendimenti netti di ogni trade)."""
    if trades.empty:
        return pd.Series({"trades": 0}, dtype="object")
    r = trades["return"]
    wins, losses = r[r > 0], r[r <= 0]
    streak = longest = 0
    for x in r:
        streak = streak + 1 if x <= 0 else 0
        longest = max(longest, streak)
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    return pd.Series({
        "trades": len(r),
        "win_rate": len(wins) / len(r),
        "avg_win": wins.mean() if len(wins) else np.nan,
        "avg_loss": losses.mean() if len(losses) else np.nan,
        "payoff_ratio": wins.mean() / -losses.mean() if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "profit_factor": trades.loc[trades["pnl"] > 0, "pnl"].sum() / gross_loss if gross_loss > 0 else np.nan,
        "expectancy": r.mean(),
        "avg_bars_held": trades["bars_held"].mean(),
        "max_consecutive_losses": longest,
        "exits_stop": int((trades["exit_reason"] == "stop").sum()),
        "exits_target": int((trades["exit_reason"] == "target").sum()),
    }, dtype="object")
