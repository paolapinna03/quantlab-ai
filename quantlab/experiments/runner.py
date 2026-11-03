"""Esecuzione di un esperimento: spec → dati → strategia → backtest → registro.

Regole fatte rispettare dal codice (non solo per convenzione):
1. Senza `final_evaluation=True` la finestra di valutazione non può toccare
   l'holdout, e i dati vengono caricati solo fino a `eval_end`.
2. La valutazione finale si può fare **una sola volta** per famiglia.
3. Ogni esecuzione viene registrata, anche se fallisce.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantlab.backtest import BacktestConfig, BacktestResult, evaluate, run_backtest, warmup_end
from quantlab.config import HOLDOUT_START
from quantlab.data import cache_info, load_prices
from quantlab.experiments.registry import ExperimentRegistry
from quantlab.experiments.spec import ExperimentSpec
from quantlab.strategies.catalog import check_params, get_strategy, is_swing, warmup_days
from quantlab.swing import SwingConfig, run_swing_backtest, trade_stats

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class HoldoutViolation(ValueError):
    """Un esperimento di ricerca sta cercando di usare i dati riservati all'holdout."""


@dataclass
class ExperimentOutcome:
    spec: ExperimentSpec
    result: BacktestResult
    metrics: pd.Series
    eval_start: pd.Timestamp
    eval_end: pd.Timestamp
    run_id: str | None
    trials_in_family: int | None


def run_experiment(
    spec: ExperimentSpec,
    registry: ExperimentRegistry | None = None,
    prices: pd.DataFrame | None = None,
    ohlc: dict[str, pd.DataFrame] | None = None,
) -> ExperimentOutcome:
    """Esegue `spec`. Con `registry` il run viene salvato.

    `prices` (chiusure) e `ohlc` (dati completi per le strategie swing) servono ai test:
    se mancano, i dati vengono caricati fino a `eval_end`.
    """
    _validate(spec)
    if spec.final_evaluation and registry is not None:
        registry.check_final_allowed(spec.family)

    code = git_version()
    data_info: dict = {}
    try:
        if is_swing(spec.strategy):
            if ohlc is None:
                ohlc = {t: load_prices(t, end=spec.eval_end) for t in spec.universe}
            data = {t: ohlc[t].loc[: spec.eval_end] for t in spec.universe}
            prices = pd.DataFrame({t: d["close"] for t, d in data.items()})
            data_info = data_fingerprint(prices)
            signals = get_strategy(spec.strategy)(data, **spec.params)
            result = run_swing_backtest(data, signals,
                                        SwingConfig(cost_bps=spec.cost_bps, **spec.engine))
        else:
            if prices is None:
                prices = load_closes(spec.universe, end=spec.eval_end)
            prices = prices.loc[: spec.eval_end, list(spec.universe)]
            data_info = data_fingerprint(prices)
            targets = get_strategy(spec.strategy)(prices, **spec.params)
            cfg = BacktestConfig(cost_bps=spec.cost_bps, execution_lag=spec.execution_lag)
            result = run_backtest(prices, targets, cfg)

        start = pd.Timestamp(spec.eval_start) if spec.eval_start else _default_start(spec, prices)
        end = min(pd.Timestamp(spec.eval_end), prices.index[-1])
        metrics = evaluate(result, start, end)
        if is_swing(spec.strategy):
            t = result.trades
            in_window = t[(t["entry_date"] >= start) & (t["exit_date"] <= end)]
            metrics = pd.concat([metrics, trade_stats(in_window).add_prefix("trade_")])
    except Exception as exc:
        if registry is not None:
            registry.log(spec, status="failed", error=f"{type(exc).__name__}: {exc}",
                         data=data_info, code=code)
        raise

    run_id = trials = None
    if registry is not None:
        run_id = registry.log(
            spec, status="completed", eval_start=str(start.date()), eval_end=str(end.date()),
            data=data_info, code=code, metrics=metrics.to_dict(),
            returns=result.returns.loc[start:end],
        )
        trials = registry.count_trials(spec.family)
    return ExperimentOutcome(spec, result, metrics, start, end, run_id, trials)


def _validate(spec: ExperimentSpec) -> None:
    check_params(spec.strategy, spec.params)
    if is_swing(spec.strategy):
        if spec.execution_lag != 1:
            raise ValueError("le strategie swing eseguono sempre all'apertura del giorno dopo")
        try:
            SwingConfig(**spec.engine)
        except TypeError as exc:
            raise ValueError(f"parametri del motore swing non validi: {exc}") from None
    elif spec.engine:
        raise ValueError("il campo engine vale solo per le strategie swing")
    holdout = pd.Timestamp(HOLDOUT_START)
    if not spec.final_evaluation and pd.Timestamp(spec.eval_end) >= holdout:
        raise HoldoutViolation(
            f"eval_end {spec.eval_end} entra nell'holdout (dal {HOLDOUT_START}). "
            "Usa final_evaluation=True solo per la valutazione finale."
        )
    if spec.final_evaluation:
        if spec.eval_start is None or pd.Timestamp(spec.eval_start) < holdout:
            raise HoldoutViolation(
                f"la valutazione finale deve iniziare dal {HOLDOUT_START} (eval_start esplicito)."
            )


def _default_start(spec: ExperimentSpec, prices: pd.DataFrame) -> pd.Timestamp:
    return warmup_end(prices, warmup_days(spec.strategy, spec.params))


def load_closes(tickers: tuple[str, ...], end: str) -> pd.DataFrame:
    return pd.DataFrame({t: load_prices(t, end=end)["close"] for t in tickers})


def data_fingerprint(prices: pd.DataFrame) -> dict:
    """Hash dei prezzi usati + data di download della cache, per ogni ticker."""
    out = {}
    for col in prices.columns:
        s = prices[col].dropna()
        digest = hashlib.sha256(pd.util.hash_pandas_object(s, index=True).values.tobytes())
        info = cache_info(col) or {}
        out[col] = {
            "sha256": digest.hexdigest()[:16],
            "first": str(s.index[0].date()) if len(s) else None,
            "last": str(s.index[-1].date()) if len(s) else None,
            "rows": len(s),
            "fetched_at": info.get("fetched_at"),
        }
    return out


def git_version() -> tuple[str | None, bool | None]:
    """(commit corrente, ci sono modifiche non salvate?) oppure (None, None) fuori da Git."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=_PROJECT_ROOT,
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip() != ""
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def spec_from_record(record) -> ExperimentSpec:
    """Ricostruisce la spec di un run registrato, con la stessa finestra di valutazione."""
    d = dict(record.spec)
    d["universe"] = tuple(d["universe"])
    d["eval_start"] = d.get("eval_start") or record.eval_start
    return ExperimentSpec(**d)


def reproduce(
    run_id: str, registry: ExperimentRegistry, prices: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Riesegue un run registrato (senza registrarlo di nuovo) e confronta le metriche.

    `data_changed` segnala se i prezzi sono diversi da quelli del run originale
    (es. Yahoo ha ricalcolato i prezzi aggiustati dopo un nuovo dividendo).
    """
    record = registry.get(run_id)
    spec = spec_from_record(record)
    if prices is None:
        prices = load_closes(spec.universe, end=spec.eval_end)
    outcome = run_experiment(spec, registry=None, prices=prices)
    fresh = data_fingerprint(prices.loc[: spec.eval_end, list(spec.universe)])
    changed = any(fresh[t]["sha256"] != record.data.get(t, {}).get("sha256") for t in fresh)
    keys = ["cagr", "volatility", "sharpe", "max_drawdown"]
    return pd.DataFrame({
        "registrato": [record.metrics.get(k) for k in keys],
        "rieseguito": [float(outcome.metrics[k]) for k in keys],
    }, index=keys).assign(data_changed=changed)
