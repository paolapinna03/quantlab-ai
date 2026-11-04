"""Ricostruisce una sessione di discovery dal registro (per dashboard e rapporti)."""

from __future__ import annotations

import pandas as pd

from quantlab.experiments import ExperimentRegistry
from quantlab.strategies.rules import describe


def load_session(registry: ExperimentRegistry, name: str | None = None) -> dict | None:
    """Ultima sessione con quel nome (o l'ultima in assoluto). None se non ce ne sono."""
    sessions = registry.discoveries()
    if name is not None:
        sessions = [s for s in sessions if s["name"] == name]
    if not sessions:
        return None
    session = sessions[0]
    name = session["name"]
    search_family, confirm_family = f"discovery_{name}", f"discovery_{name}_conferma"

    def latest(records):
        out = {}
        for r in records:                       # dal più vecchio al più recente: vince l'ultimo
            out[describe(r.spec.get("params", {}), r.spec.get("engine", {}))] = r
        return out

    search = pd.DataFrame([
        {"description": d, "run_id": r.id, "status": r.status,
         "sharpe": r.metrics.get("sharpe"), "cagr": r.metrics.get("cagr"),
         "max_drawdown": r.metrics.get("max_drawdown"), "trades": r.metrics.get("trade_trades")}
        for d, r in latest(registry.records(search_family)).items()
    ])
    verdicts = registry.validations(confirm_family)
    confirmed = pd.DataFrame([
        {"description": d, "run_id": r.id,
         "confirm_sharpe": r.metrics.get("sharpe"), "confirm_cagr": r.metrics.get("cagr"),
         "confirm_max_drawdown": r.metrics.get("max_drawdown"),
         "confirm_trades": r.metrics.get("trade_trades"),
         "eval_start": r.eval_start, "eval_end": r.eval_end, "universe": r.spec.get("universe"),
         "cost_bps": r.spec.get("cost_bps"), "cash": r.spec.get("cash", "zero"),
         "verdict": verdicts.get(r.id, {}).get("verdict", "—"),
         "checks": verdicts.get(r.id, {}).get("checks", [])}
        for d, r in latest(registry.records(confirm_family)).items()
        if r.status == "completed"
    ])
    if not search.empty:
        search["selected"] = search["description"].isin(
            confirmed["description"] if not confirmed.empty else [])
    if not confirmed.empty:
        train = search.set_index("description")["sharpe"]
        confirmed["train_sharpe"] = confirmed["description"].map(train)
        confirmed["checks_passed"] = confirmed["checks"].map(lambda cs: sum(c["passed"] for c in cs))
        confirmed = confirmed.sort_values("train_sharpe", ascending=False).reset_index(drop=True)
        confirmed.insert(0, "rank", [f"#{i}" for i in range(1, len(confirmed) + 1)])

    return {"session": session, "config": session["config"], "search": search,
            "confirmed": confirmed, "search_family": search_family,
            "confirm_family": confirm_family}
