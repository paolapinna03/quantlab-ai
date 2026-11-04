"""Registro degli esperimenti (SQLite).

Ogni esecuzione viene salvata, anche se fallisce, con:
- la specifica completa (strategia, parametri, universo, finestra, costi);
- l'impronta dei dati usati (hash + data di download per ogni ticker);
- la versione del codice (commit Git e se c'erano modifiche non salvate);
- le metriche e la serie dei rendimenti giornalieri.

Perché i rendimenti? Per calcolare in seguito il Deflated Sharpe Ratio
(Bailey & López de Prado, 2014) servono la distribuzione degli Sharpe di tutti
i tentativi e la forma (asimmetria, code) dei rendimenti del migliore.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quantlab.experiments.spec import ExperimentSpec

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = _PROJECT_ROOT / "data" / "experiments.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    family        TEXT NOT NULL,
    config_hash   TEXT NOT NULL,
    strategy      TEXT NOT NULL,
    spec_json     TEXT NOT NULL,
    hypothesis    TEXT,
    is_final      INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,          -- 'completed' | 'failed'
    error         TEXT,
    eval_start    TEXT,
    eval_end      TEXT,
    data_json     TEXT,
    code_commit   TEXT,
    code_dirty    INTEGER,
    metrics_json  TEXT,
    returns_blob  BLOB
);
CREATE INDEX IF NOT EXISTS idx_runs_family ON runs(family);
CREATE TABLE IF NOT EXISTS validations (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    family        TEXT NOT NULL,
    verdict       TEXT NOT NULL,          -- 'PROMOSSA' | 'RESPINTA'
    checks_json   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS discoveries (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    name          TEXT NOT NULL,
    config_json   TEXT NOT NULL
);
"""


class FinalEvaluationUsed(RuntimeError):
    """La valutazione finale sull'holdout per questa famiglia è già stata fatta."""


@dataclass
class RunRecord:
    id: str
    created_at: str
    family: str
    config_hash: str
    strategy: str
    spec: dict
    hypothesis: str
    is_final: bool
    status: str
    error: str | None
    eval_start: str | None
    eval_end: str | None
    data: dict
    code_commit: str | None
    code_dirty: bool | None
    metrics: dict


class ExperimentRegistry:
    def __init__(self, path: Path | str | None = None):
        # Ordine: argomento esplicito, variabile d'ambiente QUANTLAB_DB, default.
        self.path = Path(path or os.environ.get("QUANTLAB_DB") or DEFAULT_DB)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    # --- Scrittura ----------------------------------------------------------

    def log(
        self,
        spec: ExperimentSpec,
        *,
        status: str,
        error: str | None = None,
        eval_start: str | None = None,
        eval_end: str | None = None,
        data: dict | None = None,
        code: tuple[str | None, bool | None] = (None, None),
        metrics: dict | None = None,
        returns: pd.Series | None = None,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        blob = None
        if returns is not None:
            buf = io.BytesIO()
            returns.rename("r").to_frame().to_parquet(buf)
            blob = buf.getvalue()
        with closing(self._connect()) as con, con:
            con.execute(
                """INSERT INTO runs (id, created_at, family, config_hash, strategy, spec_json,
                   hypothesis, is_final, status, error, eval_start, eval_end, data_json,
                   code_commit, code_dirty, metrics_json, returns_blob)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    spec.family,
                    spec.config_hash(),
                    spec.strategy,
                    json.dumps(spec.to_dict(), default=str),
                    spec.hypothesis,
                    int(spec.final_evaluation),
                    status,
                    error,
                    eval_start,
                    eval_end,
                    json.dumps(data or {}, default=str),
                    code[0],
                    None if code[1] is None else int(code[1]),
                    json.dumps(metrics or {}, default=_json_default),
                    blob,
                ),
            )
        return run_id

    # --- Lettura ------------------------------------------------------------

    def get(self, run_id: str) -> RunRecord:
        with closing(self._connect()) as con:
            row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return _to_record(row)

    def returns(self, run_id: str) -> pd.Series | None:
        with closing(self._connect()) as con:
            row = con.execute("SELECT returns_blob FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None or row["returns_blob"] is None:
            return None
        return pd.read_parquet(io.BytesIO(row["returns_blob"]))["r"]

    def runs(self, family: str | None = None, limit: int | None = None) -> pd.DataFrame:
        """Tabella riassuntiva dei run, dal più recente."""
        query = "SELECT * FROM runs"
        args: tuple = ()
        if family:
            query += " WHERE family = ?"
            args = (family,)
        query += " ORDER BY created_at DESC, rowid DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        with closing(self._connect()) as con:
            rows = con.execute(query, args).fetchall()
        records = [_to_record(r) for r in rows]
        columns = ["id", "created_at", "family", "strategy", "params", "universe", "status",
                   "final", "sharpe", "cagr", "max_drawdown", "commit"]
        if not records:          # tabella vuota ma con le colonne: chi la usa non deve cambiare codice
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([
            {
                "id": r.id, "created_at": r.created_at, "family": r.family,
                "strategy": r.strategy, "params": json.dumps(r.spec.get("params", {})),
                "universe": ",".join(r.spec.get("universe", [])),
                "status": r.status, "final": r.is_final,
                "sharpe": r.metrics.get("sharpe"), "cagr": r.metrics.get("cagr"),
                "max_drawdown": r.metrics.get("max_drawdown"),
                "commit": (r.code_commit or "")[:7] + ("*" if r.code_dirty else ""),
            }
            for r in records
        ])

    def count_trials(self, family: str) -> int:
        """Numero di configurazioni *distinte* provate nella famiglia (fallite incluse)."""
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT COUNT(DISTINCT config_hash) AS n FROM runs WHERE family = ? AND is_final = 0",
                (family,),
            ).fetchone()
        return int(row["n"])

    def trial_sharpes(self, family: str) -> list[float]:
        """Sharpe annualizzato dell'ultimo run completato di ogni configurazione distinta."""
        with closing(self._connect()) as con:
            rows = con.execute(
                """SELECT config_hash, metrics_json FROM runs
                   WHERE family = ? AND is_final = 0 AND status = 'completed'
                   ORDER BY created_at, rowid""",
                (family,),
            ).fetchall()
        latest = {r["config_hash"]: json.loads(r["metrics_json"]).get("sharpe") for r in rows}
        return [float(s) for s in latest.values() if s is not None]

    def records(self, family: str) -> list[RunRecord]:
        """Tutti i run di una famiglia, dal più vecchio al più recente."""
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT * FROM runs WHERE family = ? ORDER BY created_at, rowid", (family,)
            ).fetchall()
        return [_to_record(r) for r in rows]

    # --- Verdetti e sessioni di discovery --------------------------------------

    def log_validation(self, run_id: str, family: str, verdict: str, checks: list[dict]) -> None:
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT OR REPLACE INTO validations VALUES (?,?,?,?,?)",
                (run_id, _now(), family, verdict, json.dumps(checks, default=_json_default)),
            )

    def validations(self, family: str) -> dict[str, dict]:
        """run_id → {"verdict": ..., "checks": [...]} per una famiglia."""
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM validations WHERE family = ?", (family,)).fetchall()
        return {r["run_id"]: {"verdict": r["verdict"], "checks": json.loads(r["checks_json"])}
                for r in rows}

    def log_discovery(self, name: str, config: dict) -> str:
        disc_id = uuid.uuid4().hex[:12]
        with closing(self._connect()) as con, con:
            con.execute("INSERT INTO discoveries VALUES (?,?,?,?)",
                        (disc_id, _now(), name, json.dumps(config, default=_json_default)))
        return disc_id

    def discoveries(self) -> list[dict]:
        """Sessioni di discovery, dalla più recente."""
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM discoveries ORDER BY created_at DESC, rowid DESC").fetchall()
        return [{"id": r["id"], "created_at": r["created_at"], "name": r["name"],
                 "config": json.loads(r["config_json"])} for r in rows]

    def families(self) -> list[str]:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT DISTINCT family FROM runs ORDER BY family").fetchall()
        return [r["family"] for r in rows]

    def check_final_allowed(self, family: str) -> None:
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT id FROM runs WHERE family = ? AND is_final = 1 AND status = 'completed'",
                (family,),
            ).fetchone()
        if row is not None:
            raise FinalEvaluationUsed(
                f"la famiglia {family!r} ha già una valutazione finale (run {row['id']}). "
                "Rifarla dopo aver visto il risultato trasformerebbe l'holdout in dati in-sample."
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_default(o):
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    return str(o)


def _to_record(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        id=row["id"],
        created_at=row["created_at"],
        family=row["family"],
        config_hash=row["config_hash"],
        strategy=row["strategy"],
        spec=json.loads(row["spec_json"]),
        hypothesis=row["hypothesis"] or "",
        is_final=bool(row["is_final"]),
        status=row["status"],
        error=row["error"],
        eval_start=row["eval_start"],
        eval_end=row["eval_end"],
        data=json.loads(row["data_json"] or "{}"),
        code_commit=row["code_commit"],
        code_dirty=None if row["code_dirty"] is None else bool(row["code_dirty"]),
        metrics=json.loads(row["metrics_json"] or "{}"),
    )
