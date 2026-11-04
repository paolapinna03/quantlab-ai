"""Archivio del paper trading (SQLite, separato dal registro degli esperimenti).

Tabelle:
- accounts: un conto simulato per strategia (spec, capitale, stato, limiti di rischio);
- daily:    valore del conto ogni giorno, con liquidità e valore di ogni posizione;
- orders:   ogni ordine eseguito o bloccato, con il motivo;
- events:   avvisi del risk manager, arresti, ripartenze, allocazione iniziale;
- pending:  pesi decisi alla chiusura di ieri, da eseguire alla chiusura di oggi.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PAPER_DB = _PROJECT_ROOT / "data" / "paper.sqlite"       # uso manuale, non versionato
STATE_DIR = _PROJECT_ROOT / "state"                             # stato condiviso, versionato su Git
SHARED_PAPER_DB = STATE_DIR / "paper.sqlite"
AUTOMATION_FILE = STATE_DIR / "automation.json"
DEFAULT_PAPER_DB = LOCAL_PAPER_DB   # compatibilità


def automation_info() -> dict | None:
    """Configurazione dell'automazione (GitHub Actions), se attiva; None in modalità manuale.

    QUANTLAB_AUTOMATION=0 forza la modalità manuale (usato nei test).
    """
    if os.environ.get("QUANTLAB_AUTOMATION") == "0" or not AUTOMATION_FILE.exists():
        return None
    info = json.loads(AUTOMATION_FILE.read_text())
    return info if info.get("enabled") else None


def default_paper_db() -> Path:
    """Variabile QUANTLAB_PAPER_DB > stato condiviso (se l'automazione è attiva) > file locale."""
    if os.environ.get("QUANTLAB_PAPER_DB"):
        return Path(os.environ["QUANTLAB_PAPER_DB"])
    return SHARED_PAPER_DB if automation_info() else LOCAL_PAPER_DB

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    name TEXT PRIMARY KEY, created_at TEXT NOT NULL, spec_json TEXT NOT NULL,
    initial_capital REAL NOT NULL, status TEXT NOT NULL, limits_json TEXT NOT NULL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS daily (
    account TEXT NOT NULL, date TEXT NOT NULL, equity REAL NOT NULL, cash REAL NOT NULL,
    values_json TEXT NOT NULL, peak REAL NOT NULL, PRIMARY KEY (account, date)
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, signal_date TEXT,
    exec_date TEXT NOT NULL, asset TEXT NOT NULL, weight_before REAL, weight_after REAL,
    value REAL, cost REAL, status TEXT NOT NULL, reason TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, date TEXT NOT NULL,
    created_at TEXT NOT NULL, kind TEXT NOT NULL, message TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending (
    account TEXT PRIMARY KEY, signal_date TEXT NOT NULL, targets_json TEXT NOT NULL,
    forced INTEGER NOT NULL DEFAULT 0
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PaperStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_paper_db()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    # --- Conti ------------------------------------------------------------------

    def create_account(self, name: str, spec: dict, capital: float, limits: dict, note: str = "") -> None:
        with closing(self._connect()) as con, con:
            if con.execute("SELECT 1 FROM accounts WHERE name = ?", (name,)).fetchone():
                raise ValueError(f"esiste già un conto chiamato {name!r}")
            con.execute("INSERT INTO accounts VALUES (?,?,?,?,?,?,?)",
                        (name, _now(), json.dumps(spec), capital, "attivo", json.dumps(limits), note))

    def accounts(self) -> list[dict]:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM accounts ORDER BY created_at").fetchall()
        return [{"name": r["name"], "created_at": r["created_at"], "spec": json.loads(r["spec_json"]),
                 "initial_capital": r["initial_capital"], "status": r["status"],
                 "limits": json.loads(r["limits_json"]), "note": r["note"]} for r in rows]

    def account(self, name: str) -> dict:
        for a in self.accounts():
            if a["name"] == name:
                return a
        raise KeyError(f"conto {name!r} inesistente")

    def set_status(self, name: str, status: str) -> None:
        with closing(self._connect()) as con, con:
            con.execute("UPDATE accounts SET status = ? WHERE name = ?", (status, name))

    # --- Stato giornaliero -------------------------------------------------------

    def last_day(self, name: str) -> dict | None:
        with closing(self._connect()) as con:
            r = con.execute("SELECT * FROM daily WHERE account = ? ORDER BY date DESC LIMIT 1",
                            (name,)).fetchone()
        if r is None:
            return None
        return {"date": pd.Timestamp(r["date"]), "equity": r["equity"], "cash": r["cash"],
                "values": json.loads(r["values_json"]), "peak": r["peak"]}

    def save_day(self, name: str, date, equity, cash, values: dict, peak) -> None:
        with closing(self._connect()) as con, con:
            con.execute("INSERT OR REPLACE INTO daily VALUES (?,?,?,?,?,?)",
                        (name, str(pd.Timestamp(date).date()), equity, cash, json.dumps(values), peak))

    def history(self, name: str) -> pd.DataFrame:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM daily WHERE account = ? ORDER BY date", (name,)).fetchall()
        if not rows:
            return pd.DataFrame(columns=["equity", "cash", "peak"])
        df = pd.DataFrame([{"date": pd.Timestamp(r["date"]), "equity": r["equity"], "cash": r["cash"],
                            "peak": r["peak"], **json.loads(r["values_json"])} for r in rows])
        return df.set_index("date")

    # --- Ordini in attesa ---------------------------------------------------------

    def set_pending(self, name: str, signal_date, targets: dict, forced: bool = False) -> None:
        with closing(self._connect()) as con, con:
            con.execute("INSERT OR REPLACE INTO pending VALUES (?,?,?,?)",
                        (name, str(pd.Timestamp(signal_date).date()), json.dumps(targets), int(forced)))

    def pending(self, name: str) -> dict | None:
        with closing(self._connect()) as con:
            r = con.execute("SELECT * FROM pending WHERE account = ?", (name,)).fetchone()
        if r is None:
            return None
        return {"signal_date": pd.Timestamp(r["signal_date"]), "targets": json.loads(r["targets_json"]),
                "forced": bool(r["forced"])}

    def clear_pending(self, name: str) -> None:
        with closing(self._connect()) as con, con:
            con.execute("DELETE FROM pending WHERE account = ?", (name,))

    # --- Ordini ed eventi ----------------------------------------------------------

    def log_order(self, name, signal_date, exec_date, asset, before, after, value, cost, status, reason=""):
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT INTO orders (account, signal_date, exec_date, asset, weight_before, weight_after,"
                " value, cost, status, reason) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (name, str(pd.Timestamp(signal_date).date()) if signal_date is not None else None,
                 str(pd.Timestamp(exec_date).date()), asset, before, after, value, cost, status, reason))

    def orders(self, name: str) -> pd.DataFrame:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM orders WHERE account = ? ORDER BY id", (name,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def log_event(self, name: str, date, kind: str, message: str) -> None:
        with closing(self._connect()) as con, con:
            con.execute("INSERT INTO events (account, date, created_at, kind, message) VALUES (?,?,?,?,?)",
                        (name, str(pd.Timestamp(date).date()), _now(), kind, message))

    def events(self, name: str) -> pd.DataFrame:
        with closing(self._connect()) as con:
            rows = con.execute("SELECT * FROM events WHERE account = ? ORDER BY id", (name,)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
