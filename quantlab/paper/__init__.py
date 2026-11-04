"""Paper trading: la strategia gira ogni giorno su dati nuovi, con soldi simulati."""

from quantlab.paper.engine import (
    RunSummary,
    create_account,
    halt,
    live_targets,
    resume,
    run_account,
    today_new_york,
)
from quantlab.paper.risk import RiskLimits, check_data, check_targets
from quantlab.paper.store import PaperStore

__all__ = [
    "PaperStore",
    "RiskLimits",
    "RunSummary",
    "check_data",
    "check_targets",
    "create_account",
    "halt",
    "live_targets",
    "resume",
    "run_account",
    "today_new_york",
]
