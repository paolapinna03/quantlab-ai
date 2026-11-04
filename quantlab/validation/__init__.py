"""Validazione statistica: le strategie devono guadagnarsi la promozione."""

from quantlab.validation.final import (
    FinalCriteria,
    FinalEvaluationRefused,
    FinalReport,
    load_record,
    run_final_evaluation,
)
from quantlab.validation.gates import Check, Criteria, ValidationReport, validate_candidate
from quantlab.validation.stats import (
    DeflatedSharpe,
    SharpeDifference,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_difference_test,
)
from quantlab.validation.walkforward import WalkForwardResult, walk_forward

__all__ = [
    "Check",
    "Criteria",
    "DeflatedSharpe",
    "FinalCriteria",
    "FinalEvaluationRefused",
    "FinalReport",
    "SharpeDifference",
    "ValidationReport",
    "WalkForwardResult",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "load_record",
    "probabilistic_sharpe_ratio",
    "run_final_evaluation",
    "sharpe_difference_test",
    "validate_candidate",
    "walk_forward",
]
