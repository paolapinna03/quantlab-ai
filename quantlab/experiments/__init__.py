"""Esperimenti: specifica dichiarativa, esecuzione e registro riproducibile."""

from quantlab.experiments.registry import ExperimentRegistry, FinalEvaluationUsed, RunRecord
from quantlab.experiments.runner import (
    ExperimentOutcome,
    HoldoutViolation,
    reproduce,
    run_experiment,
)
from quantlab.experiments.spec import ExperimentSpec

__all__ = [
    "ExperimentOutcome",
    "ExperimentRegistry",
    "ExperimentSpec",
    "FinalEvaluationUsed",
    "HoldoutViolation",
    "RunRecord",
    "reproduce",
    "run_experiment",
]
