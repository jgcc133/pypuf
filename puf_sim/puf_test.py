"""Population-level PUF evaluation entry points."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from pypuf.io import ChallengeResponseSet
from pypuf.simulation import Simulation

from .eval_utils import evaluate_exposed_metrics


def evaluate_puf_population(
    instances: Iterable[Simulation],
    test_set: Optional[ChallengeResponseSet] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a population and return official and derived metrics."""
    population = list(instances)
    if not population:
        raise ValueError("At least one PUF instance is required.")
    return evaluate_exposed_metrics(
        population[0], instances=population, test_set=test_set, **kwargs
    )


__all__ = ["evaluate_puf_population"]
