"""Consolidated calls to the public :mod:`pypuf.metrics` API."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from pypuf.io import ChallengeResponseSet
from pypuf.metrics import (
    accuracy,
    bias,
    correlation,
    influence,
    noise_sensitivity,
    reliability,
    similarity,
    total_influence,
    uniqueness,
)
from pypuf.simulation import Simulation


def evaluate_exposed_api(
    instance: Simulation,
    instances: Optional[Iterable[Simulation]] = None,
    test_set: Optional[ChallengeResponseSet] = None,
    seed: int = 0,
    samples: int = 1000,
    repetitions: int = 17,
    input_noise: float = 0.01,
) -> Dict[str, Any]:
    """Call every applicable metric exposed by ``pypuf.metrics``.

    Returned keys correspond directly to the public API names: ``bias``,
    ``reliability``, ``uniqueness``, ``similarity``, ``accuracy``,
    ``correlation``, ``influence``, ``total_influence``, and
    ``noise_sensitivity``. Metrics requiring a comparison object return
    ``None`` when that object is not supplied.
    """
    population = list(instances or [instance])
    if not population:
        raise ValueError("At least one PUF instance is required.")

    result: Dict[str, Any] = {
        "bias": np.asarray(bias(instance, seed=seed + 1, N=samples)),
        "reliability": np.asarray(
            reliability(instance, seed=seed + 3, N=samples, r=repetitions)
        ),
        "uniqueness": np.array([], dtype=float),
        "similarity": None,
        "accuracy": None,
        "correlation": None,
    }
    if len(population) >= 2:
        result["uniqueness"] = np.asarray(
            uniqueness(population, seed=seed + 2, N=samples)
        )
        result["similarity"] = np.asarray(
            similarity(population[0], population[1], seed=seed + 6, N=samples)
        )

    influence_values = [
        influence(instance, i=index, seed=seed + 10 + index, N=samples)
        for index in range(instance.challenge_length)
    ]
    result["influence"] = np.asarray(influence_values)
    result["total_influence"] = float(
        total_influence(instance, seed=seed + 4, N=samples)
    )
    result["noise_sensitivity"] = float(
        noise_sensitivity(
            instance,
            eps=input_noise,
            seed=seed + 5,
            N=samples,
        )
    )

    if test_set is not None:
        result["accuracy"] = np.asarray(accuracy(instance, test_set))
        result["correlation"] = np.asarray(correlation(instance, test_set))

    return result


__all__ = ["evaluate_exposed_api"]
