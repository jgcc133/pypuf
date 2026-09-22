"""Combined baseline report for official and derived PUF metrics."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

from pypuf.io import ChallengeResponseSet, random_inputs
from pypuf.simulation import Simulation

from puf_sim.puf_backend import get_backend

from .derived import (
    binary_entropy,
    bit_aliasing,
    hamming_distance_distribution,
    probability_of_misidentification,
    response_matrix,
)
from .exposed import evaluate_exposed_api


def evaluate_exposed_metrics(
    instance: Simulation,
    instances: Optional[Iterable[Simulation]] = None,
    test_set: Optional[ChallengeResponseSet] = None,
    seed: int = 0,
    samples: int = 1000,
    repetitions: int = 17,
    input_noise: float = 0.01,
    backend: str = "numpy",
    device: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Return one report containing official and derived evaluation metrics."""
    population = list(instances or [instance])
    if not population:
        raise ValueError("At least one PUF instance is required.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least one or None.")

    selected_backend = get_backend(backend, device=device)
    xp = selected_backend.xp
    challenges = random_inputs(instance.challenge_length, samples, seed)
    population_responses = []
    step = batch_size or len(population)
    for start in range(0, len(population), step):
        population_responses.extend(
            selected_backend.eval_array(puf, challenges)
            for puf in population[start:start + step]
        )
    population_responses = xp.stack(
        [response_matrix(response, selected_backend) for response in population_responses],
        axis=0,
    )
    result = evaluate_exposed_api(
        instance,
        instances=population,
        test_set=test_set,
        seed=seed,
        samples=samples,
        repetitions=repetitions,
        input_noise=input_noise,
    )
    result.update({
        "uniformity": 1.0 - float(np.mean(np.abs(result["bias"]))),
        "randomness": binary_entropy(
            population_responses.reshape(-1, instance.response_length),
            selected_backend,
        ),
        "bit_aliasing": bit_aliasing(population_responses, selected_backend),
        "probability_of_misidentification": probability_of_misidentification(
            population_responses,
            selected_backend,
        ),
        "hamming_distance_distribution": hamming_distance_distribution(
            population_responses.reshape(-1, instance.response_length),
            selected_backend,
        ),
        "backend": selected_backend.name,
        "device": selected_backend.device,
        "batch_size": step,
    })
    result["steadiness"] = result["reliability"]
    result["diffuseness"] = result["total_influence"] / max(
        1, instance.challenge_length
    )
    return result


__all__ = ["evaluate_exposed_metrics"]
