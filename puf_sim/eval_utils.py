"""Shared PUF evaluation utilities.

The module keeps official :mod:`pypuf.metrics` calls together with derived
response-data metrics that are useful when evaluating a PUF population.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from pypuf.io import ChallengeResponseSet, random_inputs
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


def response_matrix(responses: np.ndarray) -> np.ndarray:
    """Return responses as ``(samples, response_bits)`` in ``{-1, 1}``."""
    values = np.asarray(responses)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim == 3:
        values = np.sign(np.mean(values, axis=-1))
    if values.ndim != 2:
        raise ValueError("Responses must have one, two, or three dimensions.")
    return np.where(values >= 0, 1, -1).astype(np.int8)


def binary_entropy(responses: np.ndarray) -> float:
    """Return mean Shannon entropy of response bits, normalized to ``[0, 1]``."""
    values = response_matrix(responses)
    probability_one = np.mean(values == 1, axis=0)
    terms = np.zeros_like(probability_one, dtype=float)
    non_deterministic = (probability_one > 0) & (probability_one < 1)
    probability = probability_one[non_deterministic]
    terms[non_deterministic] = -(
        probability * np.log2(probability)
        + (1 - probability) * np.log2(1 - probability)
    )
    return float(np.mean(terms))


def hamming_distance_distribution(responses: np.ndarray) -> np.ndarray:
    """Return normalized pairwise Hamming distances between response rows."""
    values = response_matrix(responses)
    return np.asarray(
        [np.mean(left != right) for left, right in combinations(values, 2)],
        dtype=float,
    )


def bit_aliasing(responses: np.ndarray) -> float:
    """Return mean absolute response-bit bias; zero is ideal.

    ``responses`` should be shaped ``(devices, challenges, bits)``. The
    device and challenge axes are flattened before measuring each bit.
    """
    values = np.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Bit aliasing expects (devices, challenges, response_bits).")
    values = np.where(values >= 0, 1, -1)
    return float(np.mean(np.abs(np.mean(values, axis=(0, 1)))))


def probability_of_misidentification(responses: np.ndarray) -> float:
    """Return the probability an impostor matches a complete response.

    ``responses`` must have shape ``(devices, challenges, response_bits)``.
    The result averages complete-response collisions over all device pairs
    and challenges; zero is ideal.
    """
    values = np.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Misidentification expects (devices, challenges, response_bits).")
    pair_rates = [
        float(np.mean(np.all(values[left] == values[right], axis=1)))
        for left, right in combinations(range(values.shape[0]), 2)
    ]
    return float(np.mean(pair_rates)) if pair_rates else 0.0


def evaluate_exposed_metrics(
    instance: Simulation,
    instances: Optional[Iterable[Simulation]] = None,
    test_set: Optional[ChallengeResponseSet] = None,
    seed: int = 0,
    samples: int = 1000,
    repetitions: int = 17,
    input_noise: float = 0.01,
) -> Dict[str, Any]:
    """Evaluate official pypuf metrics plus derived PUF quality metrics."""
    population = list(instances or [instance])
    if not population:
        raise ValueError("At least one PUF instance is required.")

    challenges = random_inputs(instance.challenge_length, samples, seed)
    population_responses = np.stack(
        [response_matrix(puf.eval(challenges)) for puf in population],
        axis=0,
    )
    result: Dict[str, Any] = {
        "bias": np.asarray(bias(instance, seed=seed + 1, N=samples)),
        "uniformity": 1.0,
        "randomness": binary_entropy(population_responses.reshape(-1, instance.response_length)),
        "bit_aliasing": bit_aliasing(population_responses),
        "probability_of_misidentification": probability_of_misidentification(population_responses),
        "hamming_distance_distribution": hamming_distance_distribution(
            population_responses.reshape(-1, instance.response_length)
        ),
    }
    result["uniformity"] = 1.0 - float(np.mean(np.abs(result["bias"])))

    if len(population) >= 2:
        result["uniqueness"] = np.asarray(uniqueness(population, seed=seed + 2, N=samples))
    else:
        result["uniqueness"] = np.array([], dtype=float)

    reliability_values = np.asarray(reliability(instance, seed=seed + 3, N=samples, r=repetitions))
    result["reliability"] = reliability_values
    result["steadiness"] = reliability_values

    influence_values = np.asarray([
        influence(instance, i=index, seed=seed + 10 + index, N=samples)
        for index in range(instance.challenge_length)
    ])
    result["influence"] = influence_values
    result["total_influence"] = float(total_influence(instance, seed=seed + 4, N=samples))
    result["diffuseness"] = result["total_influence"] / max(1, instance.challenge_length)
    result["noise_sensitivity"] = float(
        noise_sensitivity(instance, eps=input_noise, seed=seed + 5, N=samples)
    )

    if test_set is not None:
        result["accuracy"] = np.asarray(accuracy(instance, test_set))
        result["correlation"] = np.asarray(correlation(instance, test_set))
    else:
        result["accuracy"] = None
        result["correlation"] = None

    if len(population) >= 2:
        result["similarity"] = np.asarray(similarity(population[0], population[1], seed=seed + 6, N=samples))
    else:
        result["similarity"] = None
    return result


__all__ = [
    "accuracy",
    "binary_entropy",
    "bias",
    "bit_aliasing",
    "correlation",
    "diffuseness",
    "evaluate_exposed_metrics",
    "hamming_distance_distribution",
    "influence",
    "noise_sensitivity",
    "probability_of_misidentification",
    "reliability",
    "similarity",
    "total_influence",
    "uniqueness",
]
