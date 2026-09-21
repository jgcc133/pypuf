"""Derived response-data metrics not directly exposed by pypuf."""
from __future__ import annotations

from itertools import combinations

import numpy as np


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

    ``responses`` must have shape ``(devices, challenges, response_bits)``.
    """
    values = np.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Bit aliasing expects (devices, challenges, response_bits).")
    values = np.where(values >= 0, 1, -1)
    return float(np.mean(np.abs(np.mean(values, axis=(0, 1)))))


def probability_of_misidentification(responses: np.ndarray) -> float:
    """Return the average complete-response impostor collision probability."""
    values = np.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Misidentification expects (devices, challenges, response_bits).")
    pair_rates = [
        float(np.mean(np.all(values[left] == values[right], axis=1)))
        for left, right in combinations(range(values.shape[0]), 2)
    ]
    return float(np.mean(pair_rates)) if pair_rates else 0.0


__all__ = [
    "binary_entropy",
    "bit_aliasing",
    "hamming_distance_distribution",
    "probability_of_misidentification",
    "response_matrix",
]
