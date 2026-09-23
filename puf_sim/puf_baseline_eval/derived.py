"""Derived response-data metrics not directly exposed by pypuf."""
from __future__ import annotations

import numpy as np

from puf_sim.puf_backend import Backend, get_backend


def response_matrix(responses: np.ndarray, backend: Backend | None = None):
    """Return responses as ``(samples, response_bits)`` in ``{-1, 1}``."""
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim == 3:
        values = xp.sign(xp.mean(values, axis=-1))
    if values.ndim != 2:
        raise ValueError("Responses must have one, two, or three dimensions.")
    return backend.astype(xp.where(values >= 0, 1, -1), xp.int8)


def binary_entropy(responses: np.ndarray, backend: Backend | None = None) -> float:
    """Return mean Shannon entropy of response bits, normalized to ``[0, 1]``."""
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = response_matrix(responses, backend)
    probability_one = xp.mean(
        backend.astype(values == 1, xp.float32),
        axis=0,
    )
    terms = xp.zeros_like(probability_one, dtype=xp.float32)
    non_deterministic = (probability_one > 0) & (probability_one < 1)
    probability = probability_one[non_deterministic]
    terms[non_deterministic] = -(
        probability * xp.log2(probability)
        + (1 - probability) * xp.log2(1 - probability)
    )
    return float(backend.asnumpy(xp.mean(terms)))


def hamming_distance_distribution(responses: np.ndarray, backend: Backend | None = None) -> np.ndarray:
    """Return normalized pairwise Hamming distances.

    Two-dimensional input compares response rows directly. Three-dimensional
    population input is shaped ``(devices, challenges, response_bits)`` and
    returns one distance per device pair, averaged over challenges and bits.
    """
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim == 3:
        values = backend.astype(xp.where(values >= 0, 1, -1), xp.float32)
        pair_distances = []
        for left_index in range(values.shape[0] - 1):
            right_values = values[left_index + 1:]
            distances = xp.mean(
                (1.0 - right_values * values[left_index]) / 2.0,
                axis=(1, 2),
            )
            pair_distances.append(distances)
        if not pair_distances:
            return np.array([], dtype=float)
        return backend.asnumpy(xp.concatenate(pair_distances))

    values = response_matrix(values, backend)
    values = backend.astype(values, xp.float32)
    pair_distances = []
    for left_index in range(values.shape[0] - 1):
        distances = xp.mean(
            (1.0 - values[left_index + 1:] * values[left_index]) / 2.0,
            axis=1,
        )
        pair_distances.append(distances)
    if not pair_distances:
        return np.array([], dtype=float)
    return backend.asnumpy(xp.concatenate(pair_distances))


def bit_aliasing(responses: np.ndarray, backend: Backend | None = None) -> float:
    """Return mean absolute response-bit bias; zero is ideal.

    ``responses`` must have shape ``(devices, challenges, response_bits)``.
    """
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Bit aliasing expects (devices, challenges, response_bits).")
    values = backend.astype(xp.where(values >= 0, 1, -1), xp.float32)
    return float(backend.asnumpy(xp.mean(xp.abs(xp.mean(values, axis=(0, 1))))))


def probability_of_misidentification(responses: np.ndarray, backend: Backend | None = None) -> float:
    """Return the average complete-response impostor collision probability."""
    backend = backend or get_backend("numpy")
    xp = backend.xp
    values = backend.asarray(responses)
    if values.ndim != 3:
        raise ValueError("Misidentification expects (devices, challenges, response_bits).")
    pair_rates = []
    for left_index in range(values.shape[0] - 1):
        matches = xp.all(values[left_index + 1:] == values[left_index], axis=2)
        pair_rates.append(xp.mean(backend.astype(matches, xp.float32), axis=1))
    if not pair_rates:
        return 0.0
    return float(backend.asnumpy(xp.mean(xp.concatenate(pair_rates))))


__all__ = [
    "binary_entropy",
    "bit_aliasing",
    "hamming_distance_distribution",
    "probability_of_misidentification",
    "response_matrix",
]
