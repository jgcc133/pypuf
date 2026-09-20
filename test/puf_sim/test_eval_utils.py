import numpy as np

from pypuf.simulation import Simulation
from puf_sim import evaluate_puf_population
from puf_sim.eval_utils import (
    binary_entropy,
    hamming_distance_distribution,
    probability_of_misidentification,
)


class ConstantPUF(Simulation):
    def __init__(self, value: int) -> None:
        self.value = value

    @property
    def challenge_length(self) -> int:
        return 4

    @property
    def response_length(self) -> int:
        return 1

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        return np.full((len(challenges), 1), self.value, dtype=np.int8)


def test_response_data_helpers():
    responses = np.array([[-1], [1]])
    assert binary_entropy(responses) == 1.0
    np.testing.assert_allclose(hamming_distance_distribution(responses), [1.0])
    assert probability_of_misidentification(responses.reshape(2, 1, 1)) == 0.0


def test_population_evaluation_exposes_official_and_derived_metrics():
    result = evaluate_puf_population(
        [ConstantPUF(-1), ConstantPUF(1)], samples=4, repetitions=3, seed=10
    )

    expected = {
        "uniqueness", "steadiness", "reliability", "uniformity", "randomness",
        "bit_aliasing", "diffuseness", "probability_of_misidentification",
        "similarity", "accuracy", "correlation", "influence", "total_influence",
        "noise_sensitivity",
    }
    assert expected.issubset(result)
    assert result["steadiness"].shape == (4, 1)
    assert result["reliability"].shape == (4, 1)
    assert result["diffuseness"] == 0.0
    assert result["probability_of_misidentification"] == 0.0


def test_single_device_population_has_empty_pair_metrics():
    result = evaluate_puf_population([ConstantPUF(1)], samples=2, repetitions=2)
    assert result["uniqueness"].size == 0
    assert result["similarity"] is None
    assert result["probability_of_misidentification"] == 0.0
