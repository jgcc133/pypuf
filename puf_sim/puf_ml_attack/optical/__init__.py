"""Optical PUF specific model, trainer, feature transform, and match metric.

Integrated optical PUFs (``pypuf.simulation.optical.IntegratedOpticalPUF``)
return continuous, non-negative intensity values rather than bipolar +-1
responses. The generic attack model always thresholds its output to +-1, so
it can never equal a continuous target value; every instance imitation rate
comes out as 0%. This module instead trains a regression model on the
unordered pairwise challenge-bit products known to linearize optical PUF
behavior (see ``pypuf.attack.LeastSquaresRegression``), and matches responses
within a floating-point tolerance rather than requiring bit-exact equality.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from pypuf.io import ChallengeResponseSet
from pypuf.simulation import Simulation

from ..common import AgentTrainer as _BaseAgentTrainer, TorchPUFModel as _BaseTorchPUFModel

STAGE_NAME = "optical_regression_ensemble"
BOOTSTRAP_STAGE_NAME = "optical_linear_regression"

# Continuous intensities can never match bit-for-bit; treat values within
# this relative/absolute tolerance as an exact match. The absolute floor
# matters most near-zero intensities, where a tiny relative tolerance would
# otherwise demand near-perfect precision.
RELATIVE_TOLERANCE = 0.10
ABSOLUTE_TOLERANCE = 0.10


def feature_map(challenges: np.ndarray) -> np.ndarray:
    """Map challenges to unordered pairwise products, as used by pypuf's optical PUF attack.

    Equivalent to ``pypuf.attack.LeastSquaresRegression.feature_map_optical_pufs_reloaded_improved``,
    reimplemented here to avoid importing ``pypuf.attack``, which requires TensorFlow.
    """
    challenges = np.asarray(challenges)
    n = challenges.shape[1]
    idx = np.triu_indices(n)
    return np.einsum("...i,...j->...ij", challenges, challenges)[:, idx[0], idx[1]]


_EPSILON = 1e-8


class AgentTrainer(_BaseAgentTrainer):
    """Train a regression model on standardized optical pairwise-product features.

    Optical intensities and their pairwise-product features can have a much
    larger and n-dependent scale than the bipolar +-1 features other PUF
    families use. Without standardization, gradient descent through the
    shared Tanh hidden layer converges very slowly (or not at all) for the
    default learning rate. Standardizing both features and targets here, and
    de-normalizing predictions in ``TorchPUFModel.eval``, keeps training well
    scaled regardless of challenge length.
    """

    @staticmethod
    def feature_map(challenges: np.ndarray) -> np.ndarray:
        return feature_map(challenges)

    @staticmethod
    def build_loss_function() -> Any:
        from torch import nn

        return nn.MSELoss()

    def __init__(
        self,
        challenges: np.ndarray,
        responses: np.ndarray,
        response_bits: int,
        hidden_layers: int,
        width: int,
        seed: int,
        batch_size: int,
        learning_rate: float,
        device: Any,
    ):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)

        mapped_challenges = np.asarray(self.feature_map(challenges), dtype=np.float32)
        feature_count = mapped_challenges.shape[1]
        feature_mean = mapped_challenges.mean(axis=0)
        feature_scale = mapped_challenges.std(axis=0)
        feature_scale[feature_scale < _EPSILON] = 1.0
        normalized_challenges = (mapped_challenges - feature_mean) / feature_scale

        response_matrix = np.asarray(responses, dtype=np.float32).reshape(len(responses), response_bits)
        label_mean = response_matrix.mean(axis=0)
        label_scale = response_matrix.std(axis=0)
        label_scale[label_scale < _EPSILON] = 1.0
        normalized_labels = (response_matrix - label_mean) / label_scale

        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            input_size = feature_count if not layers else width
            layers.extend([nn.Linear(input_size, width), nn.Tanh()])
        input_size = width if hidden_layers else feature_count
        layers.append(nn.Linear(input_size, response_bits))
        self.model = nn.Sequential(*layers).to(device)
        # Stashed on the module so checkpointing (copy.deepcopy) and
        # TorchPUFModel.eval can de-normalize predictions consistently.
        self.model.feature_mean = feature_mean
        self.model.feature_scale = feature_scale
        self.model.label_mean = label_mean
        self.model.label_scale = label_scale

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, foreach=False
        )
        self.loss_function = self.build_loss_function()
        self.training_inputs = torch.as_tensor(
            normalized_challenges, dtype=torch.float32, device=device
        )
        self.training_labels = torch.as_tensor(
            normalized_labels, dtype=torch.float32, device=device
        )
        self.batch_size = batch_size


class TorchPUFModel(_BaseTorchPUFModel):
    """Predict continuous optical intensities, de-normalizing the regression output."""

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        import torch

        reference = self.models[0]
        normalized_features = (
            feature_map(challenges) - reference.feature_mean
        ) / reference.feature_scale
        inputs = torch.as_tensor(normalized_features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predictions = torch.stack([model(inputs) for model in self.models]).mean(dim=0)
        predictions = predictions.cpu().numpy()
        return predictions * reference.label_scale + reference.label_mean


def bootstrap_linear_model(
    challenges: np.ndarray, responses: np.ndarray, response_bits: int, device: Any
) -> Any:
    """Exact closed-form least-squares fit in the pairwise-product feature space.

    Optical PUF intensities are an exact linear function of the pairwise
    challenge-bit products (see module docstring), so this direct solve
    (mirroring ``pypuf.attack.LeastSquaresRegression``) fits the target far
    more reliably than gradient descent through a mandatory nonlinear hidden
    layer, which fights against an already-linear relationship.
    """
    import torch
    from torch import nn

    features = np.asarray(feature_map(challenges), dtype=np.float64)
    response_matrix = np.asarray(responses, dtype=np.float64).reshape(len(responses), response_bits)
    linear_map = np.linalg.pinv(features) @ response_matrix

    linear_layer = nn.Linear(features.shape[1], response_bits, bias=False).to(device)
    with torch.no_grad():
        linear_layer.weight.copy_(
            torch.as_tensor(linear_map.T, dtype=torch.float32, device=device)
        )
    feature_count = features.shape[1]
    linear_layer.feature_mean = np.zeros(feature_count, dtype=np.float32)
    linear_layer.feature_scale = np.ones(feature_count, dtype=np.float32)
    linear_layer.label_mean = np.zeros(response_bits, dtype=np.float32)
    linear_layer.label_scale = np.ones(response_bits, dtype=np.float32)
    return linear_layer


def pair_accuracy(model: Simulation, dataset: ChallengeResponseSet) -> float:
    """Fraction of challenges where every response value matches within tolerance."""
    predicted = np.asarray(model.eval(dataset.challenges), dtype=float)
    expected = np.asarray(dataset.responses, dtype=float).reshape(len(dataset.challenges), -1)
    matches = np.isclose(predicted, expected, rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_TOLERANCE)
    return float(np.mean(np.all(matches, axis=1)))


__all__ = [
    "AgentTrainer",
    "TorchPUFModel",
    "STAGE_NAME",
    "BOOTSTRAP_STAGE_NAME",
    "feature_map",
    "pair_accuracy",
    "bootstrap_linear_model",
]
