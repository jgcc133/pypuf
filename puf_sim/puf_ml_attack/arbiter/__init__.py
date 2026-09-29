"""Arbiter PUF specific model, trainer, and feature transform for the ML attack.

The Arbiter PUF applies the "Arbiter Threshold Transform" (ATT) to each
challenge before computing its delay-based response (see
``pypuf.simulation.base.Simulation.att``). Feeding raw challenge bits to a
generic classifier ignores this transform and leaves the model unable to
represent the PUF's actual decision boundary. This module applies the same
transform to challenges before training and inference.
"""
from __future__ import annotations

import numpy as np

from pypuf.simulation import ArbiterPUF

from ..common import AgentTrainer as _BaseAgentTrainer, TorchPUFModel as _BaseTorchPUFModel

STAGE_NAME = "arbiter_mlp_ensemble"


def feature_map(challenges: np.ndarray) -> np.ndarray:
    """Apply the Arbiter Threshold Transform used by pypuf's ArbiterPUF."""
    return ArbiterPUF.transform_atf(np.asarray(challenges), 1)[:, 0, :]


class AgentTrainer(_BaseAgentTrainer):
    """Train on Arbiter Threshold Transformed challenge features."""

    @staticmethod
    def feature_map(challenges: np.ndarray) -> np.ndarray:
        return feature_map(challenges)


class TorchPUFModel(_BaseTorchPUFModel):
    """Apply the Arbiter Threshold Transform before bipolar model inference."""

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        return super().eval(feature_map(challenges))


__all__ = ["AgentTrainer", "TorchPUFModel", "STAGE_NAME", "feature_map"]
