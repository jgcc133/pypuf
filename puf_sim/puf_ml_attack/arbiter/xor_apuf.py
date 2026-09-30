"""Parity attack model and trainer for XOR Arbiter PUFs."""
from __future__ import annotations

from typing import Any

import numpy as np
from pypuf.simulation import ArbiterPUF

from ..common import TorchPUFModel

STAGE_NAME = "xor_arbiter_parity_ensemble"


def feature_map(challenges: np.ndarray) -> np.ndarray:
    """Apply the Arbiter Threshold Transform used by each XOR chain."""
    return ArbiterPUF.transform_atf(np.asarray(challenges), 1)[:, 0, :].copy()


class _XorParityModel:
    """Predict parity from one learned linear threshold per arbiter chain."""

    def __new__(cls, feature_count: int, chain_count: int, seed: int) -> Any:
        import torch
        from torch import nn

        torch.manual_seed(seed)

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.chain_logits = nn.Linear(feature_count, chain_count)

            def forward(self, features: Any) -> Any:
                chain_scores = torch.tanh(self.chain_logits(features))
                return torch.prod(chain_scores, dim=1, keepdim=True)

        return _Module()


class AgentTrainer:
    """Train component Arbiter chains with a differentiable parity objective."""

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
        chain_count: int,
    ):
        import torch

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        features = np.asarray(feature_map(challenges), dtype=np.float32)
        self.model = _XorParityModel(features.shape[1], chain_count, seed).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, foreach=False
        )
        response_matrix = np.asarray(responses).reshape(len(responses), -1)
        if response_matrix.shape[1] != 1:
            raise ValueError("XOR Arbiter PUF attacks currently require one response bit.")
        self.training_inputs = torch.as_tensor(features, dtype=torch.float32, device=device)
        self.training_labels = torch.as_tensor(
            response_matrix, dtype=torch.float32, device=device
        )
        self.batch_size = batch_size

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        order = torch.randperm(len(self.training_inputs), device=self.training_inputs.device)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            scores = self.model(self.training_inputs[indices])
            loss = 1.0 - torch.mean(self.training_labels[indices] * scores)
            loss.backward()
            self.optimizer.step()


class XorTorchPUFModel(TorchPUFModel):
    """Threshold differentiable XOR parity scores to bipolar responses."""

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        return super().eval(feature_map(challenges))


__all__ = ["AgentTrainer", "XorTorchPUFModel", "STAGE_NAME", "feature_map"]