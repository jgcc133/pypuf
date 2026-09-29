"""Arbiter PUF specific model, trainer, and feature transform for the ML attack.

The Arbiter PUF applies the "Arbiter Threshold Transform" (ATT) to each
challenge before computing its delay-based response (see
``pypuf.simulation.base.Simulation.att``). Feeding raw challenge bits to a
generic classifier ignores this transform and leaves the model unable to
represent the PUF's actual decision boundary. This module applies the same
transform to challenges before training and inference.
"""
from __future__ import annotations

from typing import Any

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


FEED_FORWARD_STAGE_NAME = "ff_apuf_structure_aware"


def _att_torch(challenges: Any) -> Any:
    """Differentiable Arbiter Threshold Transform: suffix cumulative product."""
    import torch

    return torch.flip(torch.cumprod(torch.flip(challenges, dims=[-1]), dim=-1), dims=[-1])


class _SignSTE:
    """Straight-through sign(): exact threshold forward, tanh surrogate gradient.

    ``sign()`` has zero gradient almost everywhere, so it cannot otherwise be
    trained through with backpropagation.
    """

    @staticmethod
    def apply(x: Any) -> Any:
        import torch

        class _Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value)
                return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))

            @staticmethod
            def backward(ctx, grad_output):
                (value,) = ctx.saved_tensors
                return grad_output * (1 - torch.tanh(value) ** 2)

        return _Function.apply(x)


class _FeedForwardCore:
    """Reconstructs ``FeedForwardArbiterPUF.val``'s two-segment, sign-gated delay.

    The default loop (see ``puf_sim.puf_implementations.factory``) places one
    arbiter point at ``n // 3`` and its feed point at ``2 * n // 3``. The
    first segment's delay (unknown, learnable weights) is thresholded with
    sign() into a virtual challenge bit, which is spliced into the second
    segment; the second segment's delay is then *multiplicatively gated* by
    the product of all its bits (including the virtual one) before its own
    ATT-linear term is added. Both the gating and the splice position are
    public knowledge; only the two weight vectors are unknown and trained.
    """

    def __new__(cls, n: int, response_bits: int, seed: int) -> Any:
        import torch
        from torch import nn

        torch.manual_seed(seed)
        arbiter_point = max(1, min(n - 1, n // 3))
        feed_point = max(arbiter_point + 1, min(n - 1, (2 * n) // 3))

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.arbiter_point = arbiter_point
                self.feed_point = feed_point
                self.weight_a = nn.Linear(arbiter_point, response_bits, bias=False)
                self.weight_b = nn.Linear(n - arbiter_point + 1, response_bits, bias=False)

            def forward(self, challenges: Any) -> Any:
                segment_a = _att_torch(challenges[:, :self.arbiter_point])
                delay_a = self.weight_a(segment_a)
                virtual_bit = _SignSTE.apply(delay_a)

                outputs = []
                for bit_index in range(delay_a.shape[1]):
                    segment_b_raw = torch.cat([
                        challenges[:, self.arbiter_point:self.feed_point],
                        virtual_bit[:, bit_index:bit_index + 1],
                        challenges[:, self.feed_point:],
                    ], dim=1)
                    gate = torch.prod(segment_b_raw, dim=1, keepdim=True)
                    delay_b = self.weight_b(_att_torch(segment_b_raw))[:, bit_index:bit_index + 1]
                    outputs.append(delay_a[:, bit_index:bit_index + 1] * gate + delay_b)
                return torch.cat(outputs, dim=1)

        return _Module()


class FeedForwardAgentTrainer(_BaseAgentTrainer):
    """Train the structure-aware Feed-Forward Arbiter PUF model directly on raw challenges."""

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

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        self.model = _FeedForwardCore(challenges.shape[1], response_bits, seed).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, foreach=False
        )
        self.loss_function = self.build_loss_function()
        self.training_inputs = torch.as_tensor(challenges, dtype=torch.float32, device=device)
        self.training_labels = self.build_labels(responses, response_bits, device)
        self.batch_size = batch_size


__all__ = [
    "AgentTrainer",
    "TorchPUFModel",
    "STAGE_NAME",
    "feature_map",
    "FeedForwardAgentTrainer",
    "FEED_FORWARD_STAGE_NAME",
]
