"""GPU-aware adapters for PUF families already implemented by pypuf."""
from __future__ import annotations

from typing import Any

import numpy as np

from puf_sim.puf_backend import Backend
from pypuf.simulation import (
    ArbiterPUF as _ArbiterPUF,
    FeedForwardArbiterPUF as _FeedForwardArbiterPUF,
    IntegratedOpticalPUF as _IntegratedOpticalPUF,
    InterposePUF as _InterposePUF,
    PermutationPUF as _PermutationPUF,
    XORArbiterPUF as _XORArbiterPUF,
)


def _att_backend(challenges: Any, xp: Any) -> Any:
    """Apply the arbiter threshold transform to ``(N, k, n)`` tensors."""
    return xp.flip(xp.cumprod(xp.flip(challenges, dims=(2,)), dim=2), dims=(2,))


def _noise_backend(instance: Any, backend: Backend, shape: tuple[int, ...], sigma: float) -> Any:
    noise = instance.random.normal(loc=0, scale=sigma, size=shape)
    return backend.asarray(noise, dtype=backend.xp.float32)


class _XORBackendMixin:
    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenges = backend.asarray(challenges, dtype=xp.float32)
        sub_challenges = _att_backend(
            challenges[:, None, :].expand(-1, self.k, -1),
            xp,
        )
        weights = backend.asarray(self.weight_array[:, :-1], dtype=xp.float32)
        values = xp.einsum("nki,ki->nk", sub_challenges, weights)
        values = values + backend.asarray(self.weight_array[:, -1], dtype=xp.float32)
        values = values + _noise_backend(self, backend, (challenges.shape[0], self.k), self.sigma_noise)
        responses = xp.sign(values)
        return backend.astype(xp.prod(responses, dim=1), xp.int8)


class XORArbiterPUF(_XORBackendMixin, _XORArbiterPUF):
    pass


class ArbiterPUF(_XORBackendMixin, _ArbiterPUF):
    pass


class FeedForwardArbiterPUF(_FeedForwardArbiterPUF):
    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenges = backend.asarray(challenges, dtype=xp.float32)
        count = challenges.shape[0]
        loops = sorted(self.ff, key=lambda loop: loop[0])
        feed_points = {feed_point for _, feed_point in loops}
        ff_challenges = xp.zeros((count, self.n + len(loops)), dtype=xp.float32, device=challenges.device)
        offset = 0
        for index in range(self.n + len(loops)):
            if index in feed_points:
                offset += 1
            else:
                ff_challenges[:, index] = challenges[:, index - offset]

        delay_difference = xp.zeros((count,), dtype=xp.float32, device=challenges.device)
        delay_difference_pos = 0
        for arbiter_point, feed_point in loops + [(self.n + len(loops), None)]:
            section = ff_challenges[:, delay_difference_pos:arbiter_point]
            transformed = _att_backend(section[:, None, :], xp)[:, 0, :]
            weights = backend.asarray(
                self.weight_array[:, delay_difference_pos:arbiter_point],
                dtype=xp.float32,
            )
            section_value = xp.sum(transformed * weights, dim=1)
            section_noise = np.random.default_rng(self.noise_prng.integers(2**32)).normal(
                loc=0,
                scale=self.sigma_noise_from_random_weights(
                    n=(arbiter_point or self.n) - delay_difference_pos,
                    sigma_weight=1,
                    noisiness=self.noisiness,
                ),
                size=(count,),
            )
            section_value = section_value + backend.asarray(section_noise, dtype=xp.float32)
            delay_difference = delay_difference * xp.prod(section, dim=1) + section_value
            delay_difference_pos = arbiter_point
            if feed_point:
                ff_challenges[:, feed_point] = xp.sign(delay_difference)

        return backend.astype(xp.sign(delay_difference), xp.int8)


class PermutationPUF(_XORBackendMixin, _PermutationPUF):
    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenges = backend.asarray(challenges, dtype=xp.float32)
        seeds = self.FIXED_PERMUTATION_SEEDS[challenges.shape[1]]
        if self.k > len(seeds):
            raise AssertionError(
                f"Fixed permutation for n={challenges.shape[1]} currently supports k<={len(seeds)}."
            )
        transformed = xp.stack(
            [challenges[:, np.random.RandomState(seed).permutation(challenges.shape[1])] for seed in seeds[:self.k]],
            dim=1,
        )
        return self._eval_transformed_backend(transformed, backend)

    def _eval_transformed_backend(self, transformed: Any, backend: Backend) -> Any:
        xp = backend.xp
        transformed = _att_backend(transformed, xp)
        weights = backend.asarray(self.weight_array[:, :-1], dtype=xp.float32)
        values = xp.einsum("nki,ki->nk", transformed, weights)
        values = values + backend.asarray(self.weight_array[:, -1], dtype=xp.float32)
        values = values + _noise_backend(self, backend, (transformed.shape[0], self.k), self.sigma_noise)
        return backend.astype(xp.prod(xp.sign(values), dim=1), xp.int8)


class InterposePUF(_InterposePUF):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.up = self._native_child(self.up)
        self.down = self._native_child(self.down)

    @staticmethod
    def _native_child(child: Any) -> XORArbiterPUF:
        native_child = XORArbiterPUF.__new__(XORArbiterPUF)
        native_child.__dict__.update(child.__dict__)
        return native_child

    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        challenges = backend.asarray(challenges, dtype=xp.float32)
        interpose_bits = self.up.eval_backend(challenges, backend).reshape(challenges.shape[0], 1)
        down_challenges = xp.cat(
            (challenges[:, :self.interpose_pos], interpose_bits, challenges[:, self.interpose_pos:]),
            dim=1,
        )
        return self.down.eval_backend(down_challenges, backend)


class OpticalPUF(_IntegratedOpticalPUF):
    def eval_backend(self, challenges: Any, backend: Backend) -> Any:
        xp = backend.xp
        values = backend.asarray(challenges, dtype=xp.float32)
        transfer = backend.asarray(self.T, dtype=xp.complex64)
        return xp.abs(values.to(dtype=xp.complex64) @ transfer) ** 2

__all__ = [
    "ArbiterPUF",
    "FeedForwardArbiterPUF",
    "InterposePUF",
    "OpticalPUF",
    "PermutationPUF",
    "XORArbiterPUF",
]
