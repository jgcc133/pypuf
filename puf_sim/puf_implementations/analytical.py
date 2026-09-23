"""Analytical models for PUF families not implemented by pypuf."""
from __future__ import annotations

import numpy as np

from pypuf.simulation import Simulation

from puf_sim.puf_backend import Backend


class RingOscillatorPUF(Simulation):
    """Analytical ring-oscillator PUF based on challenge-selected frequencies."""

    def __init__(self, n: int, response_bits: int = 1, seed: int = 0) -> None:
        self.n = n
        self.m = response_bits
        rng = np.random.default_rng(seed)
        self.frequencies = rng.normal(0.0, 1.0, size=(n, response_bits))
        self.offsets = rng.normal(0.0, 0.25, size=response_bits)

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        values = np.asarray(challenges, dtype=float) @ self.frequencies + self.offsets
        return np.where(values >= 0, 1, -1).astype(np.int8)

    def eval_backend(self, challenges: np.ndarray, backend: Backend):
        xp = backend.xp
        values = backend.asarray(challenges, dtype=xp.float32) @ backend.asarray(
            self.frequencies,
            dtype=xp.float32,
        )
        values = values + backend.asarray(self.offsets, dtype=xp.float32)
        return backend.astype(xp.where(values >= 0, 1, -1), xp.int8)


class StaticRandomAccessMemoryPUF(Simulation):
    """Analytical SRAM PUF represented by a device-specific power-up state."""

    def __init__(self, n: int, response_bits: int = 1, seed: int = 0) -> None:
        self.n = n
        self.m = response_bits
        rng = np.random.default_rng(seed)
        self.power_up_state = rng.choice((-1, 1), size=response_bits).astype(np.int8)

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        count = np.asarray(challenges).shape[0]
        return np.broadcast_to(self.power_up_state, (count, self.m)).copy()

    def eval_backend(self, challenges: np.ndarray, backend: Backend):
        xp = backend.xp
        count = challenges.shape[0]
        values = xp.broadcast_to(backend.asarray(self.power_up_state), (count, self.m))
        return values.clone() if backend.name == "torch_xpu" else values.copy()


class MemristivePUF(Simulation):
    """Analytical memristive PUF using device-specific conductance thresholds."""

    def __init__(self, n: int, response_bits: int = 1, seed: int = 0) -> None:
        self.n = n
        self.m = response_bits
        rng = np.random.default_rng(seed)
        self.conductance = rng.normal(0.0, 1.0, size=(n, response_bits))
        self.threshold = rng.normal(0.0, 0.25, size=response_bits)

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        values = np.asarray(challenges, dtype=float) @ self.conductance
        values = np.tanh(values) - self.threshold
        return np.where(values >= 0, 1, -1).astype(np.int8)

    def eval_backend(self, challenges: np.ndarray, backend: Backend):
        xp = backend.xp
        values = backend.asarray(challenges, dtype=xp.float32) @ backend.asarray(
            self.conductance,
            dtype=xp.float32,
        )
        values = xp.tanh(values) - backend.asarray(self.threshold, dtype=xp.float32)
        return backend.astype(xp.where(values >= 0, 1, -1), xp.int8)


class SiliconPhotonicPUF(Simulation):
    """Analytical silicon-photonic PUF based on a complex transfer matrix."""

    def __init__(self, n: int, response_bits: int = 1, seed: int = 0) -> None:
        self.n = n
        self.m = response_bits
        rng = np.random.default_rng(seed)
        amplitude = rng.uniform(0.1, 1.0, size=(n, response_bits))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, response_bits))
        self.transfer_matrix = amplitude * np.exp(1j * phase)
        self.reference = rng.normal(0.0, 0.25, size=response_bits)

    @property
    def challenge_length(self) -> int:
        return self.n

    @property
    def response_length(self) -> int:
        return self.m

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        optical_field = np.asarray(challenges, dtype=float) @ self.transfer_matrix
        intensity = np.abs(optical_field) ** 2
        return np.where(intensity >= self.reference, 1, -1).astype(np.int8)

    def eval_backend(self, challenges: np.ndarray, backend: Backend):
        xp = backend.xp
        optical_field = backend.asarray(challenges, dtype=xp.complex64) @ backend.asarray(
            self.transfer_matrix,
            dtype=xp.complex64,
        )
        intensity = xp.abs(optical_field) ** 2
        return backend.astype(
            xp.where(intensity >= backend.asarray(self.reference, dtype=xp.float32), 1, -1),
            xp.int8,
        )


__all__ = [
    "MemristivePUF",
    "RingOscillatorPUF",
    "SiliconPhotonicPUF",
    "StaticRandomAccessMemoryPUF",
]
