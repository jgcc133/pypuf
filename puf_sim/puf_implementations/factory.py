"""Factory and registry for the supported PUF families."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Type

from pypuf.simulation import Simulation

from .analytical import (
    MemristivePUF,
    RingOscillatorPUF,
    SiliconPhotonicPUF,
    StaticRandomAccessMemoryPUF,
)
from .native import (
    ArbiterPUF,
    FeedForwardArbiterPUF,
    InterposePUF,
    OpticalPUF,
    PermutationPUF,
    XORArbiterPUF,
)

PUF_IMPLEMENTATIONS: Dict[str, Type[Simulation]] = {
    "optical": OpticalPUF,
    "arbiter": ArbiterPUF,
    "feed_forward_arbiter": FeedForwardArbiterPUF,
    "ff_apuf": FeedForwardArbiterPUF,
    "xor_arbiter": XORArbiterPUF,
    "xor_apuf": XORArbiterPUF,
    "interpose": InterposePUF,
    "permutation": PermutationPUF,
    "ring_oscillator": RingOscillatorPUF,
    "ro_puf": RingOscillatorPUF,
    "sram": StaticRandomAccessMemoryPUF,
    "static_random_access_memory": StaticRandomAccessMemoryPUF,
    "memristive": MemristivePUF,
    "memristive_puf": MemristivePUF,
    "silicon_photonic": SiliconPhotonicPUF,
}


def create_puf(
    family: str,
    n: int,
    seed: int = 0,
    k: int = 1,
    response_bits: int = 1,
    noisiness: float = 0.0,
    ff: Optional[Sequence[Tuple[int, int]]] = None,
    interpose_pos: Optional[int] = None,
) -> Simulation:
    """Create a supported PUF using a common constructor interface."""
    key = family.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in PUF_IMPLEMENTATIONS:
        supported = ", ".join(sorted(PUF_IMPLEMENTATIONS))
        raise ValueError(f"Unknown PUF family {family!r}; supported names: {supported}")

    if key == "optical":
        return OpticalPUF(n=n, m=response_bits, seed=seed)
    if key == "arbiter":
        return ArbiterPUF(n=n, seed=seed, noisiness=noisiness)
    if key in {"feed_forward_arbiter", "ff_apuf"}:
        loops = list(ff) if ff is not None else [(n // 3, (2 * n) // 3)]
        return FeedForwardArbiterPUF(n=n, ff=loops, seed=seed, noisiness=noisiness)
    if key in {"xor_arbiter", "xor_apuf"}:
        return XORArbiterPUF(n=n, k=k, seed=seed, noisiness=noisiness)
    if key == "interpose":
        return InterposePUF(
            n=n,
            k_down=k,
            k_up=1,
            interpose_pos=interpose_pos or n // 2,
            seed=seed,
            noisiness=noisiness,
        )
    if key == "permutation":
        return PermutationPUF(n=n, k=k, seed=seed, noisiness=noisiness)
    if key in {"ring_oscillator", "ro_puf"}:
        return RingOscillatorPUF(n=n, response_bits=response_bits, seed=seed)
    if key in {"sram", "static_random_access_memory"}:
        return StaticRandomAccessMemoryPUF(n=n, response_bits=response_bits, seed=seed)
    if key in {"memristive", "memristive_puf"}:
        return MemristivePUF(n=n, response_bits=response_bits, seed=seed)
    if key == "silicon_photonic":
        return SiliconPhotonicPUF(n=n, response_bits=response_bits, seed=seed)

    raise AssertionError("PUF family registry and factory are inconsistent")


__all__ = ["PUF_IMPLEMENTATIONS", "create_puf"]
