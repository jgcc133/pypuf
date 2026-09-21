"""Adapters for PUF families already implemented by pypuf."""
from pypuf.simulation import (
    ArbiterPUF,
    FeedForwardArbiterPUF,
    IntegratedOpticalPUF,
    InterposePUF,
    PermutationPUF,
    XORArbiterPUF,
)

OpticalPUF = IntegratedOpticalPUF

__all__ = [
    "ArbiterPUF",
    "FeedForwardArbiterPUF",
    "InterposePUF",
    "OpticalPUF",
    "PermutationPUF",
    "XORArbiterPUF",
]
