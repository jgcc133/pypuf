"""Probe optional Intel XPU availability and execute a small operation."""
from __future__ import annotations

import numpy as np

from .api import get_backend, xpu_available


def main() -> None:
    print(f"xpu_available={xpu_available()}")
    if not xpu_available():
        print("No Intel XPU backend is available.")
        return
    backend = get_backend("torch_xpu")
    values = backend.asarray(np.arange(1_000_000, dtype=np.float32))
    total = backend.xp.sum(values * values)
    print(f"backend={backend.name}")
    print(f"device={backend.device}")
    print(f"device_total={float(backend.asnumpy(total)):.1f}")


if __name__ == "__main__":
    main()
