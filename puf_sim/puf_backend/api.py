"""Optional NumPy and Intel XPU backend selection for puf_sim."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on the local environment
    torch = None


def xpu_available() -> bool:
    """Return whether PyTorch can access an Intel XPU."""
    if torch is None or not hasattr(torch, "xpu"):
        return False
    try:
        return bool(torch.xpu.is_available())
    except Exception:
        return False


class Backend:
    """Small common interface over NumPy and PyTorch XPU."""

    def __init__(self, name: str, xp: Any, device: int | None = None) -> None:
        self.name = name
        self.xp = xp
        self.device = device

    def asarray(self, value: Any, dtype: Any = None) -> Any:
        if self.name == "torch_xpu":
            return torch.as_tensor(value, dtype=dtype, device=self.device)
        return self.xp.asarray(value, dtype=dtype)

    def asnumpy(self, value: Any) -> np.ndarray:
        if self.name == "torch_xpu":
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def eval_array(self, instance: Any, challenges: Any) -> Any:
        """Evaluate an instance on the selected backend when supported."""
        if self.name == "torch_xpu" and hasattr(instance, "eval_backend"):
            return instance.eval_backend(challenges, self)
        return self.asarray(instance.eval(self.asnumpy(challenges)))


def get_backend(name: str = "numpy", device: int | None = None) -> Backend:
    """Create a NumPy or Intel XPU backend.

    ``torch_xpu`` requires an XPU-enabled PyTorch installation. ``auto``
    selects Intel XPU when available, otherwise NumPy.
    """
    normalized = name.strip().lower()
    if normalized not in {"numpy", "torch_xpu", "auto"}:
        raise ValueError("backend must be 'numpy', 'torch_xpu', or 'auto'.")
    if normalized == "auto":
        normalized = "torch_xpu" if xpu_available() else "numpy"
    if normalized == "torch_xpu":
        if not xpu_available():
            raise RuntimeError(
                "Intel XPU backend requested, but an XPU-enabled PyTorch "
                "installation or Intel GPU is unavailable."
            )
        selected_device = torch.device(f"xpu:{0 if device is None else device}")
        return Backend("torch_xpu", torch, selected_device)
    return Backend("numpy", np, None)


__all__ = ["Backend", "get_backend", "xpu_available"]
