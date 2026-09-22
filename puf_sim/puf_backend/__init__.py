"""Optional NumPy/Intel XPU backend selection for puf_sim."""
from .api import Backend, get_backend, xpu_available

__all__ = ["Backend", "get_backend", "xpu_available"]
