"""Evaluation utilities for pypuf modeling attacks."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
from memory_profiler import memory_usage

from pypuf.io import ChallengeResponseSet
from pypuf.metrics import accuracy, correlation, similarity
from pypuf.simulation import Simulation


def evaluate_attack(
    attack: Any,
    target: Simulation,
    test_set: Optional[ChallengeResponseSet] = None,
    seed: int = 0,
    samples: int = 1000,
) -> Dict[str, Any]:
    """Fit an attack and collect quality, runtime, memory, and history measures."""
    evaluation_set = test_set or ChallengeResponseSet.from_simulation(
        target, N=samples, seed=seed
    )
    start = time.perf_counter()
    peak_memory, model = memory_usage(
        (attack.fit, ()), interval=0.1, include_children=True,
        retval=True, max_usage=True,
    )
    elapsed = time.perf_counter() - start
    if model is None:
        raise RuntimeError("The attack did not produce a model.")

    result: Dict[str, Any] = {
        "model": model,
        "fit_elapsed_seconds": float(elapsed),
        "peak_memory_mb": float(peak_memory),
        "accuracy": np.asarray(accuracy(model, evaluation_set)),
        "correlation": np.asarray(correlation(model, evaluation_set)),
        "similarity": np.asarray(similarity(target, model, seed=seed + 1, N=samples)),
    }
    history = getattr(attack, "history", None) or {}
    result["history"] = history
    result["epochs"] = len(history.get("loss", []))
    for key in ("loss", "val_loss", "accuracy", "val_accuracy"):
        values = history.get(key, [])
        result[f"final_{key}"] = float(values[-1]) if values else None
    return result


__all__ = ["evaluate_attack"]
