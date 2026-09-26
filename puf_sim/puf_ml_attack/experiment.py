"""Train progressively more expressive models against PUF implementations."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from pypuf.io import ChallengeResponseSet
from pypuf.simulation import Simulation

from puf_sim.puf_implementations import PUF_IMPLEMENTATIONS, create_puf

from .writer import save_ml_attack_report

DEFAULT_ATTACK_FAMILIES = (
    "optical",
    "arbiter",
    "ff_apuf",
    "xor_apuf",
    "interpose",
    "permutation",
    "ring_oscillator",
    "sram",
    "memristive",
    "silicon_photonic",
)


class _TorchPUFModel(Simulation):
    """Adapt one or more PyTorch classifiers to pypuf's simulation interface."""

    def __init__(self, models: Sequence[Any], n: int, response_bits: int, device: Any):
        super().__init__()
        self.models = list(models)
        self._n = n
        self._response_bits = response_bits
        self.device = device

    @property
    def challenge_length(self) -> int:
        return self._n

    @property
    def response_length(self) -> int:
        return self._response_bits

    def eval(self, challenges: np.ndarray) -> np.ndarray:
        import torch

        inputs = torch.as_tensor(challenges, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = torch.stack([model(inputs) for model in self.models]).mean(dim=0)
            return torch.where(logits >= 0, 1, -1).to(torch.int8).cpu().numpy()


def _train_agent(
    challenges: np.ndarray,
    responses: np.ndarray,
    response_bits: int,
    hidden_layers: int,
    width: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: Any,
) -> Any:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    if device.type == "xpu":
        torch.xpu.manual_seed_all(seed)
    layers: list[nn.Module] = []
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(challenges.shape[1] if not layers else width, width), nn.Tanh()])
    input_size = width if hidden_layers else challenges.shape[1]
    layers.append(nn.Linear(input_size, response_bits))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    training_inputs = torch.as_tensor(challenges, dtype=torch.float32, device=device)
    response_matrix = np.asarray(responses).reshape(len(responses), response_bits)
    training_labels = torch.as_tensor(
        (response_matrix + 1) / 2, dtype=torch.float32, device=device
    )

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(training_inputs), device=device)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(training_inputs[indices]), training_labels[indices])
            loss.backward()
            optimizer.step()
    return model


def _resolve_device(backend: str, device: Optional[int]) -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to train PUF attack models.") from error

    normalized = backend.strip().lower()
    if normalized not in {"auto", "cpu", "torch_xpu"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'torch_xpu'.")
    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    if normalized == "auto":
        normalized = "torch_xpu" if xpu_available else "cpu"
    if normalized == "torch_xpu":
        if not xpu_available:
            raise RuntimeError("torch_xpu requested, but a usable Intel XPU is unavailable.")
        selected_device = 0 if device is None else device
        if selected_device < 0 or selected_device >= torch.xpu.device_count():
            raise ValueError(f"XPU device index {selected_device} is unavailable.")
        return torch.device(f"xpu:{selected_device}")
    if device is not None:
        raise ValueError("device can only be set when backend='torch_xpu'.")
    return torch.device("cpu")


def _pair_accuracy(model: Simulation, dataset: ChallengeResponseSet) -> float:
    predicted = np.asarray(model.eval(dataset.challenges))
    expected = np.asarray(dataset.responses).reshape(len(dataset.challenges), -1)
    return float(np.mean(np.all(predicted == expected, axis=1)))


def _attack_one_puf(
    target: Simulation,
    n: int,
    training_samples: int,
    validation_samples: int,
    test_samples: int,
    seed: int,
    success_threshold: float,
    max_depth: int,
    agents: int,
    width: Optional[int],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: Any,
) -> Dict[str, Any]:
    training = ChallengeResponseSet.from_simulation(target, N=training_samples, seed=seed)
    validation = ChallengeResponseSet.from_simulation(
        target, N=validation_samples, seed=seed + 1
    )
    test = ChallengeResponseSet.from_simulation(target, N=test_samples, seed=seed + 2)
    hidden_width = width or max(16, 2 * n)
    model: Optional[_TorchPUFModel] = None
    selected_stage = "logistic_regression"
    selected_depth = 0
    selected_agent_count = 1
    fit_seconds = 0.0
    stage_history: list[Dict[str, Any]] = []

    stages = [(0, 1)] + [(layer_count, agents) for layer_count in range(1, max_depth + 1)]
    for depth, agent_count in stages:
        stage_start = time.perf_counter()
        trained_models = []
        for agent_index in range(agent_count):
            trained_model = _train_agent(
                training.challenges,
                training.responses,
                target.response_length,
                depth,
                hidden_width,
                seed + depth * agents + agent_index,
                epochs,
                batch_size,
                learning_rate,
                device,
            )
            trained_models.append(trained_model)
        model = _TorchPUFModel(trained_models, n, target.response_length, device)
        validation_accuracy = _pair_accuracy(model, validation)
        elapsed = time.perf_counter() - stage_start
        fit_seconds += elapsed
        stage_name = "logistic_regression" if depth == 0 else "nonlinear_ensemble"
        stage_history.append({
            "attack": stage_name,
            "depth": depth,
            "agents": agent_count,
            "validation_accuracy": validation_accuracy,
            "fit_elapsed_seconds": float(elapsed),
        })
        selected_stage = stage_name
        selected_depth = depth
        selected_agent_count = agent_count
        if validation_accuracy >= success_threshold or depth == max_depth:
            break

    if model is None:
        raise RuntimeError("No attack model was trained.")
    validation_accuracy = _pair_accuracy(model, validation)
    test_accuracy = _pair_accuracy(model, test)
    return {
        "attack": selected_stage,
        "depth": selected_depth,
        "agents": selected_agent_count,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "success_threshold": success_threshold,
        "success": test_accuracy >= success_threshold,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "fit_elapsed_seconds": float(fit_seconds),
        "stages": stage_history,
    }


def run_ml_attack_experiment(
    families: Optional[Sequence[str]] = None,
    n: int = 64,
    training_samples: int = 10000,
    validation_samples: int = 2000,
    test_samples: int = 2000,
    seed: int = 20260926,
    k: int = 2,
    response_bits: int = 1,
    noisiness: float = 0.0,
    success_threshold: float = 0.95,
    max_depth: int = 3,
    agents: int = 3,
    width: Optional[int] = None,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    backend: str = "auto",
    device: Optional[int] = None,
    scenario: str = "general",
    report_root: Optional[str] = None,
    save_report: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Try logistic regression first, escalating to deeper neural ensembles.

    Each stage is selected using a validation CRP set. The winning stage is
    scored again on a separate test set. Success is exact response-vector
    agreement per challenge, rather than per-bit agreement.
    """
    selected_families = tuple(
        DEFAULT_ATTACK_FAMILIES if families is None else families
    )
    if not selected_families:
        raise ValueError("At least one PUF family is required.")
    unknown = [family for family in selected_families if family not in PUF_IMPLEMENTATIONS]
    if unknown:
        raise ValueError(f"Unknown PUF families: {', '.join(unknown)}")
    if n < 1 or min(training_samples, validation_samples, test_samples, epochs, batch_size) < 1:
        raise ValueError("Dimensions, sample counts, epochs, and batch size must be positive.")
    if not 0.5 < success_threshold <= 1.0:
        raise ValueError("success_threshold must be greater than 0.5 and at most 1.0.")
    if max_depth < 0 or agents < 1 or (width is not None and width < 1):
        raise ValueError("max_depth must be nonnegative; agents and width must be positive.")
    training_device = _resolve_device(backend, device)

    results: Dict[str, Dict[str, Any]] = {}
    for family_index, family in enumerate(selected_families):
        print(f"Attacking {family!r}...")
        target = create_puf(
            family,
            n=n,
            seed=seed + family_index,
            k=k,
            response_bits=response_bits,
            noisiness=noisiness,
        )
        results[family] = _attack_one_puf(
            target,
            n,
            training_samples,
            validation_samples,
            test_samples,
            seed + family_index * 3,
            success_threshold,
            max_depth,
            agents,
            width,
            epochs,
            batch_size,
            learning_rate,
            training_device,
        )

    if save_report:
        parameters = {
            "n": n,
            "training_samples": training_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "seed": seed,
            "k": k,
            "response_bits": response_bits,
            "noisiness": noisiness,
            "success_threshold": success_threshold,
            "max_depth": max_depth,
            "agents": agents,
            "width": width,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "backend": backend,
            "device": str(training_device),
            "scenario": scenario,
            "default_families": DEFAULT_ATTACK_FAMILIES,
        }
        report_directory = save_ml_attack_report(
            results, parameters, selected_families, scenario, report_root
        )
        print(f"Saved ML attack report to: {report_directory}")
    return results


__all__ = ["DEFAULT_ATTACK_FAMILIES", "run_ml_attack_experiment"]