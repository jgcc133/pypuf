"""Train progressively more expressive models against PUF implementations."""
from __future__ import annotations

import copy
import time
from math import ceil
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

CLEAR_LINE = '\033[1A\x1b[2K'


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


class _AgentTrainer:
    """Keep one model and optimizer alive while training it epoch by epoch."""

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
        from torch import nn

        torch.manual_seed(seed)
        if device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            input_size = challenges.shape[1] if not layers else width
            layers.extend([nn.Linear(input_size, width), nn.Tanh()])
        input_size = width if hidden_layers else challenges.shape[1]
        layers.append(nn.Linear(input_size, response_bits))
        self.model = nn.Sequential(*layers).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, foreach=False
        )
        self.loss_function = nn.BCEWithLogitsLoss()
        self.training_inputs = torch.as_tensor(
            challenges, dtype=torch.float32, device=device
        )
        response_matrix = np.asarray(responses).reshape(len(responses), response_bits)
        self.training_labels = torch.as_tensor(
            (response_matrix + 1) / 2, dtype=torch.float32, device=device
        )
        self.batch_size = batch_size

    def train_epoch(self) -> None:
        import torch

        self.model.train()
        order = torch.randperm(len(self.training_inputs), device=self.training_inputs.device)
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.loss_function(
                self.model(self.training_inputs[indices]), self.training_labels[indices]
            )
            loss.backward()
            self.optimizer.step()


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
    print_progress: bool = False,
) -> Dict[str, Any]:
    training = ChallengeResponseSet.from_simulation(target, N=training_samples, seed=seed)
    validation = ChallengeResponseSet.from_simulation(
        target, N=validation_samples, seed=seed + 1
    )    
    test = ChallengeResponseSet.from_simulation(target, N=test_samples, seed=seed + 2)
    hidden_width = width or max(16, 2 * n)
    best_model: Optional[_TorchPUFModel] = None
    best_validation_accuracy = -1.0
    selected_stage = "logistic_regression"
    selected_depth = 0
    selected_agent_count = 1
    fit_seconds = 0.0
    epochs_trained = 0
    agent_epochs_trained = 0
    threshold_reached = False
    stage_history: list[Dict[str, Any]] = []

    stages = [(0, max(1,agents))] + [(layer_count, agents) for layer_count in range(1, max_depth + 1)]
    for depth, agent_count in stages:
        trainers = []
        for agent_index in range(agent_count):
            trainers.append(_AgentTrainer(
                training.challenges,
                training.responses,
                target.response_length,
                depth,
                hidden_width,
                seed + depth * agents + agent_index,
                batch_size,
                learning_rate,
                device,
            ))
        if print_progress:
            print(f"Created {agent_count} trainers for depth {depth}.")
        stage_name = "logistic_regression" if depth == 0 else "nonlinear_ensemble"
        stage_history.append({
            "attack": stage_name,
            "depth": depth,
            "agents": agent_count,
            "epochs": 0,
            "validation_accuracy": None,
            "fit_elapsed_seconds": 0.0,
        })
        while True:
            epoch_start = time.perf_counter()
            for trainer in trainers:
                trainer.train_epoch()
            model = _TorchPUFModel(
                [trainer.model for trainer in trainers], n, target.response_length, device
            )
            validation_accuracy = _pair_accuracy(model, validation)
            elapsed = time.perf_counter() - epoch_start
            fit_seconds += elapsed
            epochs_trained += 1
            agent_epochs_trained += agent_count
            stage_history[-1]["epochs"] += 1
            stage_history[-1]["validation_accuracy"] = validation_accuracy
            stage_history[-1]["fit_elapsed_seconds"] += float(elapsed)
            print(f"\n")
            if print_progress:
                print(CLEAR_LINE * 3, end="")
                print(
                    f"Depth {depth}, trainers {agent_count}, "
                    f"epoch {stage_history[-1]['epochs']}/{epochs}, "
                    f"validation accuracy {validation_accuracy:.4%}"
                )

            if validation_accuracy > best_validation_accuracy:
                import torch

                best_model = _TorchPUFModel(
                    [copy.deepcopy(trainer.model).cpu() for trainer in trainers],
                    n,
                    target.response_length,
                    torch.device("cpu"),
                )
                best_validation_accuracy = validation_accuracy
                selected_stage = stage_name
                selected_depth = depth
                selected_agent_count = agent_count
                selected_stage = stage_name
                selected_depth = depth
                selected_agent_count = agent_count

            if validation_accuracy >= success_threshold:
                threshold_reached = True
                if print_progress:
                    print(
                        "Success threshold reached with validation accuracy "
                        f"{validation_accuracy:.4f}"
                    )
                break
            if stage_history[-1]["epochs"] >= epochs:
                if print_progress:
                    print(
                        f"Maximum epochs reached for depth {depth} with validation "
                        f"accuracy {validation_accuracy:.4f}"
                    )
                break
        if threshold_reached:
            break

    if best_model is None:
        raise RuntimeError("No attack model was trained.")
    validation_accuracy = best_validation_accuracy
    test_accuracy = _pair_accuracy(best_model, test)
    return {
        "attack": selected_stage,
        "depth": selected_depth,
        "agents": selected_agent_count,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "success_threshold": success_threshold,
        "success": test_accuracy >= success_threshold,
        "threshold_reached": threshold_reached,
        "epochs_to_threshold": epochs_trained if threshold_reached else None,
        "time_to_threshold_seconds": fit_seconds if threshold_reached else None,
        "epochs_trained": epochs_trained,
        "agent_epochs_trained": agent_epochs_trained,
        "epochs_per_depth_limit": epochs,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "test_samples": test_samples,
        "fit_elapsed_seconds": float(fit_seconds),
        "stages": stage_history,
    }


def _summarize_puf_instances(
    instance_results: Sequence[Dict[str, Any]], success_threshold: float
) -> Dict[str, Any]:
    instance_count = len(instance_results)
    required_matches = ceil(instance_count * success_threshold)
    matched_instances = sum(
        instance_result["instance_matched"] for instance_result in instance_results
    )
    epochs_trained = sum(
        instance_result["epochs_trained"] for instance_result in instance_results
    )
    fit_elapsed_seconds = sum(
        instance_result["fit_elapsed_seconds"] for instance_result in instance_results
    )

    matches_so_far = 0
    epochs_to_threshold: Optional[int] = None
    time_to_threshold_seconds: Optional[float] = None
    instances_processed_to_threshold: Optional[int] = None
    cumulative_epochs = 0
    cumulative_fit_seconds = 0.0
    for index, instance_result in enumerate(instance_results, start=1):
        cumulative_epochs += instance_result["epochs_trained"]
        cumulative_fit_seconds += instance_result["fit_elapsed_seconds"]
        matches_so_far += instance_result["instance_matched"]
        if matches_so_far >= required_matches:
            epochs_to_threshold = cumulative_epochs
            time_to_threshold_seconds = cumulative_fit_seconds
            instances_processed_to_threshold = index
            break

    return {
        "puf_instances": instance_count,
        "matched_instances": matched_instances,
        "instance_match_rate": matched_instances / instance_count,
        "required_matches": required_matches,
        "success_threshold": success_threshold,
        "success": matched_instances >= required_matches,
        "threshold_reached": instances_processed_to_threshold is not None,
        "instances_processed_to_threshold": instances_processed_to_threshold,
        "epochs_to_threshold": epochs_to_threshold,
        "time_to_threshold_seconds": time_to_threshold_seconds,
        "epochs_trained": epochs_trained,
        "fit_elapsed_seconds": fit_elapsed_seconds,
        "instances": list(instance_results),
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
    puf_instances: int = 1000,
    print_progress: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Train separate staged models for a cohort of independent PUF instances.

    Each instance receives separate training, validation, and test CRP sets.
    Its model trains until every validation response vector matches exactly,
    or until validation reaches ``success_threshold``. Each depth may train for
    up to ``epochs``; the best validation model is tested if no depth reaches
    the threshold. An instance counts as matched only when every held-out test
    response vector matches exactly. The cohort succeeds when the
    matched-instance fraction reaches ``success_threshold``.
    """
    selected_families = tuple(
        DEFAULT_ATTACK_FAMILIES if families is None else families
    )
    if not selected_families:
        raise ValueError("At least one PUF family is required.")
    unknown = [family for family in selected_families if family not in PUF_IMPLEMENTATIONS]
    if unknown:
        raise ValueError(f"Unknown PUF families: {', '.join(unknown)}")
    if n < 1 or min(
        training_samples, validation_samples, test_samples, epochs, batch_size, puf_instances
    ) < 1:
        raise ValueError("Dimensions, sample counts, epochs, and batch size must be positive.")
    if not 0.5 < success_threshold <= 1.0:
        raise ValueError("success_threshold must be greater than 0.5 and at most 1.0.")
    if max_depth < 0 or agents < 1 or (width is not None and width < 1):
        raise ValueError("max_depth must be nonnegative; agents and width must be positive.")
    training_device = _resolve_device(backend, device)

    results: Dict[str, Dict[str, Any]] = {}
    for family_index, family in enumerate(selected_families):
        print(f"Attacking {family!r}...")
        instance_results = []
        for instance_index in range(puf_instances):
            print(f"Creating instance {instance_index + 1}/{puf_instances}...")
            instance_seed = seed + (family_index * puf_instances + instance_index) * 4
            target = create_puf(
                family,
                n=n,
                seed=instance_seed,
                k=k,
                response_bits=response_bits,
                noisiness=noisiness,
            )
            print(CLEAR_LINE, end='')
            print(f"Attacking instance {instance_index + 1}/{puf_instances}...")
            instance_result = _attack_one_puf(
                target,
                n,
                training_samples,
                validation_samples,
                test_samples,
                instance_seed + 1,
                success_threshold,
                max_depth,
                agents,
                width,
                epochs,
                batch_size,
                learning_rate,
                training_device,
                print_progress,
            )
            instance_result.update({
                "instance_index": instance_index + 1,
                "instance_seed": instance_seed,
                "instance_matched": instance_result["test_accuracy"] == 1.0,
            })
            instance_results.append(instance_result)
        results[family] = _summarize_puf_instances(instance_results, success_threshold)

    if save_report:
        parameters = {
            "n": n,
            "training_samples": training_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "puf_instances": puf_instances,
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