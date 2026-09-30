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

from . import arbiter as _arbiter
from . import optical as _optical
from .common import (
    AgentTrainer as _AgentTrainer,
    TorchPUFModel as _TorchPUFModel,
    pair_accuracy as _pair_accuracy,
    resolve_device as _resolve_device,
)
from .writer import save_ml_attack_report

_ArbiterAgentTrainer = _arbiter.AgentTrainer
_ArbiterTorchPUFModel = _arbiter.TorchPUFModel
_OpticalAgentTrainer = _optical.AgentTrainer
_OpticalTorchPUFModel = _optical.TorchPUFModel
_optical_pair_accuracy = _optical.pair_accuracy

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
    family: str,
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
    selected_stage = "neural_ensemble"
    selected_depth = 1
    selected_agent_count = agents
    fit_seconds = 0.0
    epochs_trained = 0
    agent_epochs_trained = 0
    threshold_reached = False
    stage_history: list[Dict[str, Any]] = []

    family_key = family.strip().lower()
    if family_key == "arbiter":
        trainer_type = _ArbiterAgentTrainer
        model_type = _ArbiterTorchPUFModel
        stage_name = _arbiter.STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key in {"ff_apuf", "feed_forward_arbiter"}:
        trainer_type = _arbiter.feed_forward.AgentTrainer
        model_type = _TorchPUFModel
        stage_name = _arbiter.feed_forward.STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key in {"xor_apuf", "xor_arbiter"}:
        trainer_type = _arbiter.xor_apuf.AgentTrainer
        model_type = _arbiter.xor_apuf.XorTorchPUFModel
        stage_name = _arbiter.xor_apuf.STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "interpose":
        trainer_type = _arbiter.interpose.AgentTrainer
        model_type = _arbiter.interpose.InterposeTorchPUFModel
        stage_name = _arbiter.interpose.STAGE_NAME
        pair_accuracy_fn = _pair_accuracy
    elif family_key == "optical":
        trainer_type = _OpticalAgentTrainer
        model_type = _OpticalTorchPUFModel
        stage_name = _optical.STAGE_NAME
        pair_accuracy_fn = _optical_pair_accuracy
    else:
        trainer_type = _AgentTrainer
        model_type = _TorchPUFModel
        stage_name = "nonlinear_ensemble"
        pair_accuracy_fn = _pair_accuracy

    if family_key == "optical":
        bootstrap_start = time.perf_counter()
        bootstrap_layer = _optical.bootstrap_linear_model(
            training.challenges, training.responses, target.response_length, device
        )
        best_model = model_type([bootstrap_layer], n, target.response_length, device)
        best_validation_accuracy = pair_accuracy_fn(best_model, validation)
        fit_seconds += time.perf_counter() - bootstrap_start
        selected_stage = _optical.BOOTSTRAP_STAGE_NAME
        selected_depth = 0
        selected_agent_count = 1
        stage_history.append({
            "attack": _optical.BOOTSTRAP_STAGE_NAME,
            "depth": 0,
            "agents": 1,
            "epochs": 1,
            "validation_accuracy": best_validation_accuracy,
            "fit_elapsed_seconds": time.perf_counter() - bootstrap_start,
        })
        if print_progress:
            print(
                "Closed-form linear regression validation accuracy "
                f"{best_validation_accuracy:.4%}"
            )
        if best_validation_accuracy >= success_threshold:
            threshold_reached = True

    stages = (
        [(1, agents)]
        if family_key in {"xor_apuf", "xor_arbiter", "interpose"}
        else [(layer_count, agents) for layer_count in range(1, max_depth + 1)]
    )
    for depth, agent_count in stages:
        if threshold_reached:
            break
        trainers = []
        for agent_index in range(agent_count):
            trainer_arguments = (
                training.challenges,
                training.responses,
                target.response_length,
                depth,
                hidden_width,
                seed + depth * agents + agent_index,
                batch_size,
                learning_rate,
                device,
            )
            if family_key in {"xor_apuf", "xor_arbiter"}:
                trainers.append(trainer_type(*trainer_arguments, chain_count=target.k))
            elif family_key == "interpose":
                trainers.append(trainer_type(
                    *trainer_arguments,
                    down_chain_count=target.down.k,
                    interpose_pos=target.interpose_pos,
                ))
            else:
                trainers.append(trainer_type(*trainer_arguments))
        if print_progress:
            print(f"Created {agent_count} trainers for depth {depth}.")
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
            model = model_type(
                [trainer.model for trainer in trainers], n, target.response_length, device
            )
            validation_accuracy = pair_accuracy_fn(model, validation)
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

                best_model = model_type(
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
                        f"{validation_accuracy:.4%}"
                    )
                break
            if stage_history[-1]["epochs"] >= epochs:
                if print_progress:
                    print(
                        f"Maximum epochs reached for depth {depth} with validation "
                        f"accuracy {validation_accuracy:.4%}"
                    )
                break
        if threshold_reached:
            break

    if best_model is None:
        raise RuntimeError("No attack model was trained.")
    validation_accuracy = best_validation_accuracy
    test_accuracy = pair_accuracy_fn(best_model, test)
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
    if max_depth < 1 or agents < 1 or (width is not None and width < 1):
        raise ValueError("max_depth must be at least 1; agents and width must be positive.")
    training_device = _resolve_device(backend, device)

    results: Dict[str, Dict[str, Any]] = {}
    for family_index, family in enumerate(selected_families):
        print(f"\nAttacking {family!r}...")
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
                family,
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