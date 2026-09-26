"""Population-level baseline evaluation entry points."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from pypuf.io import ChallengeResponseSet
from pypuf.simulation import Simulation

from puf_sim.puf_implementations import create_puf
from .report import evaluate_exposed_metrics
from .writer import save_baseline_report

DEFAULT_PUF_FAMILIES = (
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
DEFAULT_INSTANCE_COUNT = 100


def _evaluate_family(
    family_index: int,
    family: str,
    instances_per_family: int,
    n: int,
    samples: int,
    repetitions: int,
    input_noise: float,
    seed: int,
    k: int,
    response_bits: int,
    noisiness: float,
    backend: str,
    device: Optional[int],
    batch_size: Optional[int],
) -> tuple[str, Dict[str, Any]]:
    """Create and evaluate one family; suitable for a worker thread."""
    print(f"Evaluating {instances_per_family} instances of {family!r}...")
    population = [
        create_puf(
            family,
            n=n,
            seed=seed + family_index * instances_per_family + device_index,
            k=k,
            response_bits=response_bits,
            noisiness=noisiness,
        )
        for device_index in range(instances_per_family)
    ]
    report = evaluate_exposed_metrics(
        population[0],
        instances=population,
        samples=samples,
        repetitions=repetitions,
        input_noise=input_noise,
        seed=seed + family_index,
        backend=backend,
        device=device,
        batch_size=batch_size,
    )
    return family, report


def evaluate_puf_population(
    instances: Iterable[Simulation],
    test_set: Optional[ChallengeResponseSet] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate a supplied PUF population with the baseline metric report."""
    population = list(instances)
    if not population:
        raise ValueError("At least one PUF instance is required.")
    return evaluate_exposed_metrics(
        population[0], instances=population, test_set=test_set, **kwargs
    )


def run_baseline_experiment(
    families: Optional[Sequence[str]] = None,
    instances_per_family: int = DEFAULT_INSTANCE_COUNT,
    n: int = 64,
    samples: int = 1000,
    repetitions: int = 17,
    input_noise: float = 0.01,
    seed: int = 20260921,
    k: int = 2,
    response_bits: int = 1,
    noisiness: float = 0.0,
    report_root: Optional[Path | str] = None,
    save_report: bool = True,
    workers: Optional[int] = None,
    backend: str = "numpy",
    device: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate the baseline metrics for a population of each PUF family.

    By default this creates and evaluates 100 independently seeded instances
    for each of the ten canonical families in ``DEFAULT_PUF_FAMILIES``.
    """
    if instances_per_family < 1:
        raise ValueError("instances_per_family must be at least one.")
    selected_families = tuple(families or DEFAULT_PUF_FAMILIES)
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least one or None.")
    max_workers = workers or min(len(selected_families), os.cpu_count() or 1)
    results: Dict[str, Dict[str, Any]] = {}
    print(f"Running baseline evaluation for {len(selected_families)} families...")
    print(f"Using {max_workers} family worker(s).")
    print(f"Expected number of computations: {len(selected_families) * instances_per_family}*{samples}*{repetitions} = {len(selected_families) * instances_per_family * samples * repetitions}")

    worker_arguments = [
        (
            family_index,
            family,
            instances_per_family,
            n,
            samples,
            repetitions,
            input_noise,
            seed,
            k,
            response_bits,
            noisiness,
            backend,
            device,
            batch_size,
        )
        for family_index, family in enumerate(selected_families)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        completed = executor.map(lambda args: _evaluate_family(*args), worker_arguments)
        for family, report in completed:
            results[family] = report

    if save_report:
        parameters = {
            "instances_per_family": instances_per_family,
            "n": n,
            "samples": samples,
            "repetitions": repetitions,
            "input_noise": input_noise,
            "seed": seed,
            "k": k,
            "response_bits": response_bits,
            "noisiness": noisiness,
            "workers": max_workers,
            "backend": backend,
            "device": device,
            "batch_size": batch_size,
            "default_families": DEFAULT_PUF_FAMILIES,
        }
        report_directory = save_baseline_report(
            results,
            parameters=parameters,
            families=selected_families,
            report_root=report_root,
        )
        print(f"Saved baseline report to: {report_directory}")

    return results

__all__ = [
    "DEFAULT_INSTANCE_COUNT",
    "DEFAULT_PUF_FAMILIES",
    "evaluate_puf_population",
    "run_baseline_experiment",
]
