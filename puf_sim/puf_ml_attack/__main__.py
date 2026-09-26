"""Command-line entry point for adaptive PUF machine-learning attacks."""
from __future__ import annotations

import argparse

from .experiment import DEFAULT_ATTACK_FAMILIES, run_ml_attack_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_ATTACK_FAMILIES))
    parser.add_argument("--scenario", default="general")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--training-samples", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--response-bits", type=int, default=1)
    parser.add_argument("--noisiness", type=float, default=0.0)
    parser.add_argument("--success-threshold", type=float, default=0.95)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--width", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--backend", choices=("auto", "cpu", "torch_xpu"), default="auto")
    parser.add_argument("--device", type=int)
    parser.add_argument("--report-root")
    parser.add_argument("--no-save-report", action="store_true")
    arguments = parser.parse_args()
    results = run_ml_attack_experiment(
        families=arguments.families,
        n=arguments.n,
        training_samples=arguments.training_samples,
        validation_samples=arguments.validation_samples,
        test_samples=arguments.test_samples,
        seed=arguments.seed,
        k=arguments.k,
        response_bits=arguments.response_bits,
        noisiness=arguments.noisiness,
        success_threshold=arguments.success_threshold,
        max_depth=arguments.max_depth,
        agents=arguments.agents,
        width=arguments.width,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        backend=arguments.backend,
        device=arguments.device,
        scenario=arguments.scenario,
        report_root=arguments.report_root,
        save_report=not arguments.no_save_report,
    )
    for family, result in results.items():
        print(
            f"{family}: {result['test_accuracy']:.3%} test imitation; "
            f"{result['attack']} depth={result['depth']} agents={result['agents']}"
        )


if __name__ == "__main__":
    main()