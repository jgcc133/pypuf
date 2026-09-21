"""Command-line smoke runner for puf_sim PUF implementations."""
from __future__ import annotations

import argparse

import numpy as np

from pypuf.io import random_inputs

from .factory import PUF_IMPLEMENTATIONS, create_puf


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and evaluate a puf_sim PUF.")
    parser.add_argument("--family", default="arbiter", help="PUF family name")
    parser.add_argument("--bits", type=int, default=64, help="Challenge length")
    parser.add_argument("--chains", type=int, default=1, help="XOR/interpose/permutation chain count")
    parser.add_argument("--response-bits", type=int, default=1, help="Analytical response width")
    parser.add_argument("--samples", type=int, default=10, help="Number of challenges")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic model and challenge seed")
    parser.add_argument("--list", action="store_true", help="List supported family names")
    args = parser.parse_args()

    if args.list:
        for family in sorted(PUF_IMPLEMENTATIONS):
            print(family)
        return

    puf = create_puf(
        args.family,
        n=args.bits,
        k=args.chains,
        response_bits=args.response_bits,
        seed=args.seed,
    )
    challenges = random_inputs(n=puf.challenge_length, N=args.samples, seed=args.seed + 1)
    responses = np.asarray(puf.eval(challenges))
    print(f"family={args.family}")
    print(f"challenge_shape={challenges.shape}")
    print(f"response_shape={responses.shape}")


if __name__ == "__main__":
    main()
