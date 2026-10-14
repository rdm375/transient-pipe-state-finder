#!/usr/bin/env python3
"""M2.3a: Linearized measurement-noise and bias sensitivity.

This is a linearized perturbation experiment, not a nonlinear
reconstruction validation.

The sensitivity matrix K is taken from M2.2.

State coordinates use the M2.1 scaling:
    pressure: 1e5 Pa
    mass flow: 10 kg/s

Observation coordinates use:
    inlet flow: 1 kg/s
    outlet pressure: 1e4 Pa

The experiment compares regularized linearized inversions
under identical random measurement perturbations.

Requires NumPy.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def regularization(n, kind, lambda0, lambda2):
    """Construct a regularization matrix in scaled coordinates."""

    d = 2 * n + 1

    L = np.sqrt(lambda0) * np.eye(d)

    if kind == "tikhonov":
        return L

    if kind == "hybrid":
        # Second differences on internal pressure and flow.
        # The inlet-flow coordinate is excluded from curvature.
        D = np.zeros((2 * (n - 2), d))

        for block, offset in enumerate((0, n)):
            for i in range(n - 2):
                D[
                    block * (n - 2) + i,
                    offset + i:offset + i + 3
                ] = (0.25, -0.5, 0.25)

        return np.vstack((L, np.sqrt(lambda2) * D))

    if kind == "unregularized":
        return np.zeros((0, d))

    raise ValueError(f"Unknown regularizer: {kind}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--k",
        type=Path,
        default=Path("benchmarks/results-m22/K.csv")
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks/results-m23")
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=100
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20261013
    )

    parser.add_argument(
        "--noise-fraction",
        type=float,
        default=0.01,
        help="Noise standard deviation relative to M2.1 observation scale"
    )

    parser.add_argument(
        "--lambda0",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--lambda2",
        type=float,
        default=1e-2
    )

    args = parser.parse_args()

    if args.trials < 2:
        parser.error("Require at least two trials")

    if not np.isfinite(args.noise_fraction) or args.noise_fraction < 0:
        parser.error("Noise fraction must be finite and nonnegative")

    if not np.isfinite(args.lambda0) or args.lambda0 <= 0:
        parser.error("lambda0 must be finite and positive")

    if not np.isfinite(args.lambda2) or args.lambda2 < 0:
        parser.error("lambda2 must be finite and nonnegative")

    K = np.loadtxt(args.k, delimiter=",")

    m, d = K.shape

    if d % 2 != 1 or m % 2 != 0:
        raise ValueError("Unexpected sensitivity matrix dimensions")

    if not np.isfinite(K).all():
        raise ValueError("Nonfinite sensitivity matrix")

    n = (d - 1) // 2

    if n < 3:
        raise ValueError("At least three cells are required")

    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Shared noise realizations provide paired comparisons.
    noise = rng.normal(
        loc=0.0,
        scale=args.noise_fraction,
        size=(args.trials, m)
    )

    np.savetxt(
        args.out / "normalized_noise_realizations.csv",
        noise,
        delimiter=",",
        fmt="%.17g"
    )

    # Observation ordering:
    #   even rows: inlet mass flow
    #   odd rows:  outlet pressure
    #
    # A bias amplitude of 1.0 corresponds to:
    #   inlet flow: 1 kg/s
    #   outlet pressure: 1e4 Pa
    biases = {
        "none": np.zeros(m),
        "inlet_flow": np.tile([1.0, 0.0], m // 2),
        "outlet_pressure": np.tile([0.0, 1.0], m // 2),
    }

    rows = []

    for method in ("tikhonov", "hybrid"):
        R = regularization(
            n,
            method,
            args.lambda0,
            args.lambda2
        )

        # Solve regularized least squares using SVD.
        #
        # Avoid forming K.T @ K, which squares the
        # condition number.
        A = np.vstack((K, R))

        U, singular_values, Vt = np.linalg.svd(
            A,
            full_matrices=False
        )

        cutoff = (
            max(A.shape)
            * np.finfo(float).eps
            * singular_values[0]
        )

        keep = singular_values > cutoff

        # Linear map from observation perturbation
        # to estimated scaled-state perturbation.
        H = (
            Vt[keep].T / singular_values[keep]
        ) @ U[:m, keep].T

        for label, bias in biases.items():
            errors = (noise + bias) @ H.T

            state_norm = np.linalg.norm(
                errors,
                axis=1
            )

            observation_norm = np.linalg.norm(
                errors @ K.T,
                axis=1
            )

            mean_error = errors.mean(axis=0)

            rows.append({
                "method": method,
                "bias_channel": label,
                "trials": args.trials,
                "seed": args.seed,
                "noise_fraction": args.noise_fraction,
                "lambda0": args.lambda0,
                "lambda2": args.lambda2,
                "mean_scaled_state_norm": state_norm.mean(),
                "p95_scaled_state_norm": np.quantile(
                    state_norm, 0.95
                ),
                "mean_linearized_observation_norm":
                    observation_norm.mean(),
                "norm_of_mean_state_error":
                    np.linalg.norm(mean_error),
            })

    output = args.out / "linear_noise_pilot_summary.csv"

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
            lineterminator="\n"
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"state_dimension={d}")
    print(f"observation_dimension={m}")
    print(f"trials={args.trials}")
    print(f"seed={args.seed}")
    print(f"noise_fraction={args.noise_fraction}")
    print(f"lambda0={args.lambda0}")
    print(f"lambda2={args.lambda2}")

    print()

    for row in rows:
        print(
            f"{row['method']:10s} "
            f"bias={row['bias_channel']:16s} "
            f"mean_state="
            f"{row['mean_scaled_state_norm']:.6g} "
            f"p95_state="
            f"{row['p95_scaled_state_norm']:.6g} "
            f"mean_obs="
            f"{row['mean_linearized_observation_norm']:.6g}"
        )

    print()
    print(
        "NOTE: Linearized perturbation sensitivity only; "
        "not nonlinear reconstruction validation."
    )
    print(f"PASS M2.3a pilot: {output}")


if __name__ == "__main__":
    main()
