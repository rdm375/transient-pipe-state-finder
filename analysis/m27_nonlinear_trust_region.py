#!/usr/bin/env python3
"""M27-A2c: Nonlinear correlated-noise GLS reconstruction."""

from pathlib import Path
import time

import numpy as np
import pandas as pd

from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares

from m26_correlated_noise_monte_carlo import (
    load_problem,
    temporal_covariance,
)

from m26_calibration_prior_sensitivity import TAU

from m27_nonlinear_measurement_validation import Replay


OUT = Path("benchmarks/results-m27")
OUT.mkdir(parents=True, exist_ok=True)

N = 163
INITIAL_RANK = 120

# This is a diagnostic iteration budget, not a convergence claim.
MAX_EVALUATIONS = 8


def main():
    A, g = load_problem()
    J = 0.5 * A

    noise = pd.read_csv(
        OUT / "correlated_measurement_realization.csv"
    )["noise_normalized"].to_numpy(dtype=float)

    L = cholesky(
        temporal_covariance(TAU),
        lower=True,
    )

    Jw = solve_triangular(L, J, lower=True)
    yw = solve_triangular(L, noise, lower=True)

    U, s, Vt = np.linalg.svd(
        Jw,
        full_matrices=False,
    )

    x0 = Vt[:INITIAL_RANK].T @ (
        (U[:, :INITIAL_RANK].T @ yw)
        / s[:INITIAL_RANK]
    )

    restart_file = (
        OUT / "nonlinear_trust_region_solution.csv"
    )

    if restart_file.exists():
        restart = pd.read_csv(restart_file)

        x0 = restart[
            "reconstructed_scaled_value"
        ].to_numpy(dtype=float)

        if x0.shape != (N,):
            raise RuntimeError(
                "Restart solution has incorrect dimension"
            )

        print(
            "Restarting from saved nonlinear solution."
        )

    replay = Replay()

    evaluations = []
    start = time.perf_counter()

    def residual(x):
        predicted = replay.measure(x)

        whitened = solve_triangular(
            L,
            predicted - noise,
            lower=True,
        )

        objective = 0.5 * np.dot(
            whitened,
            whitened,
        )

        evaluations.append({
            "evaluation": len(evaluations),
            "elapsed_s": time.perf_counter() - start,
            "objective": objective,
            "scaled_norm": np.linalg.norm(x),
            "maximum_coordinate": np.max(np.abs(x)),
        })

        if len(evaluations) == 1 or len(evaluations) % 25 == 0:
            print(
                f"eval={len(evaluations):4d} "
                f"objective={objective:.9f} "
                f"||x||={np.linalg.norm(x):.6e}",
                flush=True,
            )

        return whitened

    print()
    print("M27-A2c: Nonlinear Trust-Region Reconstruction")
    print("=" * 78)
    print(f"Initial truncated-SVD rank: {INITIAL_RANK}")
    print(f"Maximum function evaluations: {MAX_EVALUATIONS}")

    try:
        initial_residual = residual(x0)
        initial_objective = (
            0.5 * np.dot(
                initial_residual,
                initial_residual,
            )
        )

        print(
            f"Initial nonlinear objective: "
            f"{initial_objective:.9f}"
        )

        result = least_squares(
            residual,
            x0,
            method="trf",
            jac="2-point",
            x_scale="jac",
            max_nfev=MAX_EVALUATIONS,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=1,
        )

        diagnostics = replay.diagnostics(result.x)

        print()
        print("Optimization results")
        print("-" * 78)
        print("Success:", result.success)
        print("Status:", result.status)
        print("Message:", result.message)
        print("Function evaluations:", result.nfev)
        print("Jacobian evaluations:", result.njev)
        print("Initial objective:", initial_objective)
        print("Final objective:", result.cost)
        print("Scaled solution norm:", np.linalg.norm(result.x))
        print("Maximum coordinate:", np.max(np.abs(result.x)))
        print("Terminal inventory error:", diagnostics[2], "kg")
        print("Minimum pressure:", diagnostics[4], "Pa")
        print("Maximum mass defect:", diagnostics[5], "kg")
        print("Boundary compatibility:", diagnostics[6], "Pa")

        pd.DataFrame(evaluations).to_csv(
            OUT / "nonlinear_trust_region_evaluations.csv",
            index=False,
        )

        pd.DataFrame({
            "coordinate": np.arange(N),
            "reconstructed_scaled_value": result.x,
        }).to_csv(
            OUT / "nonlinear_trust_region_solution.csv",
            index=False,
        )

        print()
        print("Saved nonlinear reconstruction diagnostics.")

    finally:
        replay.close()


if __name__ == "__main__":
    main()
