#!/usr/bin/env python3
"""M27-A2a: Linearized GLS solution with nonlinear replay validation."""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import (
    cholesky,
    solve_triangular,
)

from m26_correlated_noise_monte_carlo import (
    load_problem,
    temporal_covariance,
)

from m26_calibration_prior_sensitivity import TAU

from m27_nonlinear_measurement_validation import Replay


ROOT = Path("benchmarks/results-m27")

N_VARIABLES = 163
N_MEASUREMENTS = 244


def main():
    A, g = load_problem()

    # load_problem() returns the independent-noise-whitened
    # Jacobian A = 2J.
    J = 0.5 * A

    noise_file = (
        ROOT / "correlated_measurement_realization.csv"
    )

    noise_data = pd.read_csv(noise_file)

    noise = noise_data[
        "noise_normalized"
    ].to_numpy(dtype=float)

    if noise.shape != (N_MEASUREMENTS,):
        raise RuntimeError("Unexpected noise dimension")

    K = temporal_covariance(TAU)
    L = cholesky(K, lower=True)

    Jw = solve_triangular(
        L,
        J,
        lower=True,
    )

    yw = solve_triangular(
        L,
        noise,
        lower=True,
    )

    # Explicit full-column-rank SVD.
    #
    # Do not silently truncate weak singular directions.
    U, singular_values, Vt = np.linalg.svd(
        Jw,
        full_matrices=False,
    )

    coefficients = U.T @ yw

    d_linear = Vt.T @ (
        coefficients / singular_values
    )

    linear_residual = Jw @ d_linear - yw

    linear_objective = (
        0.5 * np.dot(
            linear_residual,
            linear_residual,
        )
    )

    predicted_inventory_error = float(
        g @ d_linear
    )

    print()
    print("M27-A2a: Linearized GLS and Nonlinear Replay")
    print("=" * 78)

    print(
        "Jacobian condition number:",
        f"{singular_values[0]/singular_values[-1]:.9e}",
    )

    print(
        "Linearized GLS objective:",
        f"{linear_objective:.9f}",
    )

    print(
        "Linearized inventory error:",
        f"{predicted_inventory_error:.9f} kg",
    )

    print(
        "Scaled reconstruction norm:",
        f"{np.linalg.norm(d_linear):.9e}",
    )

    print(
        "Largest scaled coordinate:",
        f"{np.max(np.abs(d_linear)):.9e}",
    )

    replay = Replay()

    try:
        zero = np.zeros(N_VARIABLES)

        reference = replay.diagnostics(zero)

        print()
        print("Reference diagnostics")
        print("-" * 78)

        print(
            "Minimum pressure:",
            f"{reference[4]:.6f} Pa",
        )

        print(
            "Mass defect:",
            f"{reference[5]:.9e} kg",
        )

        # Evaluate the complete linearized solution.
        try:
            nonlinear_measurement = replay.measure(
                d_linear
            )

            nonlinear_diagnostics = replay.diagnostics(
                d_linear
            )

            nonlinear_residual = solve_triangular(
                L,
                nonlinear_measurement - noise,
                lower=True,
            )

            nonlinear_objective = (
                0.5 * np.dot(
                    nonlinear_residual,
                    nonlinear_residual,
                )
            )

            print()
            print("Nonlinear replay of linearized solution")
            print("-" * 78)

            print(
                "Nonlinear GLS objective:",
                f"{nonlinear_objective:.9f}",
            )

            print(
                "Objective amplification:",
                f"{nonlinear_objective/linear_objective:.9e}",
            )

            print(
                "Nonlinear inventory error:",
                f"{nonlinear_diagnostics[2]:.9f} kg",
            )

            print(
                "Minimum pressure:",
                f"{nonlinear_diagnostics[4]:.6f} Pa",
            )

            print(
                "Maximum mass defect:",
                f"{nonlinear_diagnostics[5]:.9e} kg",
            )

            print(
                "Boundary compatibility:",
                f"{nonlinear_diagnostics[6]:.9e} Pa",
            )

            pd.DataFrame({
                "coordinate": np.arange(N_VARIABLES),
                "linearized_gls_solution": d_linear,
            }).to_csv(
                ROOT / "linearized_gls_solution.csv",
                index=False,
            )

            pd.DataFrame([{
                "linearized_objective": linear_objective,
                "nonlinear_objective": nonlinear_objective,
                "linearized_inventory_error_kg":
                    predicted_inventory_error,
                "nonlinear_inventory_error_kg":
                    nonlinear_diagnostics[2],
                "minimum_pressure_pa":
                    nonlinear_diagnostics[4],
                "maximum_mass_defect_kg":
                    nonlinear_diagnostics[5],
                "boundary_compatibility_pa":
                    nonlinear_diagnostics[6],
                "scaled_solution_norm":
                    np.linalg.norm(d_linear),
            }]).to_csv(
                ROOT / "linearized_nonlinear_comparison.csv",
                index=False,
            )

            print()
            print("Nonlinear replay: PASS")

        except RuntimeError as error:
            print()
            print("Nonlinear replay failed:")
            print(error)

            print(
                "This may indicate that the unregularized "
                "linearized GLS solution lies outside the "
                "nonlinear solver's admissible region."
            )

            pd.DataFrame({
                "coordinate": np.arange(N_VARIABLES),
                "linearized_gls_solution": d_linear,
            }).to_csv(
                ROOT / "linearized_gls_solution.csv",
                index=False,
            )

    finally:
        replay.close()


if __name__ == "__main__":
    main()
