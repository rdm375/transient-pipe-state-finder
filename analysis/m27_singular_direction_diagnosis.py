#!/usr/bin/env python3
"""M27-A2b: Diagnose weak singular directions and nonlinear admissibility."""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import cholesky, solve_triangular

from m26_correlated_noise_monte_carlo import (
    load_problem,
    temporal_covariance,
)

from m26_calibration_prior_sensitivity import TAU

from m27_nonlinear_measurement_validation import Replay


OUT = Path("benchmarks/results-m27")
OUT.mkdir(parents=True, exist_ok=True)

N = 163

RANKS = [
    40,
    80,
    100,
    120,
    130,
    140,
    145,
    150,
    155,
    158,
    160,
    161,
    162,
    163,
]


def main():
    A, g = load_problem()

    J = 0.5 * A

    data = pd.read_csv(
        OUT / "correlated_measurement_realization.csv"
    )

    noise = data["noise_normalized"].to_numpy()

    K = temporal_covariance(TAU)
    L = cholesky(K, lower=True)

    Jw = solve_triangular(L, J, lower=True)
    yw = solve_triangular(L, noise, lower=True)

    U, s, Vt = np.linalg.svd(
        Jw,
        full_matrices=False,
    )

    coefficients = U.T @ yw

    records = []

    replay = Replay()

    print()
    print("M27-A2b: Singular-Direction Diagnosis")
    print("=" * 112)

    print(
        "rank  "
        "condition       "
        "||d||          "
        "max|d|         "
        "linear_obj     "
        "inventory_kg   "
        "nonlinear_obj   "
        "status"
    )

    try:
        for rank in RANKS:
            d = Vt[:rank, :].T @ (
                coefficients[:rank] / s[:rank]
            )

            linear_residual = Jw @ d - yw

            linear_obj = (
                0.5 * np.dot(
                    linear_residual,
                    linear_residual,
                )
            )

            inventory_linear = float(g @ d)

            condition = s[0] / s[rank - 1]

            nonlinear_obj = np.nan
            nonlinear_inventory = np.nan
            min_pressure = np.nan
            mass_defect = np.nan
            compatibility = np.nan

            status = "PASS"

            try:
                predicted = replay.measure(d)

                residual = solve_triangular(
                    L,
                    predicted - noise,
                    lower=True,
                )

                nonlinear_obj = (
                    0.5 * np.dot(residual, residual)
                )

                diagnostics = replay.diagnostics(d)

                nonlinear_inventory = diagnostics[2]
                min_pressure = diagnostics[4]
                mass_defect = diagnostics[5]
                compatibility = diagnostics[6]

            except RuntimeError as error:
                status = str(error)[:100]

            records.append({
                "rank": rank,
                "condition_number": condition,
                "scaled_norm": np.linalg.norm(d),
                "maximum_scaled_coordinate":
                    np.max(np.abs(d)),
                "linearized_objective": linear_obj,
                "linearized_inventory_error_kg":
                    inventory_linear,
                "nonlinear_objective": nonlinear_obj,
                "nonlinear_inventory_error_kg":
                    nonlinear_inventory,
                "minimum_pressure_pa": min_pressure,
                "maximum_mass_defect_kg": mass_defect,
                "boundary_compatibility_pa":
                    compatibility,
                "status": status,
            })

            print(
                f"{rank:4d}  "
                f"{condition:12.4e}  "
                f"{np.linalg.norm(d):12.4e}  "
                f"{np.max(np.abs(d)):12.4e}  "
                f"{linear_obj:12.5f}  "
                f"{inventory_linear:12.4f}  "
                f"{nonlinear_obj:13.5g}  "
                f"{status}"
            )

    finally:
        replay.close()

    output = OUT / "singular_direction_diagnosis.csv"

    pd.DataFrame(records).to_csv(
        output,
        index=False,
    )

    print()
    print("Saved:", output)

    valid = [
        row
        for row in records
        if row["status"] == "PASS"
        and np.isfinite(row["nonlinear_objective"])
    ]

    print()
    print("Nonlinearly evaluable ranks:", len(valid))

    if valid:
        best = min(
            valid,
            key=lambda row: row["nonlinear_objective"],
        )

        print(
            "Lowest nonlinear objective among evaluated ranks:",
            best["rank"],
            f"({best['nonlinear_objective']:.9f})",
        )

    print()
    print("Diagnostic completed.")


if __name__ == "__main__":
    main()
