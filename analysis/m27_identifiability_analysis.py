"""M27-A5: Correlated nonlinear reconstruction identifiability.

Analyze native Gauss-Newton solutions in the whitened
measurement-Jacobian singular-vector basis.

The analysis uses:
    M24 measurement Jacobian
    M25 terminal inventory gradient
    M26 temporal covariance
    M27 saved native reconstruction vectors

No solver modifications or statistical priors.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import (
    cholesky,
    solve_triangular,
)

from m26_correlated_noise_monte_carlo import (
    temporal_covariance,
)


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "benchmarks/results-m27"

JACOBIAN_FILE = (
    ROOT
    / "benchmarks/results-m24/"
    "joint_measurement_jacobian.csv"
)

SENSITIVITY_FILE = (
    ROOT
    / "benchmarks/results-m25/"
    "operational_sensitivity.csv"
)

SWEEP_FILE = RESULTS / "native_iteration_sweep.csv"

DIRECTION_OUTPUT = (
    RESULTS / "identifiability_directions.csv"
)

SOLUTION_OUTPUT = (
    RESULTS / "identifiability_solutions.csv"
)

BAND_OUTPUT = (
    RESULTS / "identifiability_rank_bands.csv"
)

MOVEMENT_OUTPUT = (
    RESULTS / "identifiability_solution_movements.csv"
)


SIGMA_SENSOR = 0.5
TAU = 120.0

N_VARIABLES = 163
N_MEASUREMENTS = 244

ITERATIONS = (6, 12, 24)

# Rank bands are one-indexed and inclusive.
RANK_BANDS = (
    (1, 40),
    (41, 80),
    (81, 120),
    (121, 140),
    (141, 150),
    (151, 160),
    (161, 163),
)


def load_jacobian():
    frame = pd.read_csv(JACOBIAN_FILE)

    expected_columns = [
        f"x{i}" for i in range(N_VARIABLES)
    ]

    if frame.columns.tolist() != expected_columns:
        raise RuntimeError(
            "Unexpected M24 Jacobian column ordering"
        )

    J = frame.to_numpy(dtype=float)

    if J.shape != (
        N_MEASUREMENTS,
        N_VARIABLES,
    ):
        raise RuntimeError(
            f"Unexpected Jacobian shape: {J.shape}"
        )

    if not np.all(np.isfinite(J)):
        raise RuntimeError(
            "Nonfinite Jacobian entries"
        )

    return J


def load_inventory_gradient():
    frame = pd.read_csv(SENSITIVITY_FILE)

    required = {
        "time_s",
        "column",
        "quantity",
        "index",
        "sensitivity",
    }

    if not required.issubset(frame.columns):
        raise RuntimeError(
            "Unexpected M25 sensitivity schema"
        )

    terminal = frame.loc[
        np.isclose(
            frame["time_s"].to_numpy(float),
            600.0,
            rtol=0.0,
            atol=1e-10,
        )
        & frame["quantity"].eq("inventory")
        & frame["index"].eq(0)
    ].copy()

    if len(terminal) != N_VARIABLES:
        raise RuntimeError(
            "Expected exactly 163 terminal inventory "
            f"sensitivities, found {len(terminal)}"
        )

    terminal = terminal.sort_values("column")

    expected = np.arange(N_VARIABLES)

    if not np.array_equal(
        terminal["column"].to_numpy(),
        expected,
    ):
        raise RuntimeError(
            "Terminal inventory gradient ordering mismatch"
        )

    gradient = terminal[
        "sensitivity"
    ].to_numpy(float)

    if not np.all(np.isfinite(gradient)):
        raise RuntimeError(
            "Nonfinite inventory gradient"
        )

    return gradient


def load_solution(iterations):
    filename = (
        RESULTS
        / f"native_gn_{iterations:03d}_solution.csv"
    )

    frame = pd.read_csv(filename)

    if frame.columns.tolist() != [
        "coordinate",
        "reconstructed_scaled_value",
    ]:
        raise RuntimeError(
            f"Unexpected solution schema: {filename}"
        )

    if len(frame) != N_VARIABLES:
        raise RuntimeError(
            f"Unexpected solution dimension: {filename}"
        )

    if not np.array_equal(
        frame["coordinate"].to_numpy(),
        np.arange(N_VARIABLES),
    ):
        raise RuntimeError(
            f"Unexpected solution ordering: {filename}"
        )

    x = frame[
        "reconstructed_scaled_value"
    ].to_numpy(float)

    if not np.all(np.isfinite(x)):
        raise RuntimeError(
            f"Nonfinite solution: {filename}"
        )

    return x


def main():
    for filename in (
        DIRECTION_OUTPUT,
        SOLUTION_OUTPUT,
        BAND_OUTPUT,
        MOVEMENT_OUTPUT,
    ):
        if filename.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {filename}"
            )

    J = load_jacobian()
    g = load_inventory_gradient()

    K = temporal_covariance(TAU)

    if K.shape != (
        N_MEASUREMENTS,
        N_MEASUREMENTS,
    ):
        raise RuntimeError(
            "Unexpected temporal covariance shape"
        )

    # C = sigma_sensor^2 K.
    #
    # A = C^(-1/2) J
    #   = (sigma_sensor L)^(-1) J,
    #
    # where K = L L^T.
    L = cholesky(K, lower=True)

    A = solve_triangular(
        SIGMA_SENSOR * L,
        J,
        lower=True,
        check_finite=True,
    )

    U, singular_values, Vt = np.linalg.svd(
        A,
        full_matrices=False,
    )

    V = Vt.T

    if singular_values.shape != (N_VARIABLES,):
        raise RuntimeError(
            "Unexpected singular-value dimension"
        )

    gamma = Vt @ g

    # Absolute inventory response per unit whitened
    # measurement response in each singular direction.
    inventory_amplification = (
        np.abs(gamma) / singular_values
    )

    direction_frame = pd.DataFrame({
        "rank": np.arange(1, N_VARIABLES+1),
        "singular_value": singular_values,
        "inventory_gradient_projection": gamma,
        "inventory_amplification": (
            inventory_amplification
        ),
        "inventory_variance_contribution": (
            inventory_amplification**2
        ),
    })

    direction_frame.to_csv(
        DIRECTION_OUTPUT,
        index=False,
        float_format="%.17g",
    )

    sweep = pd.read_csv(SWEEP_FILE)

    sweep = sweep.set_index("max_iterations")

    solutions = {
        iterations: load_solution(iterations)
        for iterations in ITERATIONS
    }

    solution_rows = []
    band_rows = []

    for iterations, x in solutions.items():
        coefficients = Vt @ x

        linear_measurement = A @ x

        linear_inventory = g @ x

        nonlinear_inventory = float(
            sweep.loc[
                iterations,
                "terminal_inventory_error_kg",
            ]
        )

        solution_rows.append({
            "max_iterations": iterations,
            "scaled_solution_norm": np.linalg.norm(x),
            "linearized_measurement_norm": (
                np.linalg.norm(linear_measurement)
            ),
            "linearized_inventory_error_kg": (
                linear_inventory
            ),
            "nonlinear_inventory_error_kg": (
                nonlinear_inventory
            ),
            "nonlinear_minus_linear_inventory_kg": (
                nonlinear_inventory - linear_inventory
            ),
            "relative_inventory_linearization_error": (
                abs(nonlinear_inventory-linear_inventory)
                / max(
                    1.0,
                    abs(nonlinear_inventory),
                )
            ),
        })

        for start, stop in RANK_BANDS:
            sl = slice(start-1, stop)

            band_coefficients = coefficients[sl]

            band_measurement = (
                singular_values[sl]
                * band_coefficients
            )

            band_inventory = (
                gamma[sl]
                * band_coefficients
            )

            band_rows.append({
                "max_iterations": iterations,
                "rank_start": start,
                "rank_stop": stop,
                "scaled_state_norm": (
                    np.linalg.norm(band_coefficients)
                ),
                "whitened_measurement_norm": (
                    np.linalg.norm(band_measurement)
                ),
                "linear_inventory_contribution_kg": (
                    np.sum(band_inventory)
                ),
                "absolute_inventory_contribution_kg": (
                    np.sum(np.abs(band_inventory))
                ),
            })

    movement_rows = []

    for previous, current in (
        (6, 12),
        (12, 24),
        (6, 24),
    ):
        delta = (
            solutions[current]
            - solutions[previous]
        )

        coefficients = Vt @ delta

        movement_rows.append({
            "from_iterations": previous,
            "to_iterations": current,
            "scaled_movement_norm": np.linalg.norm(delta),
            "linearized_measurement_movement_norm": (
                np.linalg.norm(A @ delta)
            ),
            "linearized_inventory_change_kg": (
                g @ delta
            ),
            "actual_inventory_change_kg": (
                float(
                    sweep.loc[
                        current,
                        "terminal_inventory_error_kg",
                    ]
                )
                - float(
                    sweep.loc[
                        previous,
                        "terminal_inventory_error_kg",
                    ]
                )
            ),
        })

        for start, stop in RANK_BANDS:
            sl = slice(start-1, stop)

            movement_rows.append({
                "from_iterations": previous,
                "to_iterations": current,
                "rank_start": start,
                "rank_stop": stop,
                "scaled_movement_norm": (
                    np.linalg.norm(coefficients[sl])
                ),
                "linearized_measurement_movement_norm": (
                    np.linalg.norm(
                        singular_values[sl]
                        * coefficients[sl]
                    )
                ),
                "linearized_inventory_change_kg": (
                    np.sum(
                        gamma[sl]
                        * coefficients[sl]
                    )
                ),
                "actual_inventory_change_kg": np.nan,
            })

    solution_frame = pd.DataFrame(solution_rows)

    band_frame = pd.DataFrame(band_rows)

    movement_frame = pd.DataFrame(movement_rows)

    solution_frame.to_csv(
        SOLUTION_OUTPUT,
        index=False,
        float_format="%.17g",
    )

    band_frame.to_csv(
        BAND_OUTPUT,
        index=False,
        float_format="%.17g",
    )

    movement_frame.to_csv(
        MOVEMENT_OUTPUT,
        index=False,
        float_format="%.17g",
    )

    print("\nM27-A5 identifiability analysis")

    print("\nWhitened Jacobian")
    print("Shape:", A.shape)

    print(
        "Largest singular value:",
        singular_values[0],
    )

    print(
        "Smallest singular value:",
        singular_values[-1],
    )

    print(
        "Condition number:",
        singular_values[0]/singular_values[-1],
    )

    print(
        "\nLinearized inventory sigma:",
        np.linalg.norm(inventory_amplification),
        "kg",
    )

    print("\nReconstruction comparisons")
    print(solution_frame.to_string(index=False))

    print("\nRank-band decomposition")
    print(band_frame.to_string(index=False))

    print("\nSolution movements")
    print(movement_frame.to_string(index=False))

    print("\nSaved:")
    print(DIRECTION_OUTPUT)
    print(SOLUTION_OUTPUT)
    print(BAND_OUTPUT)
    print(MOVEMENT_OUTPUT)


if __name__ == "__main__":
    main()
