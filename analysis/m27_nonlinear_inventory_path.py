#!/usr/bin/env python3
"""M27-A6: Native tangent-linear inventory sensitivity path integrals."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results-m27"
EXECUTABLE = (
    ROOT / "build/state_finder_m27_inventory_directional_sensitivity"
)

DIMENSION = 163
TRUE_INVENTORY = 812846.35761952039

QUADRATURE_ORDERS = (2, 4, 8, 16)

PATHS = (
    ("reference", "006"),
    ("reference", "012"),
    ("reference", "024"),
    ("006", "012"),
    ("012", "024"),
    ("006", "024"),
)


def load_solution(label):
    if label == "reference":
        return np.zeros(DIMENSION)

    path = RESULTS / f"native_gn_{label}_solution.csv"

    frame = pd.read_csv(path)

    expected_columns = [
        "coordinate",
        "reconstructed_scaled_value",
    ]

    if list(frame.columns) != expected_columns:
        raise RuntimeError(f"Unexpected columns: {path}")

    coordinates = frame["coordinate"].to_numpy()

    if not np.array_equal(
        coordinates, np.arange(DIMENSION)
    ):
        raise RuntimeError(f"Invalid coordinates: {path}")

    values = frame[
        "reconstructed_scaled_value"
    ].to_numpy(dtype=float)

    if values.shape != (DIMENSION,):
        raise RuntimeError(f"Invalid dimension: {path}")

    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Nonfinite solution: {path}")

    return values


def write_vector(path, values):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")

        writer.writerow([
            "coordinate",
            "reconstructed_scaled_value",
        ])

        for index, value in enumerate(values):
            writer.writerow([
                index,
                format(float(value), ".17g"),
            ])


class NativeEvaluator:
    def __init__(self, directory):
        self.base_file = directory / "base.csv"
        self.direction_file = directory / "direction.csv"
        self.calls = 0

    def evaluate(self, base, direction):
        write_vector(self.base_file, base)
        write_vector(self.direction_file, direction)

        completed = subprocess.run(
            [
                str(EXECUTABLE),
                str(self.base_file),
                str(self.direction_file),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Native M27-A6 evaluation failed\n"
                f"Return code: {completed.returncode}\n"
                f"Base file: {self.base_file}\n"
                f"Direction file: {self.direction_file}\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        values = {}

        for line in completed.stdout.splitlines():
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            values[key.strip()] = float(value)

        required = (
            "terminal_inventory_kg",
            "terminal_inventory_true_kg",
            "directional_inventory_derivative_kg",
            "max_linear_backward_error",
        )

        for key in required:
            if key not in values:
                raise RuntimeError(
                    f"Missing native diagnostic: {key}"
                )

        if not np.isclose(
            values["terminal_inventory_true_kg"],
            TRUE_INVENTORY,
            rtol=1e-12,
            atol=1e-7,
        ):
            raise RuntimeError(
                "Native reference inventory mismatch"
            )

        self.calls += 1

        return values


def terminal_gradient():
    path = (
        ROOT
        / "benchmarks/results-m25/operational_sensitivity.csv"
    )

    frame = pd.read_csv(path)

    terminal = frame.loc[
        (frame["time_s"] == 600.0)
        & (frame["quantity"] == "inventory")
        & (frame["index"] == 0)
    ].sort_values("column")

    if not np.array_equal(
        terminal["column"].to_numpy(),
        np.arange(DIMENSION),
    ):
        raise RuntimeError("Invalid M25 gradient coordinates")

    return terminal["sensitivity"].to_numpy(dtype=float)


def main():
    if not EXECUTABLE.is_file():
        raise FileNotFoundError(EXECUTABLE)

    output_summary = (
        RESULTS / "nonlinear_inventory_path_summary.csv"
    )

    output_quadrature = (
        RESULTS / "nonlinear_inventory_path_quadrature.csv"
    )

    for output in (output_summary, output_quadrature):
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {output}"
            )

    solutions = {
        label: load_solution(label)
        for label in ("reference", "006", "012", "024")
    }

    gradient = terminal_gradient()

    summary_rows = []
    quadrature_rows = []

    with tempfile.TemporaryDirectory(
        prefix="m27_inventory_path_"
    ) as temporary:
        evaluator = NativeEvaluator(Path(temporary))

        zero = np.zeros(DIMENSION)

        reference_check = evaluator.evaluate(
            zero, solutions["006"]
        )

        native_nominal = reference_check[
            "directional_inventory_derivative_kg"
        ]

        m25_nominal = float(
            gradient @ solutions["006"]
        )

        nominal_difference = (
            native_nominal - m25_nominal
        )

        print("M27-A6 nominal tangent validation")
        print(f"  M25:    {m25_nominal:.12f} kg")
        print(f"  Native: {native_nominal:.12f} kg")
        print(
            f"  Difference: {nominal_difference:.6e} kg"
        )

        if not np.isclose(
            native_nominal,
            m25_nominal,
            rtol=1e-9,
            atol=1e-7,
        ):
            raise RuntimeError(
                "M25/native nominal derivative mismatch"
            )

        print("\nNonlinear reconstruction paths\n")

        for start_label, end_label in PATHS:
            start = solutions[start_label]
            end = solutions[end_label]

            direction = end - start

            start_result = evaluator.evaluate(
                start, direction
            )

            end_result = evaluator.evaluate(
                end, direction
            )

            midpoint_result = evaluator.evaluate(
                start + 0.5*direction,
                direction,
            )

            inventory_start = start_result[
                "terminal_inventory_kg"
            ]

            inventory_end = end_result[
                "terminal_inventory_kg"
            ]

            exact_change = (
                inventory_end - inventory_start
            )

            nominal_tangent = float(
                gradient @ direction
            )

            start_tangent = start_result[
                "directional_inventory_derivative_kg"
            ]

            midpoint_tangent = midpoint_result[
                "directional_inventory_derivative_kg"
            ]

            end_tangent = end_result[
                "directional_inventory_derivative_kg"
            ]

            label = f"{start_label}_to_{end_label}"

            row = {
                "path": label,
                "start": start_label,
                "end": end_label,
                "inventory_start_kg": inventory_start,
                "inventory_end_kg": inventory_end,
                "exact_inventory_change_kg": exact_change,
                "exact_change_pct_true_inventory":
                    100.0*exact_change/TRUE_INVENTORY,
                "nominal_tangent_kg": nominal_tangent,
                "start_tangent_kg": start_tangent,
                "midpoint_tangent_kg": midpoint_tangent,
                "end_tangent_kg": end_tangent,
                "nominal_minus_exact_kg":
                    nominal_tangent-exact_change,
                "start_minus_exact_kg":
                    start_tangent-exact_change,
                "midpoint_minus_exact_kg":
                    midpoint_tangent-exact_change,
                "direction_norm":
                    float(np.linalg.norm(direction)),
            }

            print(f"Path: {label}")
            print(
                f"  Exact change:    "
                f"{exact_change:+.9f} kg"
            )
            print(
                f"  Relative change: "
                f"{100*exact_change/TRUE_INVENTORY:+.9f}%"
            )
            print(
                f"  Nominal tangent: "
                f"{nominal_tangent:+.9f} kg"
            )
            print(
                f"  Start tangent:   "
                f"{start_tangent:+.9f} kg"
            )
            print(
                f"  Midpoint tangent:"
                f" {midpoint_tangent:+.9f} kg"
            )
            print(
                f"  End tangent:     "
                f"{end_tangent:+.9f} kg"
            )

            for order in QUADRATURE_ORDERS:
                nodes, weights = leggauss(order)

                # Map [-1,1] to [0,1].
                nodes = 0.5*(nodes+1.0)
                weights = 0.5*weights

                integral = 0.0
                maximum_backward_error = 0.0

                for s, weight in zip(nodes, weights):
                    state = start + float(s)*direction

                    result = evaluator.evaluate(
                        state, direction
                    )

                    integral += float(weight)*result[
                        "directional_inventory_derivative_kg"
                    ]

                    maximum_backward_error = max(
                        maximum_backward_error,
                        result["max_linear_backward_error"],
                    )

                error = integral-exact_change

                relative_quadrature_error = (
                    abs(error)
                    / max(abs(exact_change), 1.0)
                )

                quadrature_rows.append({
                    "path": label,
                    "quadrature_order": order,
                    "exact_inventory_change_kg":
                        exact_change,
                    "integrated_inventory_change_kg":
                        integral,
                    "quadrature_error_kg": error,
                    "quadrature_error_pct_true_inventory":
                        100.0*error/TRUE_INVENTORY,
                    "relative_quadrature_error":
                        relative_quadrature_error,
                    "max_linear_backward_error":
                        maximum_backward_error,
                })

                row[
                    f"gauss_{order}_integral_kg"
                ] = integral

                row[
                    f"gauss_{order}_error_kg"
                ] = error

                print(
                    f"  Gauss-{order:<2}:       "
                    f"{integral:+.9f} kg "
                    f"(error {error:+.3e} kg)"
                )

            summary_rows.append(row)
            print()

        print(
            f"Native sensitivity evaluations: "
            f"{evaluator.calls}"
        )

    summary = pd.DataFrame(summary_rows)
    quadrature = pd.DataFrame(quadrature_rows)

    # Require path-integral agreement with endpoint
    # inventory differences before publishing results.
    #
    # Absolute tolerance allows for forward-solver
    # nonlinear solve accuracy; relative tolerance
    # protects paths with larger inventory changes.
    finest = quadrature.loc[
        quadrature["quadrature_order"] == 16
    ]

    for record in finest.itertuples():
        allowed_error = (
            1e-3
            + 1e-5*abs(record.exact_inventory_change_kg)
        )

        if abs(record.quadrature_error_kg) > allowed_error:
            raise RuntimeError(
                f"Path-integral validation failed: "
                f"{record.path}, "
                f"error={record.quadrature_error_kg:.9g} kg, "
                f"tolerance={allowed_error:.9g} kg"
            )

    # Create both output files only after all calculations
    # and validation gates have succeeded.
    summary.to_csv(
        output_summary,
        index=False,
        float_format="%.17g",
        mode="x",
    )

    quadrature.to_csv(
        output_quadrature,
        index=False,
        float_format="%.17g",
        mode="x",
    )

    print("\nM27-A6 path-integral validation PASS")
    print(f"Saved: {output_summary}")
    print(f"Saved: {output_quadrature}")


if __name__ == "__main__":
    main()
