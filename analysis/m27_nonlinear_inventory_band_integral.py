#!/usr/bin/env python3
"""M27-A8: Nonlinear inventory path integrals by nominal SVD band."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from numpy.polynomial.legendre import leggauss
from scipy.linalg import solve_triangular

from m26_correlated_noise_monte_carlo import temporal_covariance


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "benchmarks/results-m27"

EXECUTABLE = (
    ROOT / "build/state_finder_m27_inventory_directional_sensitivity"
)

DIMENSION = 163

TRUE_INVENTORY = 812846.35761952039

LABELS = ("reference", "006", "012", "024")

PATHS = (
    ("reference", "006"),
    ("reference", "012"),
    ("reference", "024"),
    ("006", "012"),
    ("012", "024"),
    ("006", "024"),
)

BANDS = (
    (1, 40),
    (41, 80),
    (81, 120),
    (121, 140),
    (141, 150),
    (151, 160),
    (161, 163),
)

ORDERS = (4, 8)


def load_solution(label):
    if label == "reference":
        return np.zeros(DIMENSION)

    path = RESULTS / f"native_gn_{label}_solution.csv"

    frame = pd.read_csv(path)

    expected = [
        "coordinate",
        "reconstructed_scaled_value",
    ]

    if list(frame.columns) != expected:
        raise RuntimeError(f"Invalid solution header: {path}")

    if not np.array_equal(
        frame["coordinate"].to_numpy(),
        np.arange(DIMENSION),
    ):
        raise RuntimeError(f"Invalid solution coordinates: {path}")

    values = frame[
        "reconstructed_scaled_value"
    ].to_numpy(dtype=float)

    if values.shape != (DIMENSION,):
        raise RuntimeError(f"Invalid solution dimension: {path}")

    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Nonfinite solution: {path}")

    return values


def load_gradient(label):
    path = RESULTS / f"local_inventory_gradient_{label}.csv"

    frame = pd.read_csv(path)

    expected = [
        "coordinate",
        "inventory_gradient_kg",
    ]

    if list(frame.columns) != expected:
        raise RuntimeError(f"Invalid gradient header: {path}")

    if not np.array_equal(
        frame["coordinate"].to_numpy(),
        np.arange(DIMENSION),
    ):
        raise RuntimeError(f"Invalid gradient coordinates: {path}")

    values = frame[
        "inventory_gradient_kg"
    ].to_numpy(dtype=float)

    if values.shape != (DIMENSION,):
        raise RuntimeError(f"Invalid gradient dimension: {path}")

    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Nonfinite gradient: {path}")

    return values


def load_nominal_svd():
    path = (
        ROOT
        / "benchmarks/results-m24/joint_measurement_jacobian.csv"
    )

    J = pd.read_csv(path).to_numpy(dtype=float)

    if J.shape != (244, DIMENSION):
        raise RuntimeError(f"Invalid Jacobian shape: {J.shape}")

    K = temporal_covariance(120.0)
    L = np.linalg.cholesky(K)

    A = solve_triangular(
        L,
        J / 0.5,
        lower=True,
        check_finite=True,
    )

    U, singular_values, Vt = np.linalg.svd(
        A,
        full_matrices=False,
    )

    if Vt.shape != (DIMENSION, DIMENSION):
        raise RuntimeError("Invalid SVD basis")

    return singular_values, Vt


def write_vector(path, values):
    with path.open("w", newline="") as handle:
        writer = csv.writer(
            handle,
            lineterminator="\n",
        )

        writer.writerow([
            "coordinate",
            "reconstructed_scaled_value",
        ])

        for index, value in enumerate(values):
            writer.writerow([
                index,
                format(float(value), ".17g"),
            ])


class NativeGradientEvaluator:
    def __init__(self, directory):
        self.directory = directory
        self.base_file = directory / "base.csv"
        self.calls = 0

    def evaluate(self, values):
        write_vector(self.base_file, values)

        output_file = (
            self.directory
            / f"gradient_{self.calls:05d}.csv"
        )

        if output_file.exists():
            raise FileExistsError(output_file)

        completed = subprocess.run(
            [
                str(EXECUTABLE),
                "--gradient",
                str(self.base_file),
                str(output_file),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                "Native gradient evaluation failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        frame = pd.read_csv(output_file)

        if list(frame.columns) != [
            "coordinate",
            "inventory_gradient_kg",
        ]:
            raise RuntimeError(
                f"Invalid native gradient header: {output_file}"
            )

        if not np.array_equal(
            frame["coordinate"].to_numpy(),
            np.arange(DIMENSION),
        ):
            raise RuntimeError(
                f"Invalid native gradient coordinates: {output_file}"
            )

        gradient = frame[
            "inventory_gradient_kg"
        ].to_numpy(dtype=float)

        if gradient.shape != (DIMENSION,):
            raise RuntimeError(
                f"Invalid native gradient shape: {output_file}"
            )

        if not np.all(np.isfinite(gradient)):
            raise RuntimeError(
                f"Nonfinite native gradient: {output_file}"
            )

        self.calls += 1

        return gradient


def main():
    outputs = {
        "bands": (
            RESULTS
            / "nonlinear_inventory_band_integrals.csv"
        ),
        "summary": (
            RESULTS
            / "nonlinear_inventory_band_integral_summary.csv"
        ),
        "convergence": (
            RESULTS
            / "nonlinear_inventory_band_integral_convergence.csv"
        ),
    }

    for output in outputs.values():
        if output.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {output}"
            )

    if not EXECUTABLE.is_file():
        raise FileNotFoundError(EXECUTABLE)

    singular_values, Vt = load_nominal_svd()

    solutions = {
        label: load_solution(label)
        for label in LABELS
    }

    gradients = {
        label: load_gradient(label)
        for label in LABELS
    }

    path_summary = pd.read_csv(
        RESULTS / "nonlinear_inventory_path_summary.csv"
    ).set_index("path")

    nominal_gamma = Vt @ gradients["reference"]

    band_rows = []
    summary_rows = []
    convergence_rows = []

    print("M27-A8 nonlinear inventory band decomposition")
    print(f"  Dimension: {DIMENSION}")
    print(
        f"  Nominal condition number: "
        f"{singular_values[0]/singular_values[-1]:.9e}"
    )

    with tempfile.TemporaryDirectory(
        prefix="m27_band_integral_"
    ) as temporary:

        evaluator = NativeGradientEvaluator(
            Path(temporary)
        )

        for start_label, end_label in PATHS:

            path_label = (
                f"{start_label}_to_{end_label}"
            )

            start = solutions[start_label]
            end = solutions[end_label]

            direction = end-start

            alpha = Vt @ direction

            exact_change = float(
                path_summary.loc[
                    path_label,
                    "exact_inventory_change_kg",
                ]
            )

            print(f"\nPath: {path_label}")
            print(
                f"  Exact nonlinear change: "
                f"{exact_change:+.9f} kg"
            )

            order_results = {}

            for order in ORDERS:
                nodes, weights = leggauss(order)

                nodes = 0.5*(nodes+1.0)
                weights = 0.5*weights

                integrated_gamma = np.zeros(DIMENSION)

                for s, weight in zip(nodes, weights):

                    state = (
                        start
                        + float(s)*direction
                    )

                    gradient = evaluator.evaluate(
                        state
                    )

                    integrated_gamma += (
                        float(weight)
                        * (Vt @ gradient)
                    )

                contributions = []

                for first, last in BANDS:
                    sl = slice(first-1, last)

                    contribution = float(
                        integrated_gamma[sl]
                        @ alpha[sl]
                    )

                    contributions.append(
                        contribution
                    )

                total = float(
                    np.sum(contributions)
                )

                error = total-exact_change

                order_results[order] = (
                    contributions,
                    total,
                    error,
                )

                convergence_rows.append({
                    "path": path_label,
                    "quadrature_order": order,
                    "exact_inventory_change_kg":
                        exact_change,
                    "integrated_inventory_change_kg":
                        total,
                    "quadrature_error_kg":
                        error,
                    "quadrature_error_pct_true_inventory":
                        100.0*error/TRUE_INVENTORY,
                })

                print(
                    f"  Gauss-{order}: "
                    f"{total:+.9f} kg "
                    f"(error {error:+.3e} kg)"
                )

            contributions_4 = order_results[4][0]
            contributions_8 = order_results[8][0]

            nominal_total = 0.0
            nonlinear_correction_total = 0.0

            print("\n  Singular-band contributions (Gauss-8)")

            for index, (first, last) in enumerate(BANDS):

                sl = slice(first-1, last)

                nominal = float(
                    nominal_gamma[sl]
                    @ alpha[sl]
                )

                nonlinear = float(
                    contributions_8[index]
                )

                correction = nonlinear-nominal

                convergence_difference = (
                    contributions_8[index]
                    - contributions_4[index]
                )

                nominal_total += nominal
                nonlinear_correction_total += correction

                band_rows.append({
                    "path": path_label,
                    "rank_first": first,
                    "rank_last": last,
                    "minimum_singular_value":
                        float(singular_values[sl][-1]),
                    "maximum_singular_value":
                        float(singular_values[sl][0]),
                    "movement_band_norm":
                        float(np.linalg.norm(alpha[sl])),
                    "nominal_inventory_change_kg":
                        nominal,
                    "nonlinear_inventory_change_kg":
                        nonlinear,
                    "nonlinear_correction_kg":
                        correction,
                    "gauss4_inventory_change_kg":
                        float(contributions_4[index]),
                    "gauss8_minus_gauss4_kg":
                        convergence_difference,
                    "nonlinear_change_pct_true_inventory":
                        100.0*nonlinear/TRUE_INVENTORY,
                })

                print(
                    f"  {first:3d}-{last:3d}: "
                    f"nonlinear={nonlinear:+12.6f} kg, "
                    f"correction={correction:+12.6f} kg"
                )

            total_nonlinear = order_results[8][1]

            summary_rows.append({
                "path": path_label,
                "exact_inventory_change_kg":
                    exact_change,
                "nominal_inventory_change_kg":
                    nominal_total,
                "nonlinear_inventory_change_kg":
                    total_nonlinear,
                "nonlinear_correction_kg":
                    nonlinear_correction_total,
                "integration_error_kg":
                    total_nonlinear-exact_change,
                "nonlinear_correction_pct_true_inventory":
                    100.0*nonlinear_correction_total
                    /TRUE_INVENTORY,
            })

            print(
                f"\n  Nominal total: "
                f"{nominal_total:+.9f} kg"
            )
            print(
                f"  Nonlinear correction: "
                f"{nonlinear_correction_total:+.9f} kg"
            )

            if not np.isclose(
                total_nonlinear,
                exact_change,
                rtol=1e-5,
                atol=1e-3,
            ):
                raise RuntimeError(
                    f"Path-integral closure FAILED: "
                    f"{path_label}"
                )

        print(
            f"\nNative full-gradient evaluations: "
            f"{evaluator.calls}"
        )

    pd.DataFrame(band_rows).to_csv(
        outputs["bands"],
        index=False,
        float_format="%.17g",
        mode="x",
    )

    pd.DataFrame(summary_rows).to_csv(
        outputs["summary"],
        index=False,
        float_format="%.17g",
        mode="x",
    )

    pd.DataFrame(convergence_rows).to_csv(
        outputs["convergence"],
        index=False,
        float_format="%.17g",
        mode="x",
    )

    print("\nPASS: M27-A8 nonlinear band-integral closure")

    for output in outputs.values():
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
