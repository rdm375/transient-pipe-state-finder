#!/usr/bin/env python3
"""M27-A7: Local nonlinear inventory gradients in M27-A5 SVD basis."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular

from m26_correlated_noise_monte_carlo import temporal_covariance


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results-m27"
DIMENSION = 163

LABELS = ("reference", "006", "012", "024")

BANDS = (
    (1, 40),
    (41, 80),
    (81, 120),
    (121, 140),
    (141, 150),
    (151, 160),
    (161, 163),
)


def load_gradient(label):
    path = RESULTS / f"local_inventory_gradient_{label}.csv"
    frame = pd.read_csv(path)

    expected = [
        "coordinate",
        "inventory_gradient_kg",
    ]

    if list(frame.columns) != expected:
        raise RuntimeError(f"Invalid gradient columns: {path}")

    if not np.array_equal(
        frame["coordinate"].to_numpy(),
        np.arange(DIMENSION),
    ):
        raise RuntimeError(f"Invalid gradient coordinates: {path}")

    gradient = frame[
        "inventory_gradient_kg"
    ].to_numpy(dtype=float)

    if not np.all(np.isfinite(gradient)):
        raise RuntimeError(f"Nonfinite gradient: {path}")

    return gradient


def load_solution(label):
    if label == "reference":
        return np.zeros(DIMENSION)

    path = RESULTS / f"native_gn_{label}_solution.csv"
    frame = pd.read_csv(path)

    if not np.array_equal(
        frame["coordinate"].to_numpy(),
        np.arange(DIMENSION),
    ):
        raise RuntimeError(f"Invalid solution coordinates: {path}")

    values = frame[
        "reconstructed_scaled_value"
    ].to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"Nonfinite solution: {path}")

    return values


def main():
    outputs = {
        "bands": RESULTS / "local_inventory_gradient_bands.csv",
        "directions": RESULTS / "local_inventory_gradient_directions.csv",
        "movements": RESULTS / "local_inventory_gradient_movements.csv",
    }

    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")

    jacobian_path = (
        ROOT
        / "benchmarks/results-m24/joint_measurement_jacobian.csv"
    )

    J = pd.read_csv(jacobian_path).to_numpy(dtype=float)

    if J.shape != (244, DIMENSION):
        raise RuntimeError(f"Unexpected Jacobian shape: {J.shape}")

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

    gradients = {
        label: load_gradient(label)
        for label in LABELS
    }

    solutions = {
        label: load_solution(label)
        for label in LABELS
    }

    projected_gradients = {
        label: Vt @ gradients[label]
        for label in LABELS
    }

    projected_solutions = {
        label: Vt @ solutions[label]
        for label in LABELS
    }

    print("M27-A7 local inventory gradient decomposition")
    print(f"  Jacobian shape: {A.shape}")
    print(f"  Condition number: {singular_values[0]/singular_values[-1]:.9e}")

    band_rows = []
    direction_rows = []
    movement_rows = []

    for label in LABELS:
        gamma = projected_gradients[label]
        nominal = projected_gradients["reference"]

        total_energy = float(gamma @ gamma)

        for first, last in BANDS:
            sl = slice(first-1, last)

            band = gamma[sl]
            delta = gamma[sl]-nominal[sl]

            band_rows.append({
                "solution": label,
                "rank_first": first,
                "rank_last": last,
                "gradient_band_norm":
                    float(np.linalg.norm(band)),
                "gradient_band_energy_fraction":
                    float((band @ band)/total_energy),
                "gradient_change_band_norm":
                    float(np.linalg.norm(delta)),
                "gradient_change_band_energy":
                    float(delta @ delta),
                "minimum_singular_value":
                    float(singular_values[sl][-1]),
                "maximum_singular_value":
                    float(singular_values[sl][0]),
            })

        for index in range(DIMENSION):
            direction_rows.append({
                "solution": label,
                "rank": index+1,
                "singular_value":
                    float(singular_values[index]),
                "projected_gradient":
                    float(gamma[index]),
                "projected_gradient_change":
                    float(gamma[index]-nominal[index]),
                "projected_solution":
                    float(projected_solutions[label][index]),
            })

    movement_pairs = (
        ("reference", "006"),
        ("reference", "012"),
        ("reference", "024"),
        ("006", "012"),
        ("012", "024"),
        ("006", "024"),
    )

    for start_label, end_label in movement_pairs:
        dx = solutions[end_label]-solutions[start_label]
        alpha = Vt @ dx

        direct_start = float(gradients[start_label] @ dx)
        direct_end = float(gradients[end_label] @ dx)
        direct_nominal = float(gradients["reference"] @ dx)

        accumulated_start = 0.0
        accumulated_end = 0.0
        accumulated_nominal = 0.0

        print(f"\nMovement {start_label} -> {end_label}")
        print(f"  Nominal derivative: {direct_nominal:+.9f} kg")
        print(f"  Start derivative:   {direct_start:+.9f} kg")
        print(f"  End derivative:     {direct_end:+.9f} kg")

        for first, last in BANDS:
            sl = slice(first-1, last)

            nominal = float(
                projected_gradients["reference"][sl] @ alpha[sl]
            )

            start = float(
                projected_gradients[start_label][sl] @ alpha[sl]
            )

            end = float(
                projected_gradients[end_label][sl] @ alpha[sl]
            )

            accumulated_nominal += nominal
            accumulated_start += start
            accumulated_end += end

            movement_rows.append({
                "movement": f"{start_label}_to_{end_label}",
                "rank_first": first,
                "rank_last": last,
                "movement_band_norm":
                    float(np.linalg.norm(alpha[sl])),
                "nominal_inventory_derivative_kg":
                    nominal,
                "start_inventory_derivative_kg":
                    start,
                "end_inventory_derivative_kg":
                    end,
                "nonlinear_change_start_to_end_kg":
                    end-start,
                "nonlinear_change_nominal_to_end_kg":
                    end-nominal,
            })

            print(
                f"  {first:3d}-{last:3d}: "
                f"start={start:+12.6f} kg, "
                f"end={end:+12.6f} kg, "
                f"change={end-start:+12.6f} kg"
            )

        for name, accumulated, direct in (
            ("nominal", accumulated_nominal, direct_nominal),
            ("start", accumulated_start, direct_start),
            ("end", accumulated_end, direct_end),
        ):
            if not np.isclose(
                accumulated,
                direct,
                rtol=1e-10,
                atol=1e-8,
            ):
                raise RuntimeError(
                    f"Band reconstruction failed: "
                    f"{start_label}->{end_label}, {name}"
                )

    pd.DataFrame(band_rows).to_csv(
        outputs["bands"],
        index=False,
        float_format="%.17g",
        mode="x",
    )

    pd.DataFrame(direction_rows).to_csv(
        outputs["directions"],
        index=False,
        float_format="%.17g",
        mode="x",
    )

    pd.DataFrame(movement_rows).to_csv(
        outputs["movements"],
        index=False,
        float_format="%.17g",
        mode="x",
    )

    print("\nPASS: All singular-band reconstruction identities")
    for path in outputs.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
