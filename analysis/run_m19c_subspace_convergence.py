#!/usr/bin/env python3
"""M1.9c: Spatial convergence of full-window observable subspaces."""

import csv
from pathlib import Path

import numpy as np


SOURCE = Path("benchmarks/results-m19a")
OUTPUT = Path("benchmarks/results-m19c")
OUTPUT.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = (100, 200, 400)
BOUNDARY_WEIGHTS = (0.0, 1/400, 1/200, 1/100, 1.0)

LENGTH = 100_000.0

# Common physical grid.
X = np.linspace(0.0, LENGTH, 1601)

# Trapezoidal quadrature weights, normalized by L.
QUAD = np.ones(len(X))
QUAD[0] = 0.5
QUAD[-1] = 0.5
QUAD /= len(X)-1


def load_vectors(n):
    path = SOURCE / f"full_n{n}_vectors.csv"

    vectors = np.loadtxt(path, delimiter=",")

    assert vectors.shape == (2*n+1, 3)

    dx = LENGTH/n

    xp = dx*np.arange(n+1)
    xm = dx*(np.arange(n)+0.5)

    output = []

    for j in range(3):
        v = vectors[:, j]

        # Prior-standardized physical fields.
        pressure = np.concatenate((
            [0.0],
            v[:n],
        ))

        flow = v[n:2*n]

        output.append({
            "pressure": np.interp(X, xp, pressure),
            "flow": np.interp(X, xm, flow),
            "inlet": v[2*n],
        })

    return output


def embed(modes, k, boundary_weight):
    columns = []

    for mode in modes[:k]:
        pressure = np.sqrt(QUAD)*mode["pressure"]
        flow = np.sqrt(QUAD)*mode["flow"]

        components = [pressure, flow]

        if boundary_weight > 0:
            components.append(
                np.array([
                    np.sqrt(boundary_weight)*mode["inlet"]
                ])
            )

        columns.append(np.concatenate(components))

    return np.column_stack(columns)


def orthonormal_basis(matrix):
    u, singular_values, _ = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    if singular_values[-1] <= (
        1.0e-10*singular_values[0]
    ):
        raise RuntimeError(
            "Numerically rank-deficient subspace"
        )

    return u


def principal_angles(a, b):
    overlap = a.T @ b

    singular_values = np.linalg.svd(
        overlap,
        compute_uv=False,
    )

    singular_values = np.clip(
        singular_values,
        0.0,
        1.0,
    )

    return np.degrees(
        np.arccos(singular_values)
    )


modes = {
    n: load_vectors(n)
    for n in RESOLUTIONS
}

rows = []

for weight in BOUNDARY_WEIGHTS:
    for k in (1, 2, 3):
        bases = {
            n: orthonormal_basis(
                embed(modes[n], k, weight)
            )
            for n in RESOLUTIONS
        }

        for n1, n2 in (
            (100, 200),
            (200, 400),
            (100, 400),
        ):
            angles = principal_angles(
                bases[n1],
                bases[n2],
            )

            row = {
                "boundary_weight": weight,
                "modes": k,
                "n_coarse": n1,
                "n_fine": n2,
                "max_angle_degrees": float(np.max(angles)),
                "rms_angle_degrees": float(
                    np.sqrt(np.mean(angles**2))
                ),
                "angles_degrees": ";".join(
                    f"{a:.12g}" for a in angles
                ),
            }

            rows.append(row)

            print(
                f"wb={weight:8.5f} "
                f"k={k} "
                f"{n1:3d}->{n2:3d} "
                f"max={row['max_angle_degrees']:10.6f} deg "
                f"rms={row['rms_angle_degrees']:10.6f} deg"
            )

with (OUTPUT / "subspace_convergence.csv").open(
    "w", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote results to {OUTPUT}")
