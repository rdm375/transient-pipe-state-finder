#!/usr/bin/env python3

from pathlib import Path
import csv
import numpy as np

root = Path("benchmarks/results-m16b")
root.mkdir(parents=True, exist_ok=True)

horizons = [80, 100, 200, 400, 800]

rows = []

for horizon in horizons:
    path = root / f"H_{horizon}.csv"

    H = np.loadtxt(path, delimiter=",")

    expected_shape = (2 * (horizon + 1), 201)

    if H.shape != expected_shape:
        raise RuntimeError(
            f"{path}: {H.shape} != {expected_shape}"
        )

    if not np.all(np.isfinite(H)):
        raise RuntimeError(
            f"{path}: nonfinite sensitivity entries"
        )

    # Full SVD, including a basis for the nullspace
    # when there are fewer observations than unknowns.
    U, s, Vt = np.linalg.svd(
        H,
        full_matrices=True
    )

    smax = float(s[0])

    # Standard numerical-rank threshold.
    eps = np.finfo(float).eps

    tolerance = (
        max(H.shape) * eps * smax
    )

    rank = int(np.count_nonzero(s > tolerance))

    nullity = H.shape[1] - rank

    smallest = (
        float(s[-1])
        if H.shape[0] >= H.shape[1]
        else 0.0
    )

    condition = (
        smax / smallest
        if smallest > tolerance
        else float("inf")
    )

    # Explicit nullspace / weakest right singular
    # direction. The last row of Vt is valid even
    # for a wide matrix because full_matrices=True.
    weak = Vt[-1, :]

    pressure_fraction = float(
        np.sum(weak[:100] ** 2)
    )

    internal_flow_fraction = float(
        np.sum(weak[100:200] ** 2)
    )

    inlet_flow_fraction = float(
        weak[200] ** 2
    )

    # The singular vectors have unit norm.
    assert np.isclose(
        pressure_fraction +
        internal_flow_fraction +
        inlet_flow_fraction,
        1.0,
        atol=1e-10
    )

    # For wide matrices, NumPy returns only min(m,n)
    # singular values. Structural zero singular values
    # are included explicitly in the saved spectrum.
    spectrum = np.zeros(201)
    spectrum[:len(s)] = s

    np.savetxt(
        root / f"singular_values_{horizon}.csv",
        spectrum,
        delimiter=",",
        fmt="%.17e"
    )

    np.savetxt(
        root / f"weakest_direction_{horizon}.csv",
        weak,
        delimiter=",",
        fmt="%.17e"
    )

    rows.append({
        "steps": horizon,
        "minutes": horizon * 15 / 60,
        "rows": H.shape[0],
        "columns": H.shape[1],
        "rank": rank,
        "nullity": nullity,
        "sigma_max": smax,
        "sigma_min": smallest,
        "condition": condition,
        "weak_pressure_fraction": pressure_fraction,
        "weak_internal_flow_fraction":
            internal_flow_fraction,
        "weak_inlet_flow_fraction":
            inlet_flow_fraction,
    })

    print()
    print(f"Horizon: {horizon} steps")
    print(f"  Matrix: {H.shape}")
    print(f"  Numerical rank: {rank}")
    print(f"  Numerical nullity: {nullity}")
    print(f"  Largest singular value: {smax:.8e}")
    print(f"  Smallest singular value: {smallest:.8e}")
    print(f"  Condition number: {condition:.8e}")
    print(
        "  Weak direction energy:"
        f" pressure={pressure_fraction:.6f},"
        f" internal_flow={internal_flow_fraction:.6f},"
        f" inlet_flow={inlet_flow_fraction:.6f}"
    )

with (root / "observability_summary.csv").open(
    "w", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(rows[0])
    )

    writer.writeheader()
    writer.writerows(rows)

print()
print(
    "Wrote:",
    root / "observability_summary.csv"
)
