#!/usr/bin/env python3
"""M2.4-F2: Conservative measurement-conditioned uncertainty bounds.

Fixed boundary histories.
Linearized model.
Retrospective full-window measurements.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("benchmarks/results-m24")

J_PATH = ROOT / "joint_measurement_jacobian.csv"
G_PATH = ROOT / "operational_sensitivity.csv"
OUT_PATH = ROOT / "measurement_conditioned_certificate.csv"

NX = 41
RADIUS = 1.0

# Experimental bounds on the Euclidean norm of the
# complete normalized measurement residual vector.
#
# These are NOT calibrated sensor-noise confidence limits.
MEASUREMENT_BOUNDS = [1.0, 5.0, 15.0]

TOLERANCES = {
    "pressure": 100.0,
    "flow": 0.01,
    "inventory": 10.0,
}

OUTPUT_ROWS = {
    "pressure": 21,
    "flow": 22,
    "inventory": 1,
}

ALPHAS = np.linspace(0.01, 0.99, 99)


def load_jacobian():
    # M2.4-B exporter writes a numerical matrix.
    # Accept either plain CSV or CSV with a header.
    try:
        J = np.loadtxt(J_PATH, delimiter=",")
    except ValueError:
        J = np.loadtxt(
            J_PATH, delimiter=",", skiprows=1
        )

    if J.shape != (244, 163):
        raise ValueError(
            f"Unexpected Jacobian shape: {J.shape}"
        )

    if not np.isfinite(J).all():
        raise ValueError("Nonfinite Jacobian")

    return J[:, :NX]


def build_sensitivities():
    df = pd.read_csv(G_PATH)

    times = np.sort(df.time_s.unique())

    matrices = {}

    for t in times:
        for quantity, rows in OUTPUT_ROWS.items():
            subset = df[
                (df.time_s == t) &
                (df.quantity == quantity)
            ]

            G = np.full((rows, NX), np.nan)

            for row in subset.itertuples(index=False):
                i = int(row.index)
                j = int(row.column)

                if not (0 <= i < rows and 0 <= j < NX):
                    raise ValueError(
                        "Sensitivity index out of range"
                    )

                if np.isfinite(G[i, j]):
                    raise ValueError(
                        "Duplicate sensitivity entry"
                    )

                G[i, j] = row.sensitivity

            if not np.isfinite(G).all():
                raise ValueError(
                    f"Incomplete {quantity} at t={t}"
                )

            matrices[(t, quantity)] = G

    return times, matrices


def ellipsoid_row_bounds(G, H):
    """Compute sqrt(diag(G H^-1 G^T)) using Cholesky."""
    L = np.linalg.cholesky(H)

    # Solve L X = G^T.
    X = np.linalg.solve(L, G.T)

    return np.linalg.norm(X, axis=0)


Jx = load_jacobian()
times, matrices = build_sensitivities()

JTJ = Jx.T @ Jx
I = np.eye(NX)

records = []

print("\nM2.4-F2 MEASUREMENT-CONDITIONED UNCERTAINTY")
print("=" * 76)
print(f"Initial-state dimension: {NX}")
print(f"Initial uncertainty radius: {RADIUS}")
print(f"Measurement Jacobian: {Jx.shape}")
print(f"Time samples: {len(times)}")

for eps in MEASUREMENT_BOUNDS:

    # For each operational row, retain the smallest
    # conservative bound across the alpha sweep.
    best = {}

    for t in times:
        for quantity, rows in OUTPUT_ROWS.items():
            best[(t, quantity)] = np.full(
                rows, np.inf
            )

    for alpha in ALPHAS:
        H = (
            alpha * JTJ / eps**2
            + (1.0-alpha) * I / RADIUS**2
        )

        for t in times:
            for quantity in OUTPUT_ROWS:
                G = matrices[(t, quantity)]

                bounds = ellipsoid_row_bounds(G, H)

                best[(t, quantity)] = np.minimum(
                    best[(t, quantity)],
                    bounds
                )

    normalized = []

    for t in times:
        record = {
            "measurement_bound": eps,
            "time_s": t,
        }

        ratios = []

        for quantity, tolerance in TOLERANCES.items():
            bound = float(
                np.max(best[(t, quantity)])
            )

            record[f"{quantity}_bound"] = bound
            record[f"{quantity}_normalized"] = (
                bound / tolerance
            )

            ratios.append(bound / tolerance)

        record["combined_normalized"] = max(ratios)

        records.append(record)
        normalized.append(max(ratios))

    normalized = np.asarray(normalized)

    future_max = np.maximum.accumulate(
        normalized[::-1]
    )[::-1]

    valid = np.flatnonzero(future_max <= 1.0)

    settling = (
        float(times[valid[0]])
        if valid.size
        else np.nan
    )

    final = [
        record for record in records
        if record["measurement_bound"] == eps
    ][-1]

    print(f"\nMeasurement residual bound: {eps:g}")
    print("-" * 76)

    for quantity in TOLERANCES:
        print(
            f"  {quantity:12s}"
            f" final_bound="
            f"{final[f'{quantity}_bound']:.9g}"
            f" normalized="
            f"{final[f'{quantity}_normalized']:.6g}"
        )

    if np.isnan(settling):
        print("  Joint settling: NOT REACHED BY 600 s")
    else:
        print(f"  Joint settling: {settling:g} s")

pd.DataFrame(records).to_csv(
    OUT_PATH, index=False
)

print("\nClassification:")
print("  Conservative linearized upper bounds")
print("  Fixed boundary histories")
print("  Full-window retrospective measurement conditioning")
print("  Experimental uncertainty assumptions")
print("  No nonlinear guarantee")

print("\nSaved:", OUT_PATH)
