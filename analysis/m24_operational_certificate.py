#!/usr/bin/env python3
"""M2.4-F1: Worst-case fixed-boundary linearized uncertainty."""

from pathlib import Path
import numpy as np
import pandas as pd

directory = Path("benchmarks/results-m24")

source = directory / "operational_sensitivity.csv"
output = directory / "operational_certificate.csv"

df = pd.read_csv(source)

INITIAL_DIMENSION = 41

# Dimensionless initial-state uncertainty ball.
# Experimental assumption, not yet calibrated to a prior.
RADIUS = 1.0

# Illustrative operational tolerances.
TOLERANCES = {
    "pressure": 100.0,   # Pa
    "flow": 0.01,        # kg/s
    "inventory": 10.0,   # kg
}

EXPECTED_OUTPUTS = {
    "pressure": 21,
    "flow": 22,
    "inventory": 1,
}

def matrix_at_time(time, quantity):
    subset = df[
        (df.time_s == time) &
        (df.quantity == quantity)
    ]

    rows = EXPECTED_OUTPUTS[quantity]

    matrix = np.full(
        (rows, INITIAL_DIMENSION),
        np.nan
    )

    for row in subset.itertuples(index=False):
        i = int(row.index)
        j = int(row.column)

        if not (0 <= i < rows):
            raise ValueError("Invalid output index")

        if not (0 <= j < INITIAL_DIMENSION):
            raise ValueError("Invalid column index")

        if np.isfinite(matrix[i, j]):
            raise ValueError("Duplicate sensitivity entry")

        matrix[i, j] = row.sensitivity

    if not np.isfinite(matrix).all():
        raise ValueError(
            f"Incomplete matrix at t={time}, {quantity}"
        )

    return matrix


times = np.sort(df.time_s.unique())

records = []

for time in times:
    record = {"time_s": time}

    for quantity, tolerance in TOLERANCES.items():
        G = matrix_at_time(time, quantity)

        # Exact induced 2-to-infinity norm:
        # max over rows of their Euclidean norms.
        row_norms = np.linalg.norm(G, axis=1)

        worst_row = int(np.argmax(row_norms))

        bound = RADIUS * row_norms[worst_row]

        record[f"{quantity}_bound"] = bound
        record[f"{quantity}_worst_index"] = worst_row
        record[f"{quantity}_normalized"] = (
            bound / tolerance
        )

    record["combined_normalized"] = max(
        record[f"{q}_normalized"]
        for q in TOLERANCES
    )

    records.append(record)

result = pd.DataFrame(records)

# Sustained criterion over the recorded time window.
values = result["combined_normalized"].to_numpy()

future_max = np.maximum.accumulate(
    values[::-1]
)[::-1]

valid = np.flatnonzero(future_max <= 1.0)

settling_time = (
    float(times[valid[0]])
    if valid.size
    else np.nan
)

result["future_max_normalized"] = future_max

result.to_csv(output, index=False)

print()
print("M2.4-F1 FIXED-BOUNDARY UNCERTAINTY")
print("=" * 72)

print(f"Initial-state dimension: {INITIAL_DIMENSION}")
print(f"Scaled uncertainty radius: {RADIUS}")
print(f"Time horizon: {times[0]}–{times[-1]} s")

print()
print("WORST-CASE LINEARIZED ERRORS")
print("-" * 72)

for quantity, tolerance in TOLERANCES.items():
    column = f"{quantity}_bound"

    maximum = result[column].max()
    final = result[column].iloc[-1]

    print(
        f"{quantity:12s} "
        f"peak={maximum:.9g} "
        f"final={final:.9g} "
        f"tolerance={tolerance:g}"
    )

print()
print("RELIABILITY")
print("-" * 72)

if np.isnan(settling_time):
    print(
        "No joint reliability horizon established "
        "within the simulated window."
    )
else:
    print(
        f"Joint settling time: {settling_time:g} s"
    )

print()
print("Certificate classification:")
print("  LINEARIZED")
print("  FIXED BOUNDARIES")
print("  SCALED L2 INITIAL-STATE BALL")
print("  NOT MEASUREMENT-CONDITIONED")
print("  NOT A NONLINEAR GUARANTEE")

print()
print("Saved:", output)
