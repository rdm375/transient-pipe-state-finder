#!/usr/bin/env python3
"""Validate M25 joint operational sensitivities against M24."""

from pathlib import Path

import numpy as np
import pandas as pd


M24 = Path(
    "benchmarks/results-m24/operational_sensitivity.csv"
)

M25 = Path(
    "benchmarks/results-m25/operational_sensitivity.csv"
)

NX = 41
N = 20
STEPS = 60
DIMENSION = 163

old = pd.read_csv(M24)
new = pd.read_csv(M25)

expected_rows = (
    (STEPS + 1) * DIMENSION * (2*N + 4)
)

if len(new) != expected_rows:
    raise RuntimeError(
        f"Expected {expected_rows} rows, got {len(new)}"
    )

print()
print("M2.5-B1 JOINT SENSITIVITY VALIDATION")
print("=" * 72)

print("M24 rows:", len(old))
print("M25 rows:", len(new))

# Validate the complete fixed-boundary block.

fixed = new[new.column < NX].copy()

keys = [
    "time_s",
    "column",
    "quantity",
    "index",
]

comparison = old.merge(
    fixed,
    on=keys,
    suffixes=("_m24", "_m25"),
    validate="one_to_one",
)

if len(comparison) != len(old):
    raise RuntimeError(
        "Incomplete M24/M25 comparison"
    )

difference = (
    comparison.sensitivity_m25.to_numpy()
    -
    comparison.sensitivity_m24.to_numpy()
)

maximum_difference = np.max(
    np.abs(difference)
)

scale = max(
    1.0,
    np.max(
        np.abs(
            comparison.sensitivity_m24.to_numpy()
        )
    ),
)

relative_difference = maximum_difference / scale

print()
print("Fixed-boundary parity")
print("-" * 72)
print("Maximum absolute difference:", maximum_difference)
print("Relative difference:", relative_difference)

if relative_difference > 1.0e-10:
    raise RuntimeError(
        "M24/M25 fixed-boundary parity failed"
    )

# Validate direct inlet-pressure derivatives.

pressure = new[
    (new.quantity == "pressure")
    & (new["index"] == 0)
    & (new.column >= NX)
    & (new.column < NX + STEPS + 1)
]

expected = np.where(
    np.rint(pressure.time_s.to_numpy() / 10.0).astype(int)
    == pressure.column.to_numpy() - NX,
    1.0e4,
    0.0,
)

pressure_error = np.max(
    np.abs(
        pressure.sensitivity.to_numpy()
        - expected
    )
)

print()
print("Direct inlet-pressure derivative")
print("-" * 72)
print("Maximum absolute error:", pressure_error)

if pressure_error > 1.0e-8:
    raise RuntimeError(
        "Inlet-pressure derivative failed"
    )

# Validate direct outlet-flow derivatives.

outlet = new[
    (new.quantity == "flow")
    & (new["index"] == N + 1)
    & (new.column >= NX + STEPS + 1)
]

expected = np.where(
    np.rint(outlet.time_s.to_numpy() / 10.0).astype(int)
    == outlet.column.to_numpy() - NX - STEPS - 1,
    1.0,
    0.0,
)

flow_error = np.max(
    np.abs(
        outlet.sensitivity.to_numpy()
        - expected
    )
)

print()
print("Direct outlet-flow derivative")
print("-" * 72)
print("Maximum absolute error:", flow_error)

if flow_error > 1.0e-12:
    raise RuntimeError(
        "Outlet-flow derivative failed"
    )

print()
print("M2.5-B1 VALIDATION PASS")
