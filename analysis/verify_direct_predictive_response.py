#!/usr/bin/env python3
"""M1.7h: Verify direct tangent response against H @ direction."""

from pathlib import Path

import numpy as np

ROOT = Path("benchmarks/results-m17e")

H = np.loadtxt(
    "benchmarks/results-m17a/H_800.csv",
    delimiter=","
)

direction = np.loadtxt(
    ROOT / "predictive_direction.txt"
)

direct = np.genfromtxt(
    ROOT / "direct_predictive_response.csv",
    delimiter=",",
    names=True
)

assert H.shape == (1602, 201)
assert direction.shape == (201,)
assert len(direct) == 801

assert np.allclose(
    direct["time_s"],
    np.arange(801)*15.0,
    rtol=0.0,
    atol=1e-12
)

# Matrix rows alternate:
# inlet flow, outlet pressure.

reference = (H @ direction).reshape(801, 2)

# Convert from original observation normalization
# to physical units.
reference[:, 0] *= 1.0
reference[:, 1] *= 1e4

computed = np.column_stack([
    direct["inlet_flow_kg_s"],
    direct["outlet_pressure_pa"]
])

assert np.all(np.isfinite(computed))

error = computed - reference

noise = np.array([0.01, 10.0])

scaled_error = error / noise[None, :]
scaled_reference = reference / noise[None, :]

max_flow_error = np.max(
    np.abs(error[:, 0])
)

max_pressure_error = np.max(
    np.abs(error[:, 1])
)

max_noise_error = np.max(
    np.linalg.norm(scaled_error, axis=1)
)

relative_frobenius = (
    np.linalg.norm(scaled_error)
    / np.linalg.norm(scaled_reference)
)

print("M1.7h: Direct predictive response verification")
print()

print(
    "max_flow_error_kg_s="
    f"{max_flow_error:.12e}"
)

print(
    "max_pressure_error_pa="
    f"{max_pressure_error:.12e}"
)

print(
    "max_noise_normalized_error="
    f"{max_noise_error:.12e}"
)

print(
    "relative_frobenius_error="
    f"{relative_frobenius:.12e}"
)

# Numerical acceptance thresholds.
#
# These are conservative verification thresholds,
# not a claim that the underlying arithmetic
# achieves machine precision in every component.

FLOW_TOL = 1e-7
PRESSURE_TOL = 1e-3
RELATIVE_TOL = 1e-8

assert max_flow_error < FLOW_TOL, (
    "Flow sensitivity mismatch"
)

assert max_pressure_error < PRESSURE_TOL, (
    "Pressure sensitivity mismatch"
)

assert relative_frobenius < RELATIVE_TOL, (
    "Relative sensitivity mismatch"
)

print()
print("PASS M1.7h direct/matrix tangent equivalence")
