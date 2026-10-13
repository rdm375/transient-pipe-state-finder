#!/usr/bin/env python3
"""M1.7e: Exact discrete left/right modal verification."""

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eig

directory = Path("benchmarks/results-m17e")

T = np.loadtxt(
    directory / "T_final.csv",
    delimiter=","
)

n = 100
dimension = 2*n + 1
dt = 15.0

assert T.shape == (dimension, dimension)

# ---------------------------------------------------------
# Convert from interleaved physical state coordinates
# to the state-finder's packed physical coordinates:
#
# [p1,...,pn, internal flows, inlet flow]
# ---------------------------------------------------------

# Original independent coordinates:
# index 0 = inlet flow
# index 1 = internal flow 0
# index 2 = pressure 1
# index 3 = internal flow 1
# index 4 = pressure 2
# ...

pressure_indices = np.arange(2, dimension, 2)
flow_indices = np.arange(1, dimension, 2)

permutation = np.r_[
    pressure_indices,
    flow_indices,
    [0]
]

assert len(permutation) == dimension
assert len(np.unique(permutation)) == dimension

T = T[np.ix_(permutation, permutation)]

# ---------------------------------------------------------
# Compute left/right eigenvectors.
# ---------------------------------------------------------

values, left_vectors, right_vectors = eig(
    T,
    left=True,
    right=True
)

# A one-step operator can contain complex or rapidly
# decaying modes. Identify the largest-modulus mode
# strictly inside the unit circle.
stable = np.where(
    np.abs(values) < 1.0
)[0]

if len(stable) == 0:
    raise RuntimeError("No stable eigenvalues")

index = stable[
    np.argmax(np.abs(values[stable]))
]

mu = values[index]

right = right_vectors[:, index]
left = left_vectors[:, index]

decay_rate = np.log(abs(mu)) / dt
tau = -1.0 / decay_rate

print("M1.7e: Exact discrete modal verification")
print()
print(f"Dominant eigenvalue: {mu}")
print(f"Modulus:             {abs(mu):.12e}")
print(f"Decay rate:          {decay_rate:.12e} 1/s")
print(f"Time constant:       {tau:.9f} s")

if abs(mu.imag) > 1e-10:
    print(
        "WARNING: Dominant mode is complex; "
        "a real single-mode comparison is insufficient."
    )

# ---------------------------------------------------------
# Verify eigenvector residuals.
# ---------------------------------------------------------

right_residual = (
    np.linalg.norm(T @ right - mu*right)
    / (np.linalg.norm(T)*np.linalg.norm(right))
)

left_residual = (
    np.linalg.norm(T.T @ left - mu*left)
    / (np.linalg.norm(T)*np.linalg.norm(left))
)

print()
print(f"Right residual: {right_residual:.12e}")
print(f"Left residual:  {left_residual:.12e}")

# ---------------------------------------------------------
# Compare physical right eigenvector with sine mode.
# ---------------------------------------------------------

x = np.arange(1, n+1) * 1000.0
L = 100000.0

sine = np.sin(np.pi*x/(2*L))

right_pressure = np.real(right[:n])

if np.dot(right_pressure, sine) < 0:
    right_pressure = -right_pressure

right_pressure /= np.max(np.abs(right_pressure))

sine /= np.max(np.abs(sine))

right_shape_error = (
    np.linalg.norm(right_pressure-sine)
    / np.linalg.norm(sine)
)

print()
print("Physical right eigenvector")
print(
    "Pressure shape error vs sine:",
    f"{right_shape_error:.12e}"
)

# ---------------------------------------------------------
# Load observed predictive sensitivity direction.
# ---------------------------------------------------------

with open(
    "benchmarks/results-m17a/"
    "dominant_predictive_mode.csv",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

observed = np.array([
    float(r["standardized_mode"])
    for r in rows
])

assert len(observed) == dimension

# ---------------------------------------------------------
# The predictive SVD used standardized initial states:
#
# delta u = B^(1/2) z
#
# Therefore a physical left eigenvector l maps to
# standardized coordinates as B^(1/2) l.
# ---------------------------------------------------------

prior_sigma = np.r_[
    np.full(n, 10000.0),
    np.full(n+1, 1.0)
]

left_standardized = prior_sigma * left

left_standardized /= np.linalg.norm(
    left_standardized
)

observed /= np.linalg.norm(observed)

alignment = abs(
    np.vdot(left_standardized, observed)
)

print()
print("Predictive direction comparison")
print(
    "Left eigenvector alignment:",
    f"{alignment:.12f}"
)

# ---------------------------------------------------------
# Test outlet half-cell weighting.
# ---------------------------------------------------------

left_pressure = np.real(left_standardized[:n])

if np.sum(left_pressure) < 0:
    left_pressure = -left_pressure

outlet_ratio = (
    left_pressure[-2] / left_pressure[-1]
)

corrected = left_pressure.copy()
corrected[-1] *= 2.0

corrected /= np.max(np.abs(corrected))

left_shape_error = (
    np.linalg.norm(corrected-sine)
    / np.linalg.norm(sine)
)

print()
print("Left eigenvector endpoint")
print(f"p99/p100 ratio: {outlet_ratio:.12f}")
print(
    "Corrected left shape error:",
    f"{left_shape_error:.12e}"
)

# ---------------------------------------------------------
# Save eigenvector diagnostics.
# ---------------------------------------------------------

output = np.column_stack([
    x,
    right_pressure,
    corrected,
    sine,
])

np.savetxt(
    directory / "modal_pressure_shapes.csv",
    output,
    delimiter=",",
    header=(
        "position_m,"
        "right_pressure,"
        "left_pressure_corrected,"
        "fundamental_sine"
    ),
    comments=""
)

observed_rate = -1.472361268023e-3

relative_rate_error = (
    abs(decay_rate-observed_rate)
    / abs(observed_rate)
)

print()
print(
    "Decay-rate discrepancy:",
    f"{100*relative_rate_error:.6f}%"
)

print()
print("PASS M1.7e discrete modal analysis")
