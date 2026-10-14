#!/usr/bin/env python3
"""M26-A2: Joint versus fixed-boundary inventory variance."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("benchmarks/results-m26")
OUT.mkdir(parents=True, exist_ok=True)

NX = 41
NB = 122
N = NX + NB

# M24 normalized measurement Jacobian.
J = pd.read_csv(
    "benchmarks/results-m24/joint_measurement_jacobian.csv"
).to_numpy(dtype=float)

assert J.shape == (244, N)
assert np.all(np.isfinite(J))

# Illustrative M26-A independent measurement noise.
#
# Physical channel standard deviations:
#   inlet pressure:  5000 Pa
#   inlet flow:         0.5 kg/s
#   outlet pressure: 5000 Pa
#   outlet flow:        0.5 kg/s
#
# Corresponding M24 measurement normalizers:
#   pressure: 10000 Pa
#   flow:        1 kg/s

sigma_physical = np.array([
    5000.0,
    0.5,
    5000.0,
    0.5,
])

normalizers = np.array([
    10000.0,
    1.0,
    10000.0,
    1.0,
])

sigma_normalized = sigma_physical / normalizers

sigma_rows = np.tile(sigma_normalized, 61)

A = J / sigma_rows[:, None]

# Terminal inventory gradient from M25.
sens = pd.read_csv(
    "benchmarks/results-m25/operational_sensitivity.csv"
)

rows = sens[
    (sens.time_s == 600.0)
    & (sens.quantity == "inventory")
    & (sens["index"] == 0)
].sort_values("column")

assert np.array_equal(
    rows.column.to_numpy(dtype=int),
    np.arange(N)
)

g = rows.sensitivity.to_numpy(dtype=float)

gx = g[:NX]
gb = g[NX:]

# Full joint covariance factor.
U, s, Vt = np.linalg.svd(A, full_matrices=False)

assert s.size == N
assert np.all(s > 0)

B = Vt.T / s[None, :]

# Project each gradient component into the covariance factor.
#
# This avoids forming the ill-conditioned normal equations.

qx = B[:NX, :].T @ gx
qb = B[NX:, :].T @ gb

Vx = float(qx @ qx)
Vb = float(qb @ qb)
Vcross = float(2.0 * (qx @ qb))

# Combine covariance-factor projections before squaring.
# This avoids cancellation in the three-term variance sum.
qtotal = qx + qb

Vtotal = float(qtotal @ qtotal)

# Independent arithmetic path for consistency diagnostics.
Vdirect = float((B.T @ g) @ (B.T @ g))

Vdecomposed = Vx + Vb + Vcross

print()
print("Covariance decomposition diagnostics")
print("-" * 76)
print(f"Vx:       {Vx:.17e}")
print(f"Vb:       {Vb:.17e}")
print(f"Vcross:   {Vcross:.17e}")
print(f"Vtotal:   {Vtotal:.17e}")
print(f"Vdirect:  {Vdirect:.17e}")
print(f"Difference: {Vtotal - Vdirect:.17e}")
print(
    "Relative difference:",
    abs(Vtotal - Vdirect) / max(abs(Vdirect), 1.0)
)
print(
    "Cancellation factor:",
    (abs(Vx) + abs(Vb) + abs(Vcross))
    / max(abs(Vdirect), 1.0)
)

# Fixed-boundary conditional estimator.
Ax = A[:, :NX]

Ux, sx, Vtx = np.linalg.svd(
    Ax,
    full_matrices=False
)

assert sx.size == NX
assert np.all(sx > 0)

projection_fixed = Vtx @ gx

Vfixed = float(
    np.sum((projection_fixed / sx)**2)
)

# Statistical comparison.
sigma_joint = np.sqrt(Vtotal)
sigma_fixed = np.sqrt(Vfixed)

print()
print("M26-A2: Terminal Linepack Variance Decomposition")
print("=" * 76)

print()
print("Full joint estimator")
print("-" * 76)

print(f"Initial-state variance:       {Vx:18.9f} kg^2")
print(f"Boundary-history variance:    {Vb:18.9f} kg^2")
print(f"Cross-covariance contribution:{Vcross:18.9f} kg^2")
print(f"Total inventory variance:     {Vtotal:18.9f} kg^2")
print(f"Inventory standard deviation: {sigma_joint:18.9f} kg")

print()
print("Fixed-boundary conditional estimator")
print("-" * 76)

print(f"Inventory variance:          {Vfixed:18.9f} kg^2")
print(f"Inventory standard deviation:{sigma_fixed:18.9f} kg")

print()
print("Comparison")
print("-" * 76)

print(
    "Joint/fixed standard deviation ratio:",
    sigma_joint / sigma_fixed
)

print(
    "Joint/fixed variance ratio:",
    Vtotal / Vfixed
)

print()
print("Covariance-factor consistency:")
print("Decomposed variance:", Vtotal)
print("Direct variance:    ", Vdirect)
print("Absolute difference:", abs(Vtotal - Vdirect))

summary = pd.DataFrame([{
    "initial_variance_kg2": Vx,
    "boundary_variance_kg2": Vb,
    "cross_variance_kg2": Vcross,
    "joint_variance_kg2": Vtotal,
    "joint_sigma_kg": sigma_joint,
    "fixed_boundary_variance_kg2": Vfixed,
    "fixed_boundary_sigma_kg": sigma_fixed,
    "joint_fixed_sigma_ratio":
        sigma_joint / sigma_fixed,
    "joint_fixed_variance_ratio":
        Vtotal / Vfixed,
    "variance_consistency_abs":
        abs(Vtotal - Vdirect),
}])

summary.to_csv(
    OUT / "inventory_variance_decomposition.csv",
    index=False
)

print()
print(
    "Saved:",
    OUT / "inventory_variance_decomposition.csv"
)
