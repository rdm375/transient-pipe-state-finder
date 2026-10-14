#!/usr/bin/env python3
"""M26-A: Independent Gaussian measurement noise."""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("benchmarks/results-m26")
OUT.mkdir(parents=True, exist_ok=True)

J = pd.read_csv(
    "benchmarks/results-m24/joint_measurement_jacobian.csv"
).to_numpy(dtype=float)

assert J.shape == (244, 163)

# Physical measurement normalizers used by M24.
normalizers = np.array([
    1.0e4,  # inlet pressure, Pa
    1.0,    # inlet flow, kg/s
    1.0e4,  # outlet pressure, Pa
    1.0,    # outlet flow, kg/s
])

# Hypothetical independent measurement standard deviations.
sigma_physical = np.array([
    5.0e3,  # inlet pressure, Pa
    0.5,    # inlet flow, kg/s
    5.0e3,  # outlet pressure, Pa
    0.5,    # outlet flow, kg/s
])

sigma_normalized = sigma_physical / normalizers

assert np.all(sigma_normalized > 0)

sigma_rows = np.tile(sigma_normalized, 61)

# Whitened measurement Jacobian.
A = J / sigma_rows[:, None]

sens = pd.read_csv(
    "benchmarks/results-m25/operational_sensitivity.csv"
)

inventory_rows = sens[
    (sens.time_s == 600.0)
    & (sens.quantity == "inventory")
    & (sens["index"] == 0)
].sort_values("column")

assert np.array_equal(
    inventory_rows.column.to_numpy(),
    np.arange(163)
)

g = inventory_rows.sensitivity.to_numpy(dtype=float)

# Full SVD. No normal-equation inversion.
U, s, Vt = np.linalg.svd(A, full_matrices=False)

assert s.size == 163

projection = Vt @ g

print("M26-A: Independent Gaussian Noise")
print("=" * 72)

print("Jacobian shape:", A.shape)
print("Largest singular value:", s[0])
print("Smallest singular value:", s[-1])
print("Condition number:", s[0]/s[-1])

# Retained-rank analyses.
#
# NOTE:
# A truncated SVD estimates only the retained components.
# The resulting variance is NOT total inventory uncertainty
# unless discarded components are known or constrained.

thresholds = [
    0.0,
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
]

records = []

for rtol in thresholds:
    retained = s > rtol*s[0]

    rank = int(np.count_nonzero(retained))

    sigma_M = np.linalg.norm(
        projection[retained] / s[retained]
    )

    discarded_gradient = np.linalg.norm(
        projection[~retained]
    )

    records.append({
        "relative_threshold": rtol,
        "retained_rank": rank,
        "inventory_sigma_kg": sigma_M,
        "discarded_inventory_gradient_kg":
            discarded_gradient,
    })

df = pd.DataFrame(records)

print()
print(df.to_string(index=False))

df.to_csv(
    OUT / "independent_noise_rank_sweep.csv",
    index=False
)

# Monte Carlo check of the full-rank linearized estimator.
#
# Generate measurement noise in whitened units:
# eta ~ N(0, I).
#
# Delta xi_hat = V diag(1/s) U^T eta.
# Delta M_hat = g^T Delta xi_hat.

rng = np.random.default_rng(20261014)

nsamples = 10000

eta = rng.standard_normal((A.shape[0], nsamples))

inventory_error = (
    (projection/s) @ (U.T @ eta)
)

sigma_predicted = np.linalg.norm(projection/s)

sigma_empirical = np.std(
    inventory_error,
    ddof=1
)

mean_empirical = np.mean(inventory_error)

print()
print("Monte Carlo verification")
print("-" * 72)

print("Samples:", nsamples)
print("Predicted inventory sigma (kg):", sigma_predicted)
print("Empirical inventory sigma (kg):", sigma_empirical)
print("Empirical mean error (kg):", mean_empirical)

print(
    "Relative sigma discrepancy:",
    abs(sigma_empirical/sigma_predicted-1)
)

summary = pd.DataFrame([{
    "samples": nsamples,
    "predicted_sigma_kg": sigma_predicted,
    "empirical_sigma_kg": sigma_empirical,
    "empirical_mean_kg": mean_empirical,
    "relative_sigma_discrepancy":
        abs(sigma_empirical/sigma_predicted-1),
}])

summary.to_csv(
    OUT / "independent_noise_monte_carlo.csv",
    index=False
)

print()
print("Results saved to", OUT)
