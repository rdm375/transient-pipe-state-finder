#!/usr/bin/env python3
"""M1.7f: Modal evolution and eigenvalue conditioning."""

from pathlib import Path
import csv

import numpy as np
from scipy.linalg import eig

root = Path("benchmarks/results-m17e")
steps = [20, 40, 80, 200, 400, 600, 800]

n = 100
dimension = 2*n + 1
dt = 15.0

# Physical-to-packed coordinate permutation.
permutation = np.r_[
    np.arange(2, dimension, 2),
    np.arange(1, dimension, 2),
    [0],
]

# Dimensionless state coordinates.
state_scales = np.r_[
    np.full(n, 1e5),
    np.full(n+1, 10.0),
]

# Prior covariance square root in physical coordinates.
prior_sigma = np.r_[
    np.full(n, 1e4),
    np.full(n+1, 1.0),
]

with open(
    "benchmarks/results-m17a/"
    "dominant_predictive_mode.csv",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

predictive = np.array([
    float(row["standardized_mode"])
    for row in rows
])

predictive /= np.linalg.norm(predictive)

observed_rate = -1.472361268023e-3

print("M1.7f: Modal evolution")
print()

print(
    "time_min,lambda_1,tau_s,"
    "condition_number,predictive_alignment,"
    "rate_error_percent"
)

for step in steps:

    T = np.loadtxt(
        root / f"T_{step}.csv",
        delimiter=","
    )

    if T.shape != (dimension, dimension):
        raise RuntimeError(
            f"Invalid matrix at step {step}"
        )

    T = T[np.ix_(permutation, permutation)]

    # Similarity transformation:
    # T_scaled = S^-1 T S.
    T_scaled = (
        T * state_scales[None, :]
        / state_scales[:, None]
    )

    values, left, right = eig(
        T_scaled,
        left=True,
        right=True
    )

    stable = np.where(
        np.abs(values) < 1.0
    )[0]

    if len(stable) == 0:
        raise RuntimeError(
            f"No stable eigenvalue at step {step}"
        )

    index = stable[
        np.argmax(np.abs(values[stable]))
    ]

    mu = values[index]

    lam = np.log(abs(mu)) / dt
    tau = -1.0 / lam

    lv = left[:, index]
    rv = right[:, index]

    inner = np.vdot(lv, rv)

    condition = (
        np.linalg.norm(lv)
        * np.linalg.norm(rv)
        / abs(inner)
    )

    # Left eigenvector in physical coordinates:
    #
    # l_physical = S^-T l_scaled
    #
    # Standardized-prior coordinates:
    #
    # l_prior = B^(1/2) l_physical

    left_prior = (
        prior_sigma / state_scales
    ) * lv

    left_prior /= np.linalg.norm(left_prior)

    alignment = abs(
        np.vdot(left_prior, predictive)
    )

    error = (
        100 * abs(lam-observed_rate)
        / abs(observed_rate)
    )

    print(
        f"{step*dt/60:.1f},"
        f"{lam:.12e},"
        f"{tau:.9f},"
        f"{condition:.9e},"
        f"{alignment:.12f},"
        f"{error:.6f}"
    )

print()
print("PASS M1.7f trajectory modal analysis")
