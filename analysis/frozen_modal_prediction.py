#!/usr/bin/env python3
"""M1.7f: Frozen-operator rank-one prediction validation."""

from pathlib import Path

import numpy as np
from scipy.linalg import eig

root = Path("benchmarks/results-m17e")

T = np.loadtxt(
    root / "T_800.csv",
    delimiter=","
)

n = 100
dimension = 2*n + 1

permutation = np.r_[
    np.arange(2, dimension, 2),
    np.arange(1, dimension, 2),
    [0],
]

T = T[np.ix_(permutation, permutation)]

scales = np.r_[
    np.full(n, 1e5),
    np.full(n+1, 10.0),
]

T = (
    T * scales[None, :]
    / scales[:, None]
)

values, left, right = eig(
    T,
    left=True,
    right=True
)

stable = np.where(
    np.abs(values) < 1.0
)[0]

index = stable[
    np.argmax(np.abs(values[stable]))
]

mu = values[index]
l = left[:, index]
r = right[:, index]

# scipy.linalg.eig returns left eigenvectors satisfying
# l^H T = mu l^H for real eigenvalues.
denominator = np.vdot(l, r)

if abs(denominator) < 1e-12:
    raise RuntimeError(
        "Nearly defective dominant eigenpair"
    )

# Spectral projector.
P = np.outer(r, np.conjugate(l)) / denominator

# Observation matrix:
# inlet flow and outlet pressure.
#
# In packed coordinates:
# inlet flow index = 2*n
# outlet pressure index = n-1.

C = np.zeros((2, dimension))
C[0, 2*n] = 1.0
C[1, n-1] = 1.0

# Compare in physical observation units:
# inlet flow kg/s, outlet pressure Pa.
C = np.diag([10.0, 1e5]) @ C

rng = np.random.default_rng(20261013)

directions = {}

directions["right_mode"] = np.real(r).copy()

random_direction = rng.normal(size=dimension)
random_direction /= np.linalg.norm(random_direction)

directions["random"] = random_direction

# Remove the dominant modal contribution.
orthogonal = random_direction - np.real(P @ random_direction)

orthogonal /= np.linalg.norm(orthogonal)

directions["left_null"] = orthogonal

horizons = [1, 5, 10, 20, 40, 80, 160, 320]

print("M1.7f: Frozen-operator prediction")
print()

for name, initial in directions.items():

    print(f"Direction: {name}")

    exact = initial.copy()

    for k in range(1, max(horizons)+1):

        exact = T @ exact

        if k not in horizons:
            continue

        reduced = (mu**k) * (P @ initial)

        y_exact = C @ exact
        y_reduced = C @ reduced

        numerator = np.linalg.norm(
            y_exact-y_reduced
        )

        denominator_error = max(
            np.linalg.norm(y_exact),
            1e-14
        )

        relative_error = numerator / denominator_error

        print(
            f"  steps={k:3d} "
            f"time_min={k*15/60:6.2f} "
            f"relative_error={relative_error:.9e}"
        )

    print()

print("PASS M1.7f frozen modal prediction")
