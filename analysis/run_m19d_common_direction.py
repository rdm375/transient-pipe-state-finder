#!/usr/bin/env python3
"""M1.9d: Pairwise principal vectors and three-way common directions."""

import csv
from pathlib import Path

import numpy as np


SOURCE = Path("benchmarks/results-m19a")
OUTPUT = Path("benchmarks/results-m19d")

OUTPUT.mkdir(parents=True, exist_ok=True)

NS = (100, 200, 400)
PAIRS = ((100, 200), (200, 400), (100, 400))

L = 100_000.0
X = np.linspace(0.0, L, 1601)

W = np.ones(X.size)
W[0] = W[-1] = 0.5
W /= X.size - 1

# Includes a field-only comparison and two boundary weights.
BOUNDARY_WEIGHTS = (0.0, 1.0 / 400.0, 1.0)

K = 3


def load_modes(n, boundary_weight):
    vectors = np.loadtxt(
        SOURCE / f"full_n{n}_vectors.csv",
        delimiter=",",
    )

    assert vectors.shape == (2*n + 1, 3)

    dx = L / n

    xp = dx * np.arange(n + 1)
    xm = dx * (np.arange(n) + 0.5)

    columns = []

    for j in range(K):
        v = vectors[:, j]

        p = np.interp(
            X,
            xp,
            np.r_[0.0, v[:n]],
        )

        m = np.interp(
            X,
            xm,
            v[n:2*n],
        )

        parts = [
            np.sqrt(W) * p,
            np.sqrt(W) * m,
        ]

        if boundary_weight > 0:
            parts.append(
                np.array([
                    np.sqrt(boundary_weight) * v[2*n]
                ])
            )

        columns.append(np.concatenate(parts))

    return np.column_stack(columns)


def orthonormalize(A):
    # A = Q R, so a vector Q c corresponds to original
    # Lanczos coefficients a = solve(R, c).
    Q, R = np.linalg.qr(A, mode="reduced")

    singular = np.linalg.svd(A, compute_uv=False)

    if singular[-1] <= 1e-10 * singular[0]:
        raise RuntimeError("Rank-deficient interpolated basis")

    return Q, R


def original_coefficients(R, orthonormal_coefficients):
    coefficients = np.linalg.solve(
        R,
        orthonormal_coefficients,
    )

    # Normalize for reporting; direction is unchanged.
    return coefficients / np.linalg.norm(coefficients)


def fmt(values):
    return "[" + ", ".join(
        f"{value:+.6f}" for value in values
    ) + "]"


def degrees_from_cosine(value):
    return float(np.degrees(
        np.arccos(np.clip(value, 0.0, 1.0))
    ))


def analyze(weight):
    matrices = {
        n: load_modes(n, weight)
        for n in NS
    }

    factors = {
        n: orthonormalize(matrices[n])
        for n in NS
    }

    print()
    print("=" * 76)
    print(f"Boundary weight: {weight:.8g}")
    print("=" * 76)

    pair_rows = []

    for n1, n2 in PAIRS:
        Q1, R1 = factors[n1]
        Q2, R2 = factors[n2]

        U, singular, Vt = np.linalg.svd(
            Q1.T @ Q2
        )

        # First singular-vector pair gives smallest angle.
        c1 = U[:, 0]
        c2 = Vt.T[:, 0]

        a1 = original_coefficients(R1, c1)
        a2 = original_coefficients(R2, c2)

        angle = degrees_from_cosine(singular[0])

        print()
        print(f"Pair {n1} -> {n2}")
        print(f"  Smallest angle: {angle:.9f} degrees")
        print(f"  Original coefficients n={n1}: {fmt(a1)}")
        print(f"  Original coefficients n={n2}: {fmt(a2)}")

        pair_rows.append({
            "boundary_weight": weight,
            "n1": n1,
            "n2": n2,
            "angle_degrees": angle,
            "coefficients_n1": ";".join(map(str, a1)),
            "coefficients_n2": ";".join(map(str, a2)),
        })

    # The sum of the three projectors has rank at most 9.
    #
    # Instead of constructing a large dense matrix, form
    # A = [Q100 Q200 Q400].
    #
    # The nonzero eigenvalues of A A^T are those of A^T A.

    A = np.column_stack([
        factors[n][0]
        for n in NS
    ])

    gram = A.T @ A

    eigenvalues, eigenvectors = np.linalg.eigh(gram)

    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Recover the dominant eigenvector in physical-field space.
    q = A @ eigenvectors[:, 0]
    q /= np.linalg.norm(q)

    consensus = float(eigenvalues[0])

    print()
    print("Three-way consensus")
    print(f"  Projector-sum eigenvalue: {consensus:.12f}")
    print(f"  Deficit from 3: {3.0 - consensus:.12e}")

    consensus_rows = []

    for n in NS:
        Q, R = factors[n]

        projected = Q.T @ q
        cosine = np.linalg.norm(projected)

        angle = degrees_from_cosine(cosine)

        if cosine <= 0:
            raise RuntimeError("Zero consensus projection")

        coefficients = original_coefficients(
            R,
            projected / cosine,
        )

        print(
            f"  n={n}: angle={angle:.9f} degrees, "
            f"coefficients={fmt(coefficients)}"
        )

        consensus_rows.append({
            "boundary_weight": weight,
            "n": n,
            "projector_sum_eigenvalue": consensus,
            "consensus_angle_degrees": angle,
            "coefficients": ";".join(
                map(str, coefficients)
            ),
        })

    # Pairwise angles between the normalized projections
    # of the consensus vector onto each discrete subspace.
    projected_directions = {}

    for n in NS:
        Q, _ = factors[n]

        projected = Q @ (Q.T @ q)
        projected /= np.linalg.norm(projected)

        projected_directions[n] = projected

    print()
    print("Angles between projected consensus directions")

    for n1, n2 in PAIRS:
        cosine = abs(np.dot(
            projected_directions[n1],
            projected_directions[n2],
        ))

        angle = degrees_from_cosine(cosine)

        print(
            f"  {n1}->{n2}: {angle:.9f} degrees"
        )

    return pair_rows, consensus_rows


all_pairs = []
all_consensus = []

for weight in BOUNDARY_WEIGHTS:
    pair_rows, consensus_rows = analyze(weight)

    all_pairs.extend(pair_rows)
    all_consensus.extend(consensus_rows)


def write_csv(name, rows):
    with (OUTPUT / name).open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv("pairwise_principal_vectors.csv", all_pairs)
write_csv("three_way_consensus.csv", all_consensus)

print()
print(f"Results written to {OUTPUT}")
