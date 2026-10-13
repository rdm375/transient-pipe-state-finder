#!/usr/bin/env python3
"""Boundary-only linearized initial-state regularization comparison.

Methods:
  - unregularized minimum-norm least squares
  - zeroth-order Tikhonov
  - second-difference curvature
  - truncated SVD
  - hybrid Tikhonov + curvature

Reports both synthetic-truth oracle selection and discrepancy selection.
"""

from pathlib import Path
import csv
import itertools

import numpy as np
from scipy.linalg import svd

ROOT = Path("benchmarks/results-m16b-analytic")
OUT = ROOT / "regularization-comparison-v2"
OUT.mkdir(parents=True, exist_ok=True)

N = 100
D = 2 * N + 1
HORIZON = 80

H = np.loadtxt(ROOT / f"H_{HORIZON}.csv", delimiter=",")
assert H.shape == (2 * (HORIZON + 1), D), H.shape

M = H.shape[0]

# Pressure and mass-flow initial-state scales.
PRESSURE_SCALE = 1e5
FLOW_SCALE = 10.0

# Experimental settings.
SEEDS = (11, 29, 47)
NOISE_LEVELS = (0.0, 1e-4, 1e-2)
TRUTH_TYPES = ("smooth", "oscillatory", "localized")
REFERENCE_TYPES = ("correct", "biased")

LAMBDAS = np.logspace(-10, 2, 11)
HYBRID_LAMBDAS = np.logspace(-8, 1, 5)
TSVD_RANKS = (10, 20, 30, 40, 50, 60, 69, 78, 90, 110, 140, 160)

# Fixed SVD truncation for the nominally unregularized baseline.
BASELINE_RTOL = 1e-14

# Full SVD of the boundary-only observation matrix.
U, s, Vt = svd(H, full_matrices=False)

print(f"H shape: {H.shape}")
print(f"Maximum singular value: {s[0]:.8e}")
print(f"Minimum singular value: {s[-1]:.8e}")
print(f"Effective rank (relative 1e-10): "
      f"{np.count_nonzero(s > s[0] * 1e-10)}")

# Spatial second differences on pressure and internal face flow.
D2 = np.zeros((2 * (N - 2), D))

for offset, row_offset in ((0, 0), (N, N - 2)):
    for i in range(N - 2):
        D2[row_offset + i, offset + i:offset + i + 3] = (
            1.0, -2.0, 1.0
        )

D2 /= np.linalg.norm(D2, ord=2)

I = np.eye(D)

# We construct linear reconstruction operators once.
# Every candidate maps the normalized observation residual b
# to a scaled state correction x = K @ b.
#
# The augmented least-squares systems are solved using SVD.
# This avoids explicitly squaring the condition number through H.T @ H.

def augmented_operator(regularizers):
    blocks = [H]
    for weight, operator in regularizers:
        if weight > 0:
            blocks.append(np.sqrt(weight) * operator)

    A = np.vstack(blocks)
    Ua, sa, Vta = svd(A, full_matrices=False)

    cutoff = np.finfo(float).eps * max(A.shape) * sa[0]
    keep = sa > cutoff

    # Only the first M rows of the augmented RHS are nonzero.
    return (
        (Vta[keep, :].T / sa[keep])
        @ Ua[:M, keep].T
    )

candidates = []

# Unregularized baseline with an explicit numerical cutoff.
keep = s > BASELINE_RTOL * s[0]
K = (Vt[keep, :].T / s[keep]) @ U[:, keep].T

candidates.append({
    "method": "unregularized",
    "parameter": f"rtol={BASELINE_RTOL:g}",
    "K": K,
})

# Truncated SVD.
for rank in TSVD_RANKS:
    r = min(rank, len(s))
    K = (Vt[:r, :].T / s[:r]) @ U[:, :r].T

    candidates.append({
        "method": "tsvd",
        "parameter": f"rank={r}",
        "K": K,
    })

# Standard Tikhonov.
for lam in LAMBDAS:
    candidates.append({
        "method": "tikhonov",
        "parameter": f"lambda={lam:.8e}",
        "K": augmented_operator([(lam, I)]),
    })

# Curvature regularization.
for lam in LAMBDAS:
    candidates.append({
        "method": "curvature",
        "parameter": f"lambda={lam:.8e}",
        "K": augmented_operator([(lam, D2)]),
    })

# Hybrid Tikhonov + curvature.
for lam0, lam2 in itertools.product(
    HYBRID_LAMBDAS, HYBRID_LAMBDAS
):
    candidates.append({
        "method": "hybrid",
        "parameter": f"lambda0={lam0:.8e};lambda2={lam2:.8e}",
        "K": augmented_operator([
            (lam0, I),
            (lam2, D2),
        ]),
    })

print(f"Candidate configurations: {len(candidates)}")

grid = (np.arange(N) + 1) / N
faces = (np.arange(N) + 0.5) / N

def truth_profile(kind):
    x = np.zeros(D)

    if kind == "smooth":
        x[:N] = 0.03 * np.sin(2 * np.pi * grid)
        x[N:2*N] = 0.04 * np.cos(np.pi * faces)
        x[-1] = 0.025

    elif kind == "oscillatory":
        x[:N] = 0.025 * np.sin(20 * np.pi * grid)
        x[N:2*N] = 0.04 * np.cos(18 * np.pi * faces)
        x[-1] = 0.025

    elif kind == "localized":
        x[:N] = 0.04 * np.exp(
            -0.5 * ((grid - 0.4) / 0.06)**2
        )
        x[N:2*N] = -0.05 * np.exp(
            -0.5 * ((faces - 0.6) / 0.08)**2
        )
        x[-1] = 0.025

    else:
        raise ValueError(kind)

    return x

def reference_profile(kind, truth):
    if kind == "correct":
        return np.zeros(D)

    if kind == "biased":
        # Deliberately incorrect prior, shared across truth cases.
        reference = np.zeros(D)
        reference[:N] = 0.015 * np.sin(np.pi * grid)
        reference[N:2*N] = -0.02 * np.cos(2 * np.pi * faces)
        reference[-1] = -0.01
        return reference

    raise ValueError(kind)

def error_metrics(estimate, truth):
    error = estimate - truth

    return {
        "scaled_state_error": float(np.linalg.norm(error)),
        "pressure_rmse_pa": float(
            PRESSURE_SCALE * np.sqrt(np.mean(error[:N]**2))
        ),
        "flow_rmse_kg_s": float(
            FLOW_SCALE * np.sqrt(np.mean(error[N:2*N]**2))
        ),
        "inlet_flow_error_kg_s": float(
            FLOW_SCALE * abs(error[-1])
        ),
    }

rows = []

# All methods receive the same observation vector for a given trial.
for truth_kind in TRUTH_TYPES:
    truth = truth_profile(truth_kind)

    for noise_level in NOISE_LEVELS:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            noise = noise_level * rng.standard_normal(M)

            observed = H @ truth + noise

            for reference_kind in REFERENCE_TYPES:
                reference = reference_profile(reference_kind, truth)

                # Reconstruct the correction relative to the reference.
                b = observed - H @ reference

                for candidate in candidates:
                    correction = candidate["K"] @ b
                    estimate = reference + correction

                    residual = H @ estimate - observed

                    rows.append({
                        "truth": truth_kind,
                        "reference": reference_kind,
                        "noise": noise_level,
                        "seed": seed,
                        "method": candidate["method"],
                        "parameter": candidate["parameter"],
                        "residual_norm": float(
                            np.linalg.norm(residual)
                        ),
                        **error_metrics(estimate, truth),
                    })

fields = list(rows[0])

with open(OUT / "all_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# For each trial and method, select a regularization parameter.
#
# Oracle: minimizes true scaled-state error.
#
# Discrepancy: selects the strongest regularization among candidates
# whose residual does not exceed sqrt(M)*noise_level, preferring
# residuals nearest that target. This uses an assumed noise scale,
# not the true internal state.
#
# If no candidate meets the target, choose the closest residual.
#
# For zero noise, discrepancy selection is not meaningful;
# report only oracle selection.

selected = []

methods = (
    "unregularized",
    "tikhonov",
    "curvature",
    "tsvd",
    "hybrid",
)

for truth_kind in TRUTH_TYPES:
    for reference_kind in REFERENCE_TYPES:
        for noise_level in NOISE_LEVELS:
            for seed in SEEDS:
                for method in methods:
                    subset = [
                        row for row in rows
                        if row["truth"] == truth_kind
                        and row["reference"] == reference_kind
                        and row["noise"] == noise_level
                        and row["seed"] == seed
                        and row["method"] == method
                    ]

                    oracle = min(
                        subset,
                        key=lambda r: r["scaled_state_error"]
                    )

                    selected.append({
                        **oracle,
                        "selection": "oracle",
                    })

                    if noise_level > 0:
                        target = np.sqrt(M) * noise_level

                        eligible = [
                            r for r in subset
                            if r["residual_norm"] <= target
                        ]

                        pool = eligible if eligible else subset

                        discrepancy = min(
                            pool,
                            key=lambda r: abs(
                                r["residual_norm"] - target
                            )
                        )

                        selected.append({
                            **discrepancy,
                            "selection": "discrepancy",
                        })

with open(OUT / "selected_results.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(selected[0])
    )
    writer.writeheader()
    writer.writerows(selected)

print("\nMedian scaled-state error across three noise seeds")
print("Selection: oracle (synthetic-truth diagnostic)")
print()
print(
    f"{'truth':12s} {'reference':9s} {'noise':>8s} "
    f"{'method':15s} {'median error':>14s}"
)

for truth_kind in TRUTH_TYPES:
    for reference_kind in REFERENCE_TYPES:
        for noise_level in NOISE_LEVELS:
            for method in methods:
                values = [
                    row["scaled_state_error"]
                    for row in selected
                    if row["truth"] == truth_kind
                    and row["reference"] == reference_kind
                    and row["noise"] == noise_level
                    and row["method"] == method
                    and row["selection"] == "oracle"
                ]

                print(
                    f"{truth_kind:12s} "
                    f"{reference_kind:9s} "
                    f"{noise_level:8.0e} "
                    f"{method:15s} "
                    f"{np.median(values):14.6e}"
                )

print("\nDiscrepancy-principle selection, nonzero noise")
print(
    f"{'truth':12s} {'reference':9s} {'noise':>8s} "
    f"{'method':15s} {'median error':>14s}"
)

for truth_kind in TRUTH_TYPES:
    for reference_kind in REFERENCE_TYPES:
        for noise_level in NOISE_LEVELS:
            if noise_level == 0:
                continue

            for method in methods:
                values = [
                    row["scaled_state_error"]
                    for row in selected
                    if row["truth"] == truth_kind
                    and row["reference"] == reference_kind
                    and row["noise"] == noise_level
                    and row["method"] == method
                    and row["selection"] == "discrepancy"
                ]

                print(
                    f"{truth_kind:12s} "
                    f"{reference_kind:9s} "
                    f"{noise_level:8.0e} "
                    f"{method:15s} "
                    f"{np.median(values):14.6e}"
                )

print(f"\nResults written to {OUT}")
