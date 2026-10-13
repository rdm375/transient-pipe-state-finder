#!/usr/bin/env python3
"""Compare regularizers for the linearized boundary-only inverse problem."""

from pathlib import Path
import csv

import numpy as np
from scipy.linalg import lstsq

ROOT = Path("benchmarks/results-m16b-analytic")
OUT = ROOT / "regularization-comparison"
OUT.mkdir(parents=True, exist_ok=True)

N = 100
D = 2 * N + 1
HORIZON = 80
SEED = 20261013

H = np.loadtxt(ROOT / f"H_{HORIZON}.csv", delimiter=",")
assert H.shape == (162, 201)

rng = np.random.default_rng(SEED)

# Full right-singular basis, including the structural nullspace.
U, s, Vt = np.linalg.svd(H, full_matrices=True)

rank = int(np.sum(s > 1e-10 * s[0]))
Vweak = Vt[rank:, :].T

print(f"Jacobian: {H.shape}")
print(f"Effective rank: {rank}")
print(f"Weak-subspace dimension: {Vweak.shape[1]}")

# Dimensionless spatial curvature operator.
# Apply separately to pressure and internal mass flow.
D2 = np.zeros((2 * (N - 2), D))

for offset, row_offset in ((0, 0), (N, N - 2)):
    for i in range(N - 2):
        D2[row_offset + i, offset + i:offset + i + 3] = (
            1.0, -2.0, 1.0
        )

# Normalize to make lambda sweeps more interpretable.
D2 /= np.linalg.norm(D2, ord=2)

operators = {
    "unregularized": None,
    "tikhonov": np.eye(D),
    "curvature": D2,
    "weak_subspace": Vweak.T,
}

# All coordinates are dimensionless:
# pressure scale = 1e5 Pa; flow scale = 10 kg/s.
grid = (np.arange(N) + 1) / N
faces = (np.arange(N) + 0.5) / N

def make_truth(kind):
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

def solve(H, b, operator, lam):
    if operator is None or lam == 0:
        # Minimum-norm least-squares solution.
        return lstsq(H, b, cond=1e-14, lapack_driver="gelsd")[0]

    A = np.vstack((H, np.sqrt(lam) * operator))
    rhs = np.concatenate((b, np.zeros(operator.shape[0])))

    return lstsq(
        A, rhs, cond=1e-14, lapack_driver="gelsd"
    )[0]

def metrics(x, truth, b):
    e = x - truth

    return {
        "boundary_residual": float(np.linalg.norm(H @ x - b)),
        "pressure_rmse_pa": float(
            1e5 * np.sqrt(np.mean(e[:N]**2))
        ),
        "flow_rmse_kg_s": float(
            10 * np.sqrt(np.mean(e[N:2*N]**2))
        ),
        "inlet_flow_error_kg_s": float(10 * abs(e[-1])),
        "scaled_state_error": float(np.linalg.norm(e)),
    }

lambdas = np.logspace(-12, 2, 15)

# Noise is in normalized observation units.
# These are illustrative levels, not instrument specifications.
noise_levels = (0.0, 1e-4, 1e-2)

rows = []

for kind in ("smooth", "oscillatory", "localized"):
    truth = make_truth(kind)
    clean = H @ truth

    for noise_level in noise_levels:
        noise = rng.standard_normal(H.shape[0]) * noise_level
        b = clean + noise

        for method, operator in operators.items():
            strengths = (
                [0.0] if operator is None else lambdas
            )

            for lam in strengths:
                x = solve(H, b, operator, lam)
                result = metrics(x, truth, b)

                rows.append({
                    "truth": kind,
                    "noise": noise_level,
                    "method": method,
                    "lambda": lam,
                    **result,
                })

fields = list(rows[0])

with open(OUT / "results.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

# Diagnostic oracle selection: best state error using known
# synthetic truth. This is NOT a deployable lambda-selection rule.
print("\nBest synthetic-truth state recovery by method:")
print(
    "truth noise method lambda "
    "pressure_RMSE_Pa flow_RMSE_kg_s scaled_error"
)

for kind in ("smooth", "oscillatory", "localized"):
    for noise_level in noise_levels:
        for method in operators:
            candidates = [
                row for row in rows
                if row["truth"] == kind
                and row["noise"] == noise_level
                and row["method"] == method
            ]

            best = min(
                candidates,
                key=lambda row: row["scaled_state_error"]
            )

            print(
                f'{kind:12s} '
                f'{noise_level:7.0e} '
                f'{method:15s} '
                f'{best["lambda"]:9.1e} '
                f'{best["pressure_rmse_pa"]:12.4f} '
                f'{best["flow_rmse_kg_s"]:12.6f} '
                f'{best["scaled_state_error"]:12.6f}'
            )

print(f"\nWrote {OUT / 'results.csv'}")
