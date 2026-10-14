#!/usr/bin/env python3
"""M2.4: Joint and boundary-compensated observability analysis."""

from pathlib import Path
import numpy as np

SOURCE = Path(
    "benchmarks/results-m24/joint_measurement_jacobian.csv"
)
OUTPUT = Path("benchmarks/results-m24")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Must be verified against pack_joint_variables().
INITIAL_DIMENSION = 41
BOUNDARY_DIMENSION = 122

# Relative numerical rank threshold.
RTOL = 1.0e-10

J = np.loadtxt(SOURCE, delimiter=",", skiprows=1)

assert J.shape == (244, 163)
assert np.isfinite(J).all()

Jx = J[:, :INITIAL_DIMENSION]
Jb = J[:, INITIAL_DIMENSION:]

assert Jb.shape[1] == BOUNDARY_DIMENSION

def svd_report(name, A):
    U, s, Vt = np.linalg.svd(
        A, full_matrices=False
    )

    threshold = RTOL * s[0] if s.size else 0.0
    rank = int(np.count_nonzero(s > threshold))

    print()
    print(name)
    print("-" * 60)
    print("shape:", A.shape)
    print("rank:", rank)
    print("largest singular value:", s[0])
    print("smallest singular value:", s[-1])
    print("threshold:", threshold)

    np.savetxt(
        OUTPUT / f"{name}_singular_values.csv",
        np.column_stack([
            np.arange(len(s)),
            s,
            s / s[0]
        ]),
        delimiter=",",
        header="index,singular_value,relative_singular_value",
        comments=""
    )

    return U, s, Vt, rank

_, sj, _, rank_joint = svd_report("joint", J)

_, sx, Vtx, rank_initial = svd_report(
    "initial_fixed_boundary", Jx
)

Ub, sb, _, rank_boundary = svd_report(
    "boundary", Jb
)

# Remove the measurement-space directions that can be
# explained by adjustments to the boundary histories.
#
# Rank is threshold-dependent. We report the threshold
# explicitly rather than interpreting it as exact rank.

Qb = Ub[:, :rank_boundary]

Jeff = Jx - Qb @ (Qb.T @ Jx)

_, se, Vte, rank_effective = svd_report(
    "initial_compensated", Jeff
)

np.savetxt(
    OUTPUT / "initial_observability_modes.csv",
    Vtx.T,
    delimiter=","
)

np.savetxt(
    OUTPUT / "compensated_observability_modes.csv",
    Vte.T,
    delimiter=","
)

print()
print("=" * 60)
print("M2.4 OBSERVABILITY SUMMARY")
print("=" * 60)

print("Full joint rank:", rank_joint)
print("Initial rank, fixed boundaries:", rank_initial)
print("Boundary rank:", rank_boundary)
print("Initial rank, compensated:", rank_effective)

print()
print("Weakest fixed-boundary singular value:", sx[-1])
print("Weakest compensated singular value:", se[-1])

print()
print("Largest compensated singular value:", se[0])

print()
print("Outputs written to:", OUTPUT)

