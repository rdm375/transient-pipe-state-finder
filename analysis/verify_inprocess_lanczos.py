#!/usr/bin/env python3
"""Verify M1.8c-3 against M1.8b matrix-free results."""

from pathlib import Path

import numpy as np

REFERENCE = Path("benchmarks/results-m18b")
CURRENT = Path("benchmarks/results-m18c")

for name in ("full", "late"):
    reference_values = np.loadtxt(
        REFERENCE / f"{name}_eigenpairs.csv",
        delimiter=",",
        skiprows=1
    )

    current_values = np.loadtxt(
        CURRENT / f"{name}_eigenpairs.csv",
        delimiter=",",
        skiprows=1
    )

    reference_vectors = np.loadtxt(
        REFERENCE / f"{name}_vectors.csv",
        delimiter=","
    )

    current_vectors = np.loadtxt(
        CURRENT / f"{name}_vectors.csv",
        delimiter=","
    )

    assert reference_vectors.shape == (201, 3)
    assert current_vectors.shape == (201, 3)

    print()
    print(f"Window: {name}")

    for i in range(3):
        ref = reference_values[i, 1]
        actual = current_values[i, 1]

        relative_error = (
            abs(actual-ref) /
            max(abs(ref), 1.0e-30)
        )

        alignment = abs(
            np.dot(
                reference_vectors[:, i],
                current_vectors[:, i]
            )
        )

        residual = current_values[i, 3]

        print(
            f"mode={i+1} "
            f"eigenvalue={actual:.12e} "
            f"relative_error={relative_error:.6e} "
            f"alignment={alignment:.12f} "
            f"ritz_residual={residual:.6e}"
        )

        # The dominant mode must agree tightly.
        if i == 0:
            assert relative_error < 1e-8
            assert alignment > 0.999999

        # Subdominant modes may be much smaller,
        # especially in the late observation window.
        else:
            assert relative_error < 1e-4
            assert alignment > 0.999

        assert residual < 1e-9

    print(f"PASS M1.8c-3 {name}")

print()
print("PASS M1.8c-3 in-process Lanczos verification")
