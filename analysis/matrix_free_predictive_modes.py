#!/usr/bin/env python3
"""M1.8b: Matrix-free predictive-mode identification."""

from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

N = 201
STEPS = 800
DT = 15.0

EXECUTABLE = Path(
    "./build/state_finder_matrix_free_normal_operator"
)

OUTPUT = Path("benchmarks/results-m18b")
OUTPUT.mkdir(parents=True, exist_ok=True)

H_PATH = Path("benchmarks/results-m17a/H_800.csv")

rng = np.random.default_rng(20261013)


class MatrixFreeNormalOperator:
    def __init__(self, start_step):
        self.start_step = start_step
        self.calls = 0
        self.elapsed = 0.0

        self.workspace = tempfile.TemporaryDirectory()

        root = Path(self.workspace.name)

        self.input_path = root / "input.txt"
        self.output_path = root / "output.txt"

        self.operator = LinearOperator(
            shape=(N, N),
            dtype=np.float64,
            matvec=self.matvec
        )

    def matvec(self, v):
        v = np.asarray(v, dtype=np.float64)

        assert v.shape == (N,)
        assert np.all(np.isfinite(v))

        np.savetxt(
            self.input_path,
            v,
            fmt="%.17e"
        )

        start = time.perf_counter()

        subprocess.run(
            [
                str(EXECUTABLE),
                str(self.start_step),
                str(self.input_path),
                str(self.output_path)
            ],
            check=True,
            capture_output=True,
            text=True
        )

        result = np.loadtxt(self.output_path)

        self.elapsed += time.perf_counter() - start
        self.calls += 1

        assert result.shape == (N,)
        assert np.all(np.isfinite(result))

        return result

    def close(self):
        self.workspace.cleanup()


def explicit_reference(start_step):
    H = np.loadtxt(H_PATH, delimiter=",")

    assert H.shape == (1602, N)

    # Original H normalization:
    # flow / 1 kg/s
    # pressure / 10000 Pa
    #
    # Target normalization:
    # flow / 0.01 kg/s
    # pressure / 10 Pa
    conversion = np.tile(
        [100.0, 1000.0],
        STEPS + 1
    )

    # Prior sigma / state-finder scale = 0.1.
    K = (
        H[2*start_step:, :]
        * conversion[2*start_step:, None]
        * 0.1
    )

    # Symmetric eigendecomposition of the
    # explicit reference normal matrix.
    G = K.T @ K

    values, vectors = np.linalg.eigh(G)

    order = np.argsort(values)[::-1]

    return (
        values[order],
        vectors[:, order]
    )


def run_window(name, start_step):
    print()
    print("=" * 64)
    print(
        f"Window: {name}, "
        f"start={start_step*DT/60:.2f} min"
    )
    print("=" * 64)

    engine = MatrixFreeNormalOperator(start_step)

    try:
        initial = rng.standard_normal(N)
        initial /= np.linalg.norm(initial)

        start = time.perf_counter()

        values, vectors = eigsh(
            engine.operator,
            k=3,
            which="LA",
            tol=1e-9,
            maxiter=1000,
            ncv=24,
            v0=initial
        )

        total_elapsed = time.perf_counter() - start

        order = np.argsort(values)[::-1]

        values = values[order]
        vectors = vectors[:, order]

        oracle_values, oracle_vectors = (
            explicit_reference(start_step)
        )

        rows = []

        for i in range(3):
            value = values[i]
            vector = vectors[:, i]

            # Independent explicit residual:
            # the oracle is used for validation only.
            H = np.loadtxt(
                H_PATH,
                delimiter=","
            )

            conversion = np.tile(
                [100.0, 1000.0],
                STEPS + 1
            )

            K = (
                H[2*start_step:, :]
                * conversion[2*start_step:, None]
                * 0.1
            )

            Gv = K.T @ (K @ vector)

            residual = (
                np.linalg.norm(Gv - value*vector)
                / max(abs(value), 1.0)
            )

            alignment = abs(
                np.dot(
                    vector,
                    oracle_vectors[:, i]
                )
            )

            relative_eigenvalue_error = (
                abs(value-oracle_values[i])
                / max(abs(oracle_values[i]), 1.0)
            )

            print(
                f"mode={i+1} "
                f"eigenvalue={value:.12e} "
                f"singular_value={np.sqrt(max(value, 0)):.12e}"
            )

            print(
                f"  alignment={alignment:.12f} "
                f"relative_eigenvalue_error="
                f"{relative_eigenvalue_error:.6e} "
                f"residual={residual:.6e}"
            )

            rows.append([
                i+1,
                value,
                np.sqrt(max(value, 0.0)),
                oracle_values[i],
                alignment,
                relative_eigenvalue_error,
                residual
            ])

            assert relative_eigenvalue_error < 1e-6
            assert residual < 1e-6

            # Eigenvector alignment is meaningful
            # only for sufficiently isolated modes.
            gap = (
                abs(oracle_values[i] - oracle_values[i+1])
                / max(abs(oracle_values[i]), 1.0)
            )

            if gap > 1e-4:
                assert alignment > 0.999

        np.savetxt(
            OUTPUT / f"{name}_eigenpairs.csv",
            np.asarray(rows),
            delimiter=",",
            header=(
                "mode,eigenvalue,singular_value,"
                "oracle_eigenvalue,alignment,"
                "relative_eigenvalue_error,residual"
            ),
            comments=""
        )

        np.savetxt(
            OUTPUT / f"{name}_vectors.csv",
            vectors,
            delimiter=",",
            fmt="%.17e"
        )

        print()
        print(f"operator_calls={engine.calls}")
        print(f"operator_seconds={engine.elapsed:.6f}")
        print(f"total_seconds={total_elapsed:.6f}")

        print(f"PASS M1.8b {name}")

        return vectors[:, 0].copy()

    finally:
        engine.close()


def main():
    full = run_window(
        "full",
        0
    )

    late = run_window(
        "late",
        80
    )

    cross_alignment = abs(
        np.dot(full, late)
    )

    print()
    print("=" * 64)
    print("Full-window versus late-window")
    print("=" * 64)

    print(
        "dominant_mode_alignment="
        f"{cross_alignment:.12f}"
    )

    # Compare the late-window mode with the
    # previously identified predictive direction.
    import csv

    with open(
        "benchmarks/results-m17a/"
        "dominant_predictive_mode.csv",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    previous = np.array([
        float(row["standardized_mode"])
        for row in rows
    ])

    previous /= np.linalg.norm(previous)

    alignment = abs(
        np.dot(late, previous)
    )

    print(
        "late_vs_m17c_alignment="
        f"{alignment:.12f}"
    )

    print()
    print("PASS M1.8b matrix-free predictive modes")


if __name__ == "__main__":
    main()
