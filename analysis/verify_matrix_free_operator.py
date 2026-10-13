#!/usr/bin/env python3

from pathlib import Path
import subprocess
import tempfile

import numpy as np

H = np.loadtxt(
    "benchmarks/results-m17a/H_800.csv",
    delimiter=","
)

assert H.shape == (1602, 201)

noise_conversion = np.tile(
    [100.0, 1000.0],
    801
)

K = H * noise_conversion[:, None] * 0.1

executable = (
    "./build/state_finder_matrix_free_normal_operator"
)

rng = np.random.default_rng(20261013)

def apply_matrix_free(v, start_step):
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.txt"
        output_path = Path(tmp) / "output.txt"

        np.savetxt(
            input_path,
            v,
            fmt="%.17e"
        )

        subprocess.run(
            [
                executable,
                str(start_step),
                str(input_path),
                str(output_path)
            ],
            check=True
        )

        result = np.loadtxt(output_path)

    assert result.shape == (201,)
    assert np.all(np.isfinite(result))

    return result


for start_step in [0, 80, 160, 400]:
    v = rng.standard_normal(201)
    v /= np.linalg.norm(v)

    K_window = K[2*start_step:, :]

    expected = K_window.T @ (K_window @ v)

    actual = apply_matrix_free(v, start_step)

    absolute_error = np.linalg.norm(
        actual - expected
    )

    relative_error = (
        absolute_error /
        max(np.linalg.norm(expected), 1.0)
    )

    print(
        f"start_step={start_step:3d} "
        f"absolute_error={absolute_error:.9e} "
        f"relative_error={relative_error:.9e}"
    )

    assert relative_error < 1e-8

print("PASS M1.8b matrix-free normal operator")
