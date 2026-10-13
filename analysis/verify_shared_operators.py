#!/usr/bin/env python3
"""Independent verification of M1.8c shared operators."""

from pathlib import Path
import subprocess
import tempfile

import numpy as np

N = 201
STEPS = 800

H = np.loadtxt(
    "benchmarks/results-m17a/H_800.csv",
    delimiter=","
)

assert H.shape == (2*(STEPS+1), N)

conversion = np.tile(
    [100.0, 1000.0],
    STEPS+1
)

K = H * conversion[:, None] * 0.1

rng = np.random.Generator(
    np.random.MT19937(20261013)
)

# The C++ test uses mt19937_64 and std::normal_distribution.
# Python does not reproduce that exact sequence, so use
# a separate explicit-oracle test with forcing exported
# from the C++ executable.
#
# For now, the C++ executable verifies the dot identity
# and the shared-tangent equivalence. The explicit transpose
# comparison requires exporting the actual C++ forcing.

executable = (
    "./build/state_finder_shared_operators_test"
)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    for start_step in (0, 80, 160, 400):
        output = root / f"ktw_{start_step}.txt"

        result = subprocess.run(
            [
                executable,
                str(start_step),
                str(output)
            ],
            capture_output=True,
            text=True
        )

        print(result.stdout, end="")

        if result.returncode:
            print(result.stderr)
            raise RuntimeError(
                f"Window {start_step} failed"
            )

        ktw = np.loadtxt(output)

        assert ktw.shape == (N,)
        assert np.all(np.isfinite(ktw))

        forcing = np.loadtxt(
            str(output) + ".forcing"
        )

        assert forcing.shape == (2*(STEPS+1),)

        reference = (
            K[2*start_step:, :].T
            @ forcing[2*start_step:]
        )

        absolute_error = np.linalg.norm(
            ktw-reference
        )

        relative_error = (
            absolute_error /
            max(np.linalg.norm(reference), 1.0)
        )

        print(
            f"explicit_transpose_start={start_step} "
            f"absolute_error={absolute_error:.9e} "
            f"relative_error={relative_error:.9e}"
        )

        assert relative_error < 1e-9

print("PASS M1.8c shared operators and transpose identities")
