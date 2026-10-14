#!/usr/bin/env python3
"""M27-A1: Nonlinear measurement and correlated-noise validation."""

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from scipy.linalg import cholesky, solve_triangular

from m26_correlated_noise_monte_carlo import (
    load_problem,
    temporal_covariance,
)

from m26_calibration_prior_sensitivity import TAU


ROOT = Path("benchmarks/results-m27")
ROOT.mkdir(parents=True, exist_ok=True)

REPLAY = Path("build/state_finder_m25_joint_nonlinear_replay")

N_VARIABLES = 163
N_MEASUREMENTS = 244

SEED = 27001

FD_STEPS = (1e-2, 1e-3, 1e-4)


class Replay:
    def __init__(self):
        self.process = subprocess.Popen(
            [str(REPLAY)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ready = self.process.stdout.readline().strip()

        if ready != f"READY {N_VARIABLES}":
            raise RuntimeError(
                f"Unexpected replay handshake: {ready!r}"
            )

    def request(self, prefix, direction):
        direction = np.asarray(direction, dtype=float)

        if direction.shape != (N_VARIABLES,):
            raise ValueError("Incorrect direction dimension")

        if not np.all(np.isfinite(direction)):
            raise ValueError("Nonfinite direction")

        payload = ",".join(f"{x:.17g}" for x in direction)

        self.process.stdin.write(prefix + payload + "\n")
        self.process.stdin.flush()

        response = self.process.stdout.readline().strip()

        if response.startswith("ERROR,"):
            raise RuntimeError(response)

        fields = response.split(",")

        expected = (
            N_MEASUREMENTS
            if prefix == "MEASURE,"
            else 7
        )

        expected_tag = (
            "MEASURE"
            if prefix == "MEASURE,"
            else "OK"
        )

        if fields[0] != expected_tag:
            raise RuntimeError(
                f"Unexpected response: {response[:200]}"
            )

        values = np.array(
            [float(x) for x in fields[1:]],
            dtype=float,
        )

        if values.size != expected:
            raise RuntimeError(
                f"Expected {expected} values, "
                f"received {values.size}"
            )

        if not np.all(np.isfinite(values)):
            raise RuntimeError("Nonfinite replay result")

        return values

    def measure(self, direction):
        return self.request("MEASURE,", direction)

    def diagnostics(self, direction):
        return self.request("", direction)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write("QUIT\n")
            self.process.stdin.flush()

        self.process.communicate(timeout=10)


def main():
    if not REPLAY.exists():
        raise FileNotFoundError(
            f"Replay executable missing: {REPLAY}"
        )

    A, g = load_problem()

    if A.shape != (N_MEASUREMENTS, N_VARIABLES):
        raise RuntimeError(
            f"Unexpected M24 Jacobian shape: {A.shape}"
        )

    # M26 load_problem returns the noise-whitened scaling
    # A = 2J for the independent-noise convention.
    #
    # Recover the original normalized measurement Jacobian.
    J = 0.5 * A

    K = temporal_covariance(TAU)

    if K.shape != (N_MEASUREMENTS, N_MEASUREMENTS):
        raise RuntimeError(
            f"Unexpected covariance shape: {K.shape}"
        )

    L = cholesky(K, lower=True)

    rng = np.random.default_rng(SEED)

    z = rng.standard_normal(N_MEASUREMENTS)

    # M26 normalized channel sigma = 0.5.
    #
    # The covariance returned by temporal_covariance()
    # is assumed to contain that physical normalization.
    noise = L @ z

    whitened_noise = solve_triangular(
        L,
        noise,
        lower=True,
    )

    print()
    print("M27-A1: Nonlinear Measurement Validation")
    print("=" * 78)
    print(f"Random seed: {SEED}")
    print(f"Variables: {N_VARIABLES}")
    print(f"Measurements: {N_MEASUREMENTS}")
    print(f"Correlation time: {TAU:.1f} s")
    print(f"Noise norm: {np.linalg.norm(noise):.9f}")
    print(
        "Whitened noise norm:",
        f"{np.linalg.norm(whitened_noise):.9f}",
    )

    replay = Replay()

    try:
        zero = np.zeros(N_VARIABLES)

        baseline = replay.measure(zero)

        baseline_norm = np.linalg.norm(baseline)

        print()
        print("Reference replay")
        print("-" * 78)
        print(f"Measurement norm: {baseline_norm:.12e}")

        if baseline_norm > 1e-10:
            raise RuntimeError(
                "Nonlinear reference replay is not zero"
            )

        diagnostics = replay.diagnostics(zero)

        print(
            "Reference inventory delta:",
            f"{diagnostics[2]:.12e} kg",
        )
        print(
            "Reference maximum mass defect:",
            f"{diagnostics[5]:.12e} kg",
        )
        print(
            "Reference boundary compatibility:",
            f"{diagnostics[6]:.12e} Pa",
        )

        if abs(diagnostics[2]) > 1e-8:
            raise RuntimeError(
                "Reference inventory delta is nonzero"
            )

        if diagnostics[5] > 1e-6:
            raise RuntimeError(
                "Reference mass defect exceeds tolerance"
            )

        if diagnostics[6] > 1e-8:
            raise RuntimeError(
                "Reference boundary compatibility failed"
            )

        direction = rng.standard_normal(N_VARIABLES)
        direction /= np.linalg.norm(direction)

        linear = J @ direction

        records = []

        print()
        print("Directional derivative consistency")
        print("-" * 78)

        for h in FD_STEPS:
            plus = replay.measure(h * direction)
            minus = replay.measure(-h * direction)

            finite_difference = (plus - minus) / (2 * h)

            error = np.linalg.norm(
                finite_difference - linear
            )

            relative_error = error / max(
                np.linalg.norm(linear),
                1e-15,
            )

            records.append({
                "step": h,
                "absolute_error": error,
                "relative_error": relative_error,
                "linear_norm": np.linalg.norm(linear),
                "finite_difference_norm":
                    np.linalg.norm(finite_difference),
            })

            print(
                f"h={h:.1e}  "
                f"absolute={error:.9e}  "
                f"relative={relative_error:.9e}"
            )

        best_error = min(
            row["relative_error"]
            for row in records
        )

        if best_error > 1e-4:
            raise RuntimeError(
                "M24 Jacobian and nonlinear replay disagree"
            )

        # Save the exact normalized observation perturbation.
        pd.DataFrame({
            "measurement_index": np.arange(N_MEASUREMENTS),
            "noise_normalized": noise,
            "whitened_noise": whitened_noise,
        }).to_csv(
            ROOT / "correlated_measurement_realization.csv",
            index=False,
        )

        pd.DataFrame(records).to_csv(
            ROOT / "nonlinear_directional_validation.csv",
            index=False,
        )

        print()
        print("Saved:")
        print(
            "  benchmarks/results-m27/"
            "correlated_measurement_realization.csv"
        )
        print(
            "  benchmarks/results-m27/"
            "nonlinear_directional_validation.csv"
        )

        print()
        print("Reference replay: PASS")
        print("Directional derivative: PASS")
        print("Correlated-noise generation: PASS")

    finally:
        replay.close()


if __name__ == "__main__":
    main()
