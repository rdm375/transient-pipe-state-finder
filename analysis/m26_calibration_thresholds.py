#!/usr/bin/env python3
"""M26-C4: Calibration accuracy thresholds for terminal linepack.

Find the marginal pressure-calibration standard deviation required
to produce specified increases in terminal inventory uncertainty.

Uses the M26-C3 Gaussian calibration-prior formulation.

All results are linearized and scenario-dependent.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import brentq

from m26_correlated_noise_monte_carlo import (
    load_problem,
    temporal_covariance,
    N_TIME,
    N_CHANNELS,
)

from m26_calibration_prior_sensitivity import (
    TAU,
    CORRELATION_MODELS,
    physical_bias_matrix,
    calibration_covariance,
    inventory_sigma_from_svd,
    augmented_prior_sigma,
)

OUT = Path("benchmarks/results-m26")
OUT.mkdir(parents=True, exist_ok=True)

TARGET_AMPLIFICATIONS = (1.05, 1.10, 1.25)

ROOT_XTOL_KPA = 1e-6
TARGET_RTOL = 1e-5


def main():
    A, g = load_problem()

    K = temporal_covariance(TAU)
    L = cholesky(K, lower=True, check_finite=True)

    Aw = solve_triangular(L, A, lower=True)

    B = physical_bias_matrix()
    Bw = solve_triangular(L, B, lower=True)

    sigma0, _ = inventory_sigma_from_svd(Aw, g)

    print()
    print("M26-C4: Calibration Accuracy Thresholds")
    print("=" * 94)
    print(f"Temporal correlation: {TAU:.1f} s")
    print(f"Known-calibration inventory sigma: {sigma0:.9f} kg")
    print()

    records = []

    for model_name, rho in CORRELATION_MODELS.items():

        def amplification(sigma_kpa):
            if sigma_kpa == 0.0:
                return 1.0

            Cbeta = calibration_covariance(
                sigma_kpa * 1000.0,
                rho,
            )

            sigma_M, _ = augmented_prior_sigma(
                Aw,
                Bw,
                g,
                Cbeta,
            )

            return sigma_M / sigma0

        print(f"Model: {model_name} (rho={rho:+.2f})")
        print("-" * 94)

        for target in TARGET_AMPLIFICATIONS:

            def residual(sigma_kpa):
                return amplification(sigma_kpa) - target

            lower = 0.0
            upper = 10.0

            # Expand the bracket rather than assuming that
            # every target is reached below 10 kPa.
            while residual(upper) < 0.0:
                upper *= 2.0

                if upper > 1e6:
                    raise RuntimeError(
                        f"Could not bracket target {target} "
                        f"for {model_name}"
                    )

            root = brentq(
                residual,
                lower,
                upper,
                xtol=ROOT_XTOL_KPA,
                rtol=1e-12,
            )

            achieved = amplification(root)
            sigma_M = achieved * sigma0

            if not np.isclose(
                achieved,
                target,
                rtol=TARGET_RTOL,
                atol=1e-7,
            ):
                raise RuntimeError(
                    "Threshold verification failed: "
                    f"{model_name}, target={target}, "
                    f"achieved={achieved}"
                )

            records.append({
                "tau_s": TAU,
                "calibration_model": model_name,
                "rho": rho,
                "target_amplification": target,
                "target_increase_pct":
                    100.0 * (target - 1.0),
                "calibration_sigma_kpa": root,
                "calibration_sigma_pa": root * 1000.0,
                "achieved_amplification": achieved,
                "inventory_sigma_kg": sigma_M,
                "baseline_sigma_kg": sigma0,
                "absolute_target_error":
                    abs(achieved - target),
            })

            print(
                f"Target +{100*(target-1):5.1f}% | "
                f"sigma_cal={root:11.6f} kPa | "
                f"sigma_M={sigma_M:12.6f} kg | "
                f"verified={achieved:.9f}"
            )

        print()

    results = pd.DataFrame(records)

    output = OUT / "calibration_accuracy_thresholds.csv"
    results.to_csv(output, index=False)

    print("Saved:", output)

    # Confirm that the thresholds increase with the
    # permitted inventory-uncertainty amplification.
    for _, group in results.groupby("calibration_model"):
        ordered = group.sort_values("target_amplification")
        values = ordered["calibration_sigma_kpa"].to_numpy()

        if not np.all(np.diff(values) > 0):
            raise RuntimeError(
                "Calibration thresholds are not monotone"
            )

    print("Threshold verification: PASS")
    print("Monotonicity verification: PASS")


if __name__ == "__main__":
    main()
