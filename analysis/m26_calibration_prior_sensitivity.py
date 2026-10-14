#!/usr/bin/env python3
"""M26-C3: Calibration prior sensitivity and terminal linepack uncertainty.

Quantifies the effect of uncertain inlet/outlet pressure calibration
offsets on terminal inventory uncertainty.

Model:
    y = A x + B beta + epsilon

    epsilon ~ N(0, K)
    beta    ~ N(0, C_beta)

The physical unknowns x have no additional prior.

Calibration uncertainty is represented by a Gaussian prior, not
an arbitrary numerical regularization penalty.

All results are linearized and conditional on the M26 measurement model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import (
    cholesky,
    solve_triangular,
)

from m26_correlated_noise_monte_carlo import (
    load_problem,
    temporal_covariance,
    N_TIME,
    N_CHANNELS,
)

OUT = Path("benchmarks/results-m26")
OUT.mkdir(parents=True, exist_ok=True)

TAU = 120.0

# Pressure measurement marginal standard deviation.
PRESSURE_SIGMA_PA = 5000.0

CALIBRATION_SIGMA_KPA = [
    0.0,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]

CORRELATION_MODELS = {
    "independent": 0.0,
    "positive_0p5": 0.5,
    "common_0p9": 0.9,
    "differential_m0p9": -0.9,
}


def physical_bias_matrix():
    """Measurement response to one pascal of pressure sensor offset.

    B has units of normalized measurement / Pa.
    """

    B = np.zeros((
        N_TIME * N_CHANNELS,
        2,
    ))

    B[0::N_CHANNELS, 0] = 1.0 / PRESSURE_SIGMA_PA
    B[2::N_CHANNELS, 1] = 1.0 / PRESSURE_SIGMA_PA

    return B


def calibration_covariance(sigma_pa, rho):
    """Two-sensor calibration covariance in Pa^2."""

    return sigma_pa**2 * np.array([
        [1.0, rho],
        [rho, 1.0],
    ])


def inventory_sigma_from_svd(M, gradient):
    """Compute formal inventory sigma using an SVD factorization.

    Assumes full column rank in the mathematical model.

    Does not truncate weak singular directions.
    """

    U, s, Vt = np.linalg.svd(
        M,
        full_matrices=False,
    )

    if np.any(s <= 0):
        raise RuntimeError(
            "Nonpositive singular value encountered"
        )

    q = (Vt @ gradient) / s

    return float(np.linalg.norm(q)), s


def augmented_prior_sigma(Aw, Bw, g, Cbeta):
    """Inventory uncertainty using a Gaussian calibration prior.

    Parameterization:
        beta = Lbeta z, z ~ N(0, I).

    Augmented whitened least-squares system:

        [Aw  Bw Lbeta] [x] = whitened measurements
        [ 0       I ] [z] = prior pseudo-observation

    The inventory gradient is [g, 0, 0].
    """

    Lbeta = cholesky(
        Cbeta,
        lower=True,
        check_finite=True,
    )

    n = Aw.shape[1]

    upper = np.column_stack([
        Aw,
        Bw @ Lbeta,
    ])

    lower = np.column_stack([
        np.zeros((2, n)),
        np.eye(2),
    ])

    M = np.vstack([
        upper,
        lower,
    ])

    gradient = np.concatenate([
        g,
        np.zeros(2),
    ])

    return inventory_sigma_from_svd(
        M,
        gradient,
    )


def main():
    A, g = load_problem()

    K = temporal_covariance(TAU)

    L = cholesky(
        K,
        lower=True,
        check_finite=True,
    )

    B = physical_bias_matrix()

    Aw = solve_triangular(
        L,
        A,
        lower=True,
    )

    Bw = solve_triangular(
        L,
        B,
        lower=True,
    )

    sigma_baseline, s_baseline = inventory_sigma_from_svd(
        Aw,
        g,
    )

    print()
    print("M26-C3: Calibration Prior Sensitivity")
    print("=" * 104)

    print(f"Temporal correlation: {TAU:.1f} s")
    print(f"Known-calibration sigma: {sigma_baseline:.6f} kg")
    print(
        "Known-calibration condition number:",
        f"{s_baseline[0] / s_baseline[-1]:.6e}",
    )

    records = []

    for model_name, rho in CORRELATION_MODELS.items():

        print()
        print(f"Calibration model: {model_name}, rho={rho:+.2f}")
        print("-" * 104)

        for sigma_kpa in CALIBRATION_SIGMA_KPA:

            sigma_pa = sigma_kpa * 1000.0

            if sigma_pa == 0:
                sigma_inventory = sigma_baseline
                s = s_baseline
            else:
                Cbeta = calibration_covariance(
                    sigma_pa,
                    rho,
                )

                sigma_inventory, s = augmented_prior_sigma(
                    Aw,
                    Bw,
                    g,
                    Cbeta,
                )

            amplification = (
                sigma_inventory / sigma_baseline
            )

            extra_variance = max(
                0.0,
                sigma_inventory**2 - sigma_baseline**2,
            )

            calibration_variance_fraction = (
                extra_variance / sigma_inventory**2
            )

            records.append({
                "tau_s": TAU,
                "calibration_model": model_name,
                "rho": rho,
                "calibration_sigma_kpa": sigma_kpa,
                "inventory_sigma_kg": sigma_inventory,
                "baseline_sigma_kg": sigma_baseline,
                "amplification": amplification,
                "extra_variance_kg2": extra_variance,
                "extra_variance_fraction":
                    calibration_variance_fraction,
                "smallest_singular_value": s[-1],
                "condition_number": s[0] / s[-1],
            })

            print(
                f"sigma_cal={sigma_kpa:5.1f} kPa | "
                f"sigma_M={sigma_inventory:12.4f} kg | "
                f"amplification={amplification:9.4f} | "
                f"extra variance fraction="
                f"{calibration_variance_fraction:8.4%}"
            )

    results = pd.DataFrame(records)

    output = OUT / "calibration_prior_sensitivity.csv"

    results.to_csv(
        output,
        index=False,
    )

    print()
    print("Saved:", output)

    # ----------------------------------------------------------
    # Independent cross-check:
    #
    # For one calibration model, verify that marginalizing
    # beta gives the same result as the augmented prior system.
    # ----------------------------------------------------------

    sigma_pa = 2000.0
    rho = 0.5

    Cbeta = calibration_covariance(
        sigma_pa,
        rho,
    )

    Keff = K + B @ Cbeta @ B.T

    Leff = cholesky(
        Keff,
        lower=True,
        check_finite=True,
    )

    Aeff = solve_triangular(
        Leff,
        A,
        lower=True,
    )

    sigma_marginal, _ = inventory_sigma_from_svd(
        Aeff,
        g,
    )

    sigma_augmented, _ = augmented_prior_sigma(
        Aw,
        Bw,
        g,
        Cbeta,
    )

    relative_difference = abs(
        sigma_marginal - sigma_augmented
    ) / sigma_marginal

    print()
    print("Marginal covariance cross-check")
    print("-" * 104)

    print(
        f"Marginalized sigma: {sigma_marginal:.9f} kg"
    )

    print(
        f"Augmented sigma:    {sigma_augmented:.9f} kg"
    )

    print(
        f"Relative difference: {relative_difference:.3e}"
    )

    if relative_difference > 1e-5:
        raise RuntimeError(
            "Marginal covariance and augmented prior "
            "calculations disagree"
        )

    print("Cross-check: PASS")


if __name__ == "__main__":
    main()
