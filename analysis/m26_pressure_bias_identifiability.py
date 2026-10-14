#!/usr/bin/env python3
"""M26-C2: Combined pressure bias and nuisance-parameter identifiability.

Study common-mode/differential-mode pressure calibration errors and
the local identifiability of two unknown constant pressure biases.

Uses the M26-B correlation model at tau=120 s.

All results are linearized. Bias parameters are scaled in units of 5 kPa.
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
    svd_influence,
    gls_influence,
    variance,
    N_TIME,
    N_CHANNELS,
    DT,
)

OUT = Path("benchmarks/results-m26")
OUT.mkdir(parents=True, exist_ok=True)

TAU = 120.0
BIAS_SCALE_PA = 5000.0

N_VARIABLES = 163
N_BIASES = 2

# Normalized measurement units are marginal sensor sigmas.
# Pressure marginal sigma = 5000 Pa.
#
# Therefore a 5000 Pa sensor bias corresponds to +1 in
# the normalized measurement coordinates.

PRESSURE_SIGMA_PA = 5000.0


def pressure_bias_matrix():
    """Return 244x2 measurement response to scaled pressure biases."""

    B = np.zeros((
        N_TIME * N_CHANNELS,
        N_BIASES,
    ))

    amplitude = BIAS_SCALE_PA / PRESSURE_SIGMA_PA

    # Channel 0: inlet pressure.
    B[0::N_CHANNELS, 0] = amplitude

    # Channel 2: outlet pressure.
    B[2::N_CHANNELS, 1] = amplitude

    return B


def pressure_drift_matrix(profile):
    """Return measurement response to pressure drift amplitudes."""

    B = pressure_bias_matrix()

    times = DT * np.arange(N_TIME)

    if profile == "constant":
        weights = np.ones(N_TIME)
    elif profile == "linear":
        weights = times / times[-1]
    elif profile == "late_onset":
        weights = np.maximum(
            0.0,
            (times - 300.0) / 300.0,
        )
    else:
        raise ValueError(profile)

    for channel, column in [(0, 0), (2, 1)]:
        indices = np.arange(
            channel,
            N_TIME * N_CHANNELS,
            N_CHANNELS,
        )

        B[indices, column] *= weights

    return B


def svd_rank(s, rtol):
    """Numerical rank relative to largest singular value."""

    return int(np.count_nonzero(
        s > rtol * s[0]
    ))


def main():
    A, g = load_problem()

    K = temporal_covariance(TAU)

    L = cholesky(
        K,
        lower=True,
        check_finite=True,
    )

    h_ols, sigma_ols_reported = svd_influence(A, g)

    h_gls, sigma_gls_svd = gls_influence(
        A,
        g,
        L,
    )

    sigma_gls = np.sqrt(variance(h_gls, K))

    assert np.isclose(
        sigma_gls,
        sigma_gls_svd,
        rtol=1e-6,
    )

    print()
    print("M26-C2: Pressure Bias and Identifiability")
    print("=" * 88)

    print(f"Correlation time: {TAU:.1f} s")
    print(f"GLS inventory sigma: {sigma_gls:.6f} kg")
    print(
        f"OLS reported inventory sigma: "
        f"{sigma_ols_reported:.6f} kg"
    )

    # ------------------------------------------------------------
    # Part A: Combined pressure errors.
    # ------------------------------------------------------------

    combinations = {
        "inlet_only": (1.0, 0.0),
        "outlet_only": (0.0, 1.0),
        "common_positive": (1.0, 1.0),
        "differential_positive": (1.0, -1.0),
        "common_negative": (-1.0, -1.0),
        "differential_negative": (-1.0, 1.0),
    }

    combined_records = []

    print()
    print("Combined pressure errors")
    print("-" * 88)

    for profile in [
        "constant",
        "linear",
        "late_onset",
    ]:
        B_profile = pressure_drift_matrix(profile)

        for name, amplitudes in combinations.items():
            beta = np.asarray(amplitudes)

            b = B_profile @ beta

            error_gls = float(h_gls @ b)
            error_ols = float(h_ols @ b)

            combined_records.append({
                "profile": profile,
                "combination": name,
                "inlet_bias_pa":
                    BIAS_SCALE_PA * beta[0],
                "outlet_bias_pa":
                    BIAS_SCALE_PA * beta[1],
                "gls_inventory_error_kg": error_gls,
                "ols_inventory_error_kg": error_ols,
                "gls_error_over_sigma":
                    error_gls / sigma_gls,
            })

            print(
                f"{profile:12s} "
                f"{name:24s} "
                f"GLS={error_gls:+12.5f} kg "
                f"OLS={error_ols:+12.5f} kg"
            )

    pd.DataFrame(combined_records).to_csv(
        OUT / "combined_pressure_bias.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Part B: Augmented Jacobian identifiability.
    # ------------------------------------------------------------

    B = pressure_bias_matrix()

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

    Aaug = np.column_stack([Aw, Bw])

    U, s, Vt = np.linalg.svd(
        Aw,
        full_matrices=False,
    )

    Uaug, saug, Vtaug = np.linalg.svd(
        Aaug,
        full_matrices=False,
    )

    # Orthogonal projection onto complement of existing
    # measurement-response space.
    #
    # Using all 163 singular vectors tests exact-arithmetic
    # local identifiability, subject to floating-point precision.

    Bperp = Bw - U @ (U.T @ Bw)

    _, s_perp, Vt_perp = np.linalg.svd(
        Bperp,
        full_matrices=False,
    )

    print()
    print("Bias-direction identifiability")
    print("-" * 88)

    for i in range(2):
        direction = Vt_perp[i]

        print(
            f"Mode {i}: "
            f"singular={s_perp[i]:.9e}, "
            f"inlet={direction[0]:+.6f}, "
            f"outlet={direction[1]:+.6f}"
        )

    original_rank = svd_rank(s, 1e-10)
    augmented_rank = svd_rank(saug, 1e-10)

    print()
    print("Augmented identifiability")
    print("-" * 88)

    print("Original Jacobian shape:", Aw.shape)
    print("Augmented Jacobian shape:", Aaug.shape)

    print(
        "Original singular-value range:",
        f"{s[-1]:.9e}",
        "to",
        f"{s[0]:.9e}",
    )

    print(
        "Augmented singular-value range:",
        f"{saug[-1]:.9e}",
        "to",
        f"{saug[0]:.9e}",
    )

    print(
        "Projected bias singular values:",
        s_perp,
    )

    print(
        "Rank at rtol=1e-10:",
        original_rank,
        "->",
        augmented_rank,
    )

    rank_records = []

    for rtol in [
        1e-12,
        1e-10,
        1e-8,
        1e-6,
        1e-4,
    ]:
        rank_records.append({
            "rtol": rtol,
            "original_rank": svd_rank(s, rtol),
            "augmented_rank": svd_rank(saug, rtol),
            "original_smallest_singular": s[-1],
            "augmented_smallest_singular": saug[-1],
            "projected_bias_singular_1": s_perp[0],
            "projected_bias_singular_2": s_perp[1],
        })

    pd.DataFrame(rank_records).to_csv(
        OUT / "pressure_bias_identifiability_rank.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Part C: Bias-augmented inventory uncertainty.
    # ------------------------------------------------------------

    # The inventory quantity depends on the physical pipeline
    # variables, not directly on the sensor calibration biases.
    #
    # Therefore the augmented inventory gradient is [g, 0, 0].

    gaug = np.concatenate([
        g,
        np.zeros(N_BIASES),
    ])

    # Formal unregularized GLS inventory uncertainty.
    #
    # This is meaningful only if the augmented system is
    # sufficiently identifiable and numerically stable.

    qaug = (Vtaug @ gaug) / saug

    sigma_aug = float(np.linalg.norm(qaug))

    print()
    print("Formal augmented inventory uncertainty")
    print("-" * 88)

    print(
        f"Without bias parameters: {sigma_gls:.6f} kg"
    )

    print(
        f"With two free bias parameters: {sigma_aug:.6f} kg"
    )

    print(
        "Uncertainty amplification:",
        sigma_aug / sigma_gls,
    )

    pd.DataFrame([{
        "tau_s": TAU,
        "bias_scale_pa": BIAS_SCALE_PA,
        "original_sigma_kg": sigma_gls,
        "augmented_sigma_kg": sigma_aug,
        "amplification": sigma_aug / sigma_gls,
        "original_condition_number":
            s[0] / s[-1],
        "augmented_condition_number":
            saug[0] / saug[-1],
        "projected_bias_singular_1":
            s_perp[0],
        "projected_bias_singular_2":
            s_perp[1],
    }]).to_csv(
        OUT / "pressure_bias_augmented_uncertainty.csv",
        index=False,
    )

    print()
    print("Saved:")
    print(" ", OUT / "combined_pressure_bias.csv")
    print(" ", OUT / "pressure_bias_identifiability_rank.csv")
    print(" ", OUT / "pressure_bias_augmented_uncertainty.csv")


if __name__ == "__main__":
    main()
