#!/usr/bin/env python3
"""M26-B1: Temporal correlation and terminal inventory uncertainty.

Compare:
  1. Correlation-aware generalized least squares (GLS).
  2. Actual uncertainty of correlation-blind ordinary least squares.
  3. Uncertainty incorrectly reported by correlation-blind OLS.

The covariance model has independent channels and AR(1)-equivalent
exponential temporal correlations within each channel.

All reported inventory uncertainties are linearized standard deviations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import (
    cholesky,
    solve_triangular,
)

OUT = Path("benchmarks/results-m26")
OUT.mkdir(parents=True, exist_ok=True)

N_TIME = 61
N_CHANNELS = 4
N_VARIABLES = 163
DT = 10.0

TAU_VALUES = [
    0.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
]

# Measurement normalizers, in physical units.
NORMALIZERS = np.array([
    1.0e4,  # inlet pressure, Pa
    1.0,    # inlet mass flow, kg/s
    1.0e4,  # outlet pressure, Pa
    1.0,    # outlet mass flow, kg/s
])

# Hypothetical marginal measurement standard deviations.
SIGMA_PHYSICAL = np.array([
    5.0e3,
    0.5,
    5.0e3,
    0.5,
])

SIGMA_NORMALIZED = SIGMA_PHYSICAL / NORMALIZERS


def load_problem():
    """Load the validated M24/M25 linearized inverse model."""

    J = pd.read_csv(
        "benchmarks/results-m24/joint_measurement_jacobian.csv"
    ).to_numpy(dtype=float)

    assert J.shape == (
        N_TIME * N_CHANNELS,
        N_VARIABLES,
    )

    assert np.all(np.isfinite(J))

    sensitivities = pd.read_csv(
        "benchmarks/results-m25/operational_sensitivity.csv"
    )

    rows = sensitivities[
        (sensitivities.time_s == 600.0)
        & (sensitivities.quantity == "inventory")
        & (sensitivities["index"] == 0)
    ].sort_values("column")

    assert np.array_equal(
        rows.column.to_numpy(dtype=int),
        np.arange(N_VARIABLES),
    )

    g = rows.sensitivity.to_numpy(dtype=float)

    assert np.all(np.isfinite(g))

    sigma_rows = np.tile(
        SIGMA_NORMALIZED,
        N_TIME,
    )

    A = J / sigma_rows[:, None]

    return A, g


def temporal_covariance(tau):
    """Build the full 244x244 correlation matrix.

    Measurement ordering:
      row = 4*time_index + channel_index.

    Channels are independent. Each channel has the same
    exponential temporal correlation in this experiment.
    """

    K = np.zeros((
        N_TIME * N_CHANNELS,
        N_TIME * N_CHANNELS,
    ))

    if tau == 0.0:
        np.fill_diagonal(K, 1.0)
        return K

    times = DT * np.arange(N_TIME)

    distance = np.abs(
        times[:, None] - times[None, :]
    )

    temporal = np.exp(-distance / tau)

    for channel in range(N_CHANNELS):
        indices = np.arange(
            channel,
            N_TIME * N_CHANNELS,
            N_CHANNELS,
        )

        K[np.ix_(indices, indices)] = temporal

    return K


def svd_inventory_sigma(A, g):
    """Return inventory sigma from full-column-rank SVD."""

    U, s, Vt = np.linalg.svd(
        A,
        full_matrices=False,
    )

    if not np.all(s > 0):
        raise RuntimeError("Rank-deficient Jacobian")

    q = (Vt @ g) / s

    sigma = float(np.linalg.norm(q))

    return sigma, U, s, Vt


def main():
    A, g = load_problem()

    # OLS estimator computed once.
    sigma_reported, U, s, Vt = (
        svd_inventory_sigma(A, g)
    )

    # Measurement-space influence vector:
    #
    #   h = A (A^T A)^(-1) g
    #     = U diag(1/s) V^T g.
    #
    # Inventory error = h^T eta.

    h = U @ ((Vt @ g) / s)

    assert np.isclose(
        np.linalg.norm(h),
        sigma_reported,
        rtol=1e-10,
    )

    records = []

    print()
    print("M26-B1: Temporal Correlation")
    print("=" * 88)
    print(
        "Independent-noise OLS reported sigma:",
        f"{sigma_reported:.9f} kg",
    )
    print()

    for tau in TAU_VALUES:
        K = temporal_covariance(tau)

        L = cholesky(
            K,
            lower=True,
            check_finite=True,
        )

        # Correctly whitened GLS design matrix.
        AK = solve_triangular(
            L,
            A,
            lower=True,
            check_finite=True,
        )

        sigma_gls, _, sg, _ = (
            svd_inventory_sigma(AK, g)
        )

        # Actual variance of the OLS estimator.
        sigma_ols_true = float(
            np.sqrt(h @ K @ h)
        )

        # Under-reporting factor for correlation-blind OLS.
        reporting_factor = (
            sigma_ols_true / sigma_reported
        )

        # Efficiency of GLS relative to OLS.
        efficiency_ratio = (
            sigma_ols_true / sigma_gls
        )

        # Gauss-Markov consistency.
        if sigma_gls > sigma_ols_true * (
            1.0 + 1e-5
        ):
            raise RuntimeError(
                f"GLS variance exceeds OLS variance at tau={tau}"
            )

        records.append({
            "tau_s": tau,
            "rho_lag1": (
                0.0 if tau == 0.0
                else np.exp(-DT / tau)
            ),
            "gls_sigma_kg": sigma_gls,
            "ols_actual_sigma_kg": sigma_ols_true,
            "ols_reported_sigma_kg": sigma_reported,
            "ols_underreporting_factor":
                reporting_factor,
            "ols_to_gls_sigma_ratio":
                efficiency_ratio,
            "gls_condition_number":
                float(sg[0] / sg[-1]),
        })

        print(
            f"tau={tau:6.1f} s | "
            f"GLS={sigma_gls:12.6f} kg | "
            f"OLS actual={sigma_ols_true:12.6f} kg | "
            f"OLS reported={sigma_reported:12.6f} kg | "
            f"reporting factor={reporting_factor:8.4f} | "
            f"OLS/GLS={efficiency_ratio:8.4f}"
        )

    results = pd.DataFrame(records)

    results.to_csv(
        OUT / "temporal_correlation_sweep.csv",
        index=False,
    )

    print()
    print(
        "Saved:",
        OUT / "temporal_correlation_sweep.csv",
    )


if __name__ == "__main__":
    main()
