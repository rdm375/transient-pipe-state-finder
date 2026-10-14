#!/usr/bin/env python3
"""M26-B2: Correlated-noise Monte Carlo and inventory influence analysis.

Validates the analytical GLS and OLS inventory variances from M26-B1
and examines the measurement-space inventory influence vectors.

The experiment is linearized. It does not solve the nonlinear inverse
problem or establish nonlinear uncertainty coverage.
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

N_SAMPLES = 10000
SEED = 2602

TAU_VALUES = [
    0.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
]

CHANNEL_NAMES = [
    "inlet_pressure",
    "inlet_flow",
    "outlet_pressure",
    "outlet_flow",
]

NORMALIZERS = np.array([
    1.0e4,
    1.0,
    1.0e4,
    1.0,
])

SIGMA_PHYSICAL = np.array([
    5.0e3,
    0.5,
    5.0e3,
    0.5,
])

SIGMA_NORMALIZED = SIGMA_PHYSICAL / NORMALIZERS


def load_problem():
    J = pd.read_csv(
        "benchmarks/results-m24/joint_measurement_jacobian.csv"
    ).to_numpy(dtype=float)

    assert J.shape == (
        N_TIME * N_CHANNELS,
        N_VARIABLES,
    )

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

    sigma_rows = np.tile(
        SIGMA_NORMALIZED,
        N_TIME,
    )

    A = J / sigma_rows[:, None]

    assert np.all(np.isfinite(A))
    assert np.all(np.isfinite(g))

    return A, g


def temporal_covariance(tau):
    """Return the channel-independent temporal correlation matrix."""

    if tau == 0.0:
        return np.eye(N_TIME * N_CHANNELS)

    times = DT * np.arange(N_TIME)

    temporal = np.exp(
        -np.abs(times[:, None] - times[None, :]) / tau
    )

    K = np.zeros((
        N_TIME * N_CHANNELS,
        N_TIME * N_CHANNELS,
    ))

    for channel in range(N_CHANNELS):
        indices = np.arange(
            channel,
            N_TIME * N_CHANNELS,
            N_CHANNELS,
        )

        K[np.ix_(indices, indices)] = temporal

    return K


def svd_influence(A, g):
    """Compute minimum-variance inventory influence for identity noise."""

    U, s, Vt = np.linalg.svd(
        A,
        full_matrices=False,
    )

    if not np.all(s > 0):
        raise RuntimeError("Rank-deficient design matrix")

    h = U @ ((Vt @ g) / s)

    sigma = float(
        np.linalg.norm((Vt @ g) / s)
    )

    return h, sigma


def gls_influence(A, g, L):
    """Compute GLS influence in original measurement coordinates."""

    AK = solve_triangular(
        L,
        A,
        lower=True,
        check_finite=True,
    )

    h_white, sigma = svd_influence(AK, g)

    h = solve_triangular(
        L.T,
        h_white,
        lower=False,
        check_finite=True,
    )

    return h, sigma


def variance(h, K):
    value = float(h @ K @ h)

    if value <= 0:
        raise RuntimeError("Nonpositive inventory variance")

    return value


def main():
    A, g = load_problem()

    rng = np.random.default_rng(SEED)

    h_ols, sigma_ols_reported = svd_influence(A, g)

    # An influence vector must reproduce the inventory gradient
    # when applied to the measurement Jacobian.
    #
    # A.T @ h = g.

    ols_reconstruction_error = np.linalg.norm(
        A.T @ h_ols - g
    ) / max(np.linalg.norm(g), 1.0)

    print()
    print("M26-B2: Correlated-Noise Monte Carlo")
    print("=" * 92)

    print(
        "OLS influence reconstruction relative error:",
        f"{ols_reconstruction_error:.6e}",
    )

    if ols_reconstruction_error > 1e-6:
        raise RuntimeError(
            "OLS influence does not reproduce inventory gradient"
        )

    summary_records = []
    channel_records = []
    influence_records = []

    for tau in TAU_VALUES:
        K = temporal_covariance(tau)

        L = cholesky(
            K,
            lower=True,
            check_finite=True,
        )

        h_gls, sigma_gls_svd = gls_influence(
            A,
            g,
            L,
        )

        gls_reconstruction_error = np.linalg.norm(
            A.T @ h_gls - g
        ) / max(np.linalg.norm(g), 1.0)

        if gls_reconstruction_error > 1e-6:
            raise RuntimeError(
                f"GLS influence reconstruction failed: tau={tau}"
            )

        sigma_gls = np.sqrt(
            variance(h_gls, K)
        )

        sigma_ols = np.sqrt(
            variance(h_ols, K)
        )

        if not np.isclose(
            sigma_gls,
            sigma_gls_svd,
            rtol=1e-6,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"GLS covariance consistency failed: tau={tau}"
            )

        # Generate correlated Gaussian errors:
        #
        # epsilon = L z, z ~ N(0,I).
        #
        # Use the same samples for GLS and OLS.

        z = rng.standard_normal((
            N_TIME * N_CHANNELS,
            N_SAMPLES,
        ))

        epsilon = L @ z

        errors_gls = h_gls @ epsilon
        errors_ols = h_ols @ epsilon

        empirical_gls = float(
            np.std(errors_gls, ddof=1)
        )

        empirical_ols = float(
            np.std(errors_ols, ddof=1)
        )

        mean_gls = float(np.mean(errors_gls))
        mean_ols = float(np.mean(errors_ols))

        summary_records.append({
            "tau_s": tau,
            "samples": N_SAMPLES,
            "gls_sigma_predicted_kg": sigma_gls,
            "gls_sigma_empirical_kg": empirical_gls,
            "gls_relative_sigma_error":
                empirical_gls / sigma_gls - 1.0,
            "gls_mean_error_kg": mean_gls,
            "ols_sigma_predicted_kg": sigma_ols,
            "ols_sigma_empirical_kg": empirical_ols,
            "ols_relative_sigma_error":
                empirical_ols / sigma_ols - 1.0,
            "ols_mean_error_kg": mean_ols,
            "ols_sigma_reported_kg":
                sigma_ols_reported,
            "gls_reconstruction_relative_error":
                gls_reconstruction_error,
        })

        # Channel-specific variance decomposition.
        #
        # With independent channels, the four channel
        # contributions sum to the total OLS variance.

        channel_variance_sum = 0.0

        for channel, name in enumerate(CHANNEL_NAMES):
            indices = np.arange(
                channel,
                N_TIME * N_CHANNELS,
                N_CHANNELS,
            )

            hc = h_ols[indices]

            Kc = K[np.ix_(indices, indices)]

            vc = float(hc @ Kc @ hc)

            channel_variance_sum += vc

            channel_records.append({
                "tau_s": tau,
                "channel": name,
                "variance_kg2": vc,
                "sigma_equivalent_kg": np.sqrt(vc),
                "fraction_total_variance":
                    vc / (sigma_ols**2),
            })

            # Record both OLS and GLS influence coefficients.
            #
            # These coefficients multiply measurement errors
            # expressed in units of their marginal sigma.

            for time_index, row_index in enumerate(indices):
                influence_records.append({
                    "tau_s": tau,
                    "time_s": time_index * DT,
                    "channel": name,
                    "ols_influence_kg_per_sigma":
                        h_ols[row_index],
                    "gls_influence_kg_per_sigma":
                        h_gls[row_index],
                })

        if not np.isclose(
            channel_variance_sum,
            sigma_ols**2,
            rtol=1e-8,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Channel variance decomposition failed: tau={tau}"
            )

        print()
        print(f"Correlation time: {tau:.0f} s")
        print("-" * 92)

        print(
            f"GLS predicted: {sigma_gls:12.6f} kg"
            f" | empirical: {empirical_gls:12.6f} kg"
            f" | discrepancy: "
            f"{100*(empirical_gls/sigma_gls-1):+7.3f}%"
        )

        print(
            f"OLS predicted: {sigma_ols:12.6f} kg"
            f" | empirical: {empirical_ols:12.6f} kg"
            f" | discrepancy: "
            f"{100*(empirical_ols/sigma_ols-1):+7.3f}%"
        )

        print(
            f"OLS incorrectly reported: "
            f"{sigma_ols_reported:.6f} kg"
        )

    summary = pd.DataFrame(summary_records)
    channels = pd.DataFrame(channel_records)
    influence = pd.DataFrame(influence_records)

    summary_path = (
        OUT / "correlated_noise_monte_carlo.csv"
    )

    channels_path = (
        OUT / "correlated_noise_channel_variance.csv"
    )

    influence_path = (
        OUT / "correlated_noise_influence.csv"
    )

    summary.to_csv(summary_path, index=False)
    channels.to_csv(channels_path, index=False)
    influence.to_csv(influence_path, index=False)

    print()
    print("=" * 92)
    print("Channel variance fractions at tau=120 s")
    print("-" * 92)

    selected = channels[
        channels.tau_s == 120.0
    ]

    for row in selected.itertuples():
        print(
            f"{row.channel:20s} "
            f"{100*row.fraction_total_variance:8.3f}%"
        )

    print()
    print("Largest absolute OLS influence coefficients")
    print("at tau=120 s")
    print("-" * 92)

    selected = influence[
        influence.tau_s == 120.0
    ].copy()

    selected["abs_influence"] = (
        selected.ols_influence_kg_per_sigma.abs()
    )

    selected = selected.nlargest(
        12,
        "abs_influence",
    )

    for row in selected.itertuples():
        print(
            f"t={row.time_s:6.1f} s "
            f"{row.channel:20s} "
            f"h={row.ols_influence_kg_per_sigma:+12.6f}"
        )

    print()
    print("Saved:")
    print(" ", summary_path)
    print(" ", channels_path)
    print(" ", influence_path)


if __name__ == "__main__":
    main()
