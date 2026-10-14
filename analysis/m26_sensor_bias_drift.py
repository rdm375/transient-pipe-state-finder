#!/usr/bin/env python3
"""M26-C1: Systematic sensor bias and drift sensitivity.

Evaluate terminal inventory estimation errors from deterministic
measurement biases under the M26-B correlation model.

All results are linearized and conditional on the existing
joint inverse problem and hypothetical measurement noise model.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.linalg import cholesky

from m26_correlated_noise_monte_carlo import (
    load_problem,
    temporal_covariance,
    svd_influence,
    gls_influence,
    variance,
    N_TIME,
    N_CHANNELS,
    DT,
    CHANNEL_NAMES,
    SIGMA_PHYSICAL,
)

OUT = Path("benchmarks/results-m26")
OUT.mkdir(parents=True, exist_ok=True)

TAU = 120.0

# Illustrative systematic error magnitudes.
#
# Pressure channels: 5000 Pa.
# Flow channels: 0.5 kg/s.
#
# These are scenarios, not calibrated instrument specifications.

REFERENCE_BIAS = np.array([
    5000.0,
    0.5,
    5000.0,
    0.5,
])

PROFILES = [
    "constant",
    "linear",
    "late_onset",
]


def profile_values(name):
    """Return a dimensionless time profile."""

    times = DT * np.arange(N_TIME)

    if name == "constant":
        return np.ones(N_TIME)

    if name == "linear":
        return times / times[-1]

    if name == "late_onset":
        return np.maximum(
            0.0,
            (times - 300.0) / 300.0,
        )

    raise ValueError(name)


def bias_vector(channel, profile, physical_amplitude):
    """Construct systematic error in marginal-sigma units."""

    values = profile_values(profile)

    b = np.zeros(N_TIME * N_CHANNELS)

    indices = np.arange(
        channel,
        N_TIME * N_CHANNELS,
        N_CHANNELS,
    )

    b[indices] = (
        physical_amplitude
        / SIGMA_PHYSICAL[channel]
    ) * values

    return b


def main():
    A, g = load_problem()

    K = temporal_covariance(TAU)

    L = cholesky(
        K,
        lower=True,
        check_finite=True,
    )

    h_ols, sigma_ols_reported = svd_influence(
        A,
        g,
    )

    h_gls, sigma_gls_svd = gls_influence(
        A,
        g,
        L,
    )

    sigma_gls = np.sqrt(
        variance(h_gls, K)
    )

    sigma_ols_actual = np.sqrt(
        variance(h_ols, K)
    )

    assert np.isclose(
        sigma_gls,
        sigma_gls_svd,
        rtol=1e-6,
    )

    # Reference threshold: one GLS random-noise sigma.
    threshold = sigma_gls

    records = []

    print()
    print("M26-C1: Sensor Bias and Drift")
    print("=" * 100)

    print(f"Correlation time: {TAU:.1f} s")
    print(f"GLS random sigma: {sigma_gls:.6f} kg")
    print(f"OLS actual random sigma: {sigma_ols_actual:.6f} kg")
    print(f"OLS reported random sigma: {sigma_ols_reported:.6f} kg")

    for profile in PROFILES:
        print()
        print(f"Profile: {profile}")
        print("-" * 100)

        for channel, name in enumerate(CHANNEL_NAMES):
            # Sensitivity to one physical unit:
            #   Pa for pressure channels
            #   kg/s for flow channels.

            b_unit = bias_vector(
                channel,
                profile,
                1.0,
            )

            sensitivity_gls = float(
                h_gls @ b_unit
            )

            sensitivity_ols = float(
                h_ols @ b_unit
            )

            reference_amplitude = (
                REFERENCE_BIAS[channel]
            )

            error_gls = (
                sensitivity_gls * reference_amplitude
            )

            error_ols = (
                sensitivity_ols * reference_amplitude
            )

            threshold_gls = (
                np.inf
                if sensitivity_gls == 0
                else threshold / abs(sensitivity_gls)
            )

            threshold_ols = (
                np.inf
                if sensitivity_ols == 0
                else threshold / abs(sensitivity_ols)
            )

            mse_gls = (
                sigma_gls**2 + error_gls**2
            )

            mse_ols = (
                sigma_ols_actual**2 + error_ols**2
            )

            units = (
                "Pa"
                if "pressure" in name
                else "kg/s"
            )

            records.append({
                "tau_s": TAU,
                "profile": profile,
                "channel": name,
                "units": units,
                "reference_amplitude": reference_amplitude,
                "gls_inventory_sensitivity_per_unit":
                    sensitivity_gls,
                "ols_inventory_sensitivity_per_unit":
                    sensitivity_ols,
                "gls_reference_bias_error_kg":
                    error_gls,
                "ols_reference_bias_error_kg":
                    error_ols,
                "gls_amplitude_for_one_gls_sigma":
                    threshold_gls,
                "ols_amplitude_for_one_gls_sigma":
                    threshold_ols,
                "gls_random_sigma_kg":
                    sigma_gls,
                "ols_actual_random_sigma_kg":
                    sigma_ols_actual,
                "ols_reported_random_sigma_kg":
                    sigma_ols_reported,
                "gls_reference_rmse_kg":
                    np.sqrt(mse_gls),
                "ols_reference_rmse_kg":
                    np.sqrt(mse_ols),
            })

            print(
                f"{name:20s} "
                f"GLS error={error_gls:+12.4f} kg | "
                f"OLS error={error_ols:+12.4f} kg | "
                f"GLS threshold={threshold_gls:12.4f} {units}"
            )

    results = pd.DataFrame(records)

    output = OUT / "sensor_bias_drift_sweep.csv"

    results.to_csv(
        output,
        index=False,
    )

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()
