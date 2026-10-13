#!/usr/bin/env python3
"""M1.7a: Prior-whitened pipeline observability analysis."""

from pathlib import Path
import csv

import numpy as np

ROOT = Path("benchmarks/results-m17a")
HORIZONS = (80, 100, 200, 400, 800)

N_PRESSURE = 100
N_FLOW = 101
N_STATE = N_PRESSURE + N_FLOW

# Scaling conventions verified against the C++ implementation.
STATE_SCALE = np.r_[
    np.full(N_PRESSURE, 1.0e5),
    np.full(N_FLOW, 10.0),
]

OBSERVATION_SCALE = (1.0, 1.0e4)

# Illustrative observation-error standard deviations.
SIGMA_OBS_FLOW = 0.01       # kg/s
SIGMA_OBS_PRESSURE = 10.0   # Pa

# Illustrative initial-state uncertainty.
PRESSURE_PRIORS = (1.0e3, 1.0e4, 1.0e5)  # Pa
FLOW_PRIORS = (0.1, 1.0, 10.0)           # kg/s

rows = []

for horizon in HORIZONS:
    H = np.loadtxt(
        ROOT / f"H_{horizon}.csv",
        delimiter=",",
    )

    expected_shape = (2 * (horizon + 1), N_STATE)

    if H.shape != expected_shape:
        raise RuntimeError(
            f"H_{horizon}: {H.shape} != {expected_shape}"
        )

    if not np.all(np.isfinite(H)):
        raise RuntimeError("Nonfinite sensitivity matrix")

    # H is already scaled by C++ observation and state scales.
    #
    # Convert observation errors into those same coordinates.
    obs_sigma_scaled = np.empty(H.shape[0])

    obs_sigma_scaled[0::2] = (
        SIGMA_OBS_FLOW / OBSERVATION_SCALE[0]
    )

    obs_sigma_scaled[1::2] = (
        SIGMA_OBS_PRESSURE / OBSERVATION_SCALE[1]
    )

    for sigma_p in PRESSURE_PRIORS:
        for sigma_m in FLOW_PRIORS:

            prior_sigma_physical = np.r_[
                np.full(N_PRESSURE, sigma_p),
                np.full(N_FLOW, sigma_m),
            ]

            prior_sigma_scaled = (
                prior_sigma_physical / STATE_SCALE
            )

            # K = R^(-1/2) H B^(1/2)
            K = (
                H * prior_sigma_scaled[np.newaxis, :]
            ) / obs_sigma_scaled[:, np.newaxis]

            # Symmetric positive-semidefinite information matrix.
            G = K.T @ K
            G = 0.5 * (G + G.T)

            eigvals, V = np.linalg.eigh(G)

            # Remove small negative eigenvalues caused by roundoff.
            eigvals = np.maximum(eigvals, 0.0)

            shrinkage = 1.0 / (1.0 + eigvals)

            information_fraction = (
                eigvals / (1.0 + eigvals)
            )

            effective_dimension = information_fraction.sum()

            strongly_constrained = np.count_nonzero(
                information_fraction >= 0.9
            )

            moderately_constrained = np.count_nonzero(
                information_fraction >= 0.5
            )

            # Posterior covariance in standardized prior coordinates:
            # C_z = V diag(1/(1+lambda)) V.T
            #
            # Only the diagonal is needed here.
            posterior_variance_ratio = (
                (V * V) @ shrinkage
            )

            posterior_sigma_physical = (
                prior_sigma_physical
                * np.sqrt(
                    np.clip(
                        posterior_variance_ratio,
                        0.0,
                        1.0,
                    )
                )
            )

            pressure_posterior_rms = np.sqrt(
                np.mean(
                    posterior_sigma_physical[:N_PRESSURE] ** 2
                )
            )

            flow_posterior_rms = np.sqrt(
                np.mean(
                    posterior_sigma_physical[N_PRESSURE:] ** 2
                )
            )

            condition_number = (
                (1.0 + eigvals[-1]) /
                (1.0 + eigvals[0])
            )

            row = {
                "horizon": horizon,
                "duration_min": horizon * 15.0 / 60.0,
                "prior_pressure_pa": sigma_p,
                "prior_flow_kg_s": sigma_m,
                "effective_dimension": effective_dimension,
                "directions_50pct": moderately_constrained,
                "directions_90pct": strongly_constrained,
                "posterior_pressure_rms_pa":
                    pressure_posterior_rms,
                "posterior_flow_rms_kg_s":
                    flow_posterior_rms,
                "transformed_hessian_condition":
                    condition_number,
            }

            rows.append(row)

            print(
                f"h={horizon:3d} "
                f"prior_p={sigma_p:8.0f} "
                f"prior_m={sigma_m:5.1f} "
                f"d_eff={effective_dimension:8.3f} "
                f"N90={strongly_constrained:3d} "
                f"post_p={pressure_posterior_rms:10.3f} "
                f"post_m={flow_posterior_rms:10.6f}"
            )

output = ROOT / "prior_observability_summary.csv"

with output.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"PASS M1.7a: {len(rows)} configurations")
print(f"Summary: {output}")
