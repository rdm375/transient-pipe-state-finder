#!/usr/bin/env python3
"""M1.7b: Prediction-informed observability.

Quantifies uncertainty in future boundary predictions arising
from uncertain initial state, before and after assimilation.

Assumptions:
    - Linearized model around the reference trajectory.
    - Gaussian, independent initial-state prior errors.
    - Gaussian, temporally independent measurement errors.
    - Exact future boundary inputs and model parameters.
    - No additional future measurement assimilation.
"""

from pathlib import Path
import csv

import numpy as np

ROOT = Path("benchmarks/results-m17a")
OUTPUT = ROOT / "predictive_observability.csv"

DT = 15.0
N = 201

# Initial-state coordinates:
#   0:100    pressure nodes 1..100
#   100:200  internal flow values 0..99
#   200      initial inlet flow

STATE_SCALE = np.r_[
    np.full(100, 1.0e5),
    np.full(101, 10.0),
]

PRIOR_SIGMA_PHYSICAL = np.r_[
    np.full(100, 1.0e4),  # 10 kPa
    np.full(101, 1.0),    # 1 kg/s
]

# The tangent matrix has scaled output rows:
# even rows: inlet flow / 1 kg/s
# odd rows: outlet pressure / 10000 Pa
OUTPUT_SCALE = np.array([1.0, 1.0e4])

# Assumed measurement noise, in physical units.
OBS_SIGMA_PHYSICAL = np.array([0.01, 10.0])

ASSIMILATION_WINDOWS_MIN = (5, 10, 20)

# Absolute times measured from the initial state.
FORECAST_TIMES_MIN = (30, 50, 100, 200)

H = np.loadtxt(
    ROOT / "H_800.csv",
    delimiter=",",
)

if H.shape != (1602, N):
    raise RuntimeError(f"Unexpected H shape: {H.shape}")

if not np.all(np.isfinite(H)):
    raise RuntimeError("Nonfinite sensitivity matrix")

# Initial-state standard deviations in the coordinates
# used by the C++ sensitivity matrix.
prior_sigma_scaled = (
    PRIOR_SIGMA_PHYSICAL / STATE_SCALE
)

# Physical output sensitivities multiplied by the
# initial-state prior standard deviations.
#
# A = F B^(1/2)
#
# Rows alternate between inlet flow and outlet pressure.
A = H * prior_sigma_scaled[np.newaxis, :]

A[0::2] *= OUTPUT_SCALE[0]
A[1::2] *= OUTPUT_SCALE[1]

# Measurement-noise whitening:
#
# K = R^(-1/2) H_a B^(1/2)
#
# Since A is in physical output units, divide each
# observation row by its physical noise SD.
obs_sigma = np.tile(
    OBS_SIGMA_PHYSICAL,
    H.shape[0] // 2,
)

K_all = A / obs_sigma[:, np.newaxis]


def posterior_prediction_std(A_future, Vt, s):
    """Compute posterior predictive marginal SDs via SVD.

    K = U diag(s) Vt.

    Posterior covariance in prior-standardized coordinates:

        C_z = I - V diag(s^2/(1+s^2)) V.T

    Use an equivalent nonnegative sum to avoid subtracting
    nearly equal quantities for strongly constrained modes.
    """

    # full_matrices=True gives a complete orthogonal basis,
    # including the nullspace when K has fewer than N rows.
    projected = A_future @ Vt.T

    shrinkage = np.ones(N)

    shrinkage[:len(s)] = 1.0 / (1.0 + s*s)

    variance = (
        projected * projected
    ) @ shrinkage

    return np.sqrt(np.maximum(variance, 0.0))


def effective_dimension(s):
    lam = s*s
    return np.sum(lam / (1.0 + lam))


rows = []

for window_min in ASSIMILATION_WINDOWS_MIN:

    last_assimilated_step = int(
        round(window_min * 60.0 / DT)
    )

    # Include both channels at t=0 through the final
    # assimilation sample, inclusive.
    assimilation_rows = 2 * (
        last_assimilated_step + 1
    )

    K = K_all[:assimilation_rows]

    # Full right-singular-vector basis is needed to
    # preserve posterior variance in the nullspace.
    U, s, Vt = np.linalg.svd(
        K,
        full_matrices=True,
    )

    d_eff = effective_dimension(s)

    print()
    print(
        f"Assimilation window: {window_min} min "
        f"(d_eff={d_eff:.6f})"
    )

    print(
        "Time   Output             "
        "Prior SD       Posterior SD    Reduction"
    )

    for forecast_min in FORECAST_TIMES_MIN:

        step = int(round(
            forecast_min * 60.0 / DT
        ))

        if step <= last_assimilated_step:
            raise RuntimeError(
                "Forecast time must follow assimilation"
            )

        # Two physical-output rows at the forecast time.
        future = A[2*step:2*step+2]

        prior_std = np.linalg.norm(
            future,
            axis=1,
        )

        posterior_std = posterior_prediction_std(
            future,
            Vt,
            s,
        )

        for channel, name, unit in (
            (0, "inlet_flow", "kg/s"),
            (1, "outlet_pressure", "Pa"),
        ):
            prior = prior_std[channel]
            posterior = posterior_std[channel]

            reduction = (
                1.0 - posterior / prior
                if prior > 0.0
                else 0.0
            )

            rows.append({
                "assimilation_min": window_min,
                "forecast_min": forecast_min,
                "output": name,
                "unit": unit,
                "effective_dimension": d_eff,
                "prior_std": prior,
                "posterior_std": posterior,
                "std_reduction_fraction": reduction,
            })

            print(
                f"{forecast_min:4d}   "
                f"{name:17s} "
                f"{prior:13.6e} "
                f"{posterior:13.6e} "
                f"{100*reduction:9.3f}%"
            )

with OUTPUT.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"PASS M1.7b: {len(rows)} predictive diagnostics")
print(f"Summary: {OUTPUT}")
