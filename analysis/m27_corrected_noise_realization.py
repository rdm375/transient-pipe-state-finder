"""Create M27 noise realization consistent with M26 sensor sigma.

Preserves the original M27-A1 realization as a 2x-noise experiment.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular

from m26_correlated_noise_monte_carlo import temporal_covariance
from m26_calibration_prior_sensitivity import TAU

SOURCE = Path(
    "benchmarks/results-m27/"
    "correlated_measurement_realization.csv"
)

DESTINATION = Path(
    "benchmarks/results-m27/"
    "correlated_measurement_realization_sigma0p5.csv"
)

SIGMA = 0.5

if DESTINATION.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing result: {DESTINATION}"
    )

original = pd.read_csv(SOURCE)

required = {
    "measurement_index",
    "noise_normalized",
    "whitened_noise",
}

if not required.issubset(original.columns):
    raise RuntimeError(
        f"Missing columns: {required - set(original.columns)}"
    )

indices = original["measurement_index"].to_numpy()

if not np.array_equal(
    indices,
    np.arange(len(original))
):
    raise RuntimeError("Unexpected measurement ordering")

K = temporal_covariance(TAU)

if K.shape != (len(original), len(original)):
    raise RuntimeError("Covariance dimension mismatch")

L = cholesky(K, lower=True)

original_noise = original[
    "noise_normalized"
].to_numpy(float)

original_whitened = solve_triangular(
    L,
    original_noise,
    lower=True,
)

stored_whitened = original[
    "whitened_noise"
].to_numpy(float)

if not np.allclose(
    original_whitened,
    stored_whitened,
    rtol=1e-10,
    atol=1e-10,
):
    raise RuntimeError(
        "Original whitened noise does not match covariance"
    )

# Preserve the exact random realization, changing only
# the sensor standard deviation.
corrected_noise = SIGMA * original_noise

# Whitening with the actual covariance C = sigma^2 K:
#
# C^(-1/2) noise = (sigma L)^(-1) noise.
#
# This equals the original standard-normal innovation.
corrected_whitened = solve_triangular(
    SIGMA * L,
    corrected_noise,
    lower=True,
)

result = pd.DataFrame({
    "measurement_index": indices,
    "noise_normalized": corrected_noise,
    "whitened_noise": corrected_whitened,
})

result.to_csv(
    DESTINATION,
    index=False,
    float_format="%.17g",
)

print("M27 corrected measurement realization")
print("source:", SOURCE)
print("destination:", DESTINATION)
print("measurements:", len(result))
print("sigma:", SIGMA)

print(
    "original RMS:",
    np.sqrt(np.mean(original_noise**2)),
)

print(
    "corrected RMS:",
    np.sqrt(np.mean(corrected_noise**2)),
)

print(
    "corrected standardized innovation norm:",
    np.linalg.norm(corrected_whitened),
)

print(
    "maximum innovation difference:",
    np.max(
        np.abs(
            corrected_whitened - stored_whitened
        )
    ),
)

print("PASS corrected covariance convention")
