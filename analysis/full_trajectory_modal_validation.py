#!/usr/bin/env python3
"""M1.7f: Full-trajectory validation of a frozen rank-one predictor.

Reference:
    Exact time-varying tangent sensitivity H_800.csv.

Reduced model:
    Dominant spectral projector of final-step operator T_800.

Coordinates:
    Packed initial state:
      [p1,...,pn, internal flows, inlet flow]

    State-finder dimensionless coordinates:
      pressure / 1e5 Pa
      flow / 10 kg/s

Observation order:
    [inlet flow, outlet pressure] at every time step.
"""

from pathlib import Path
import numpy as np
from scipy.linalg import eig

N = 100
DIM = 2*N + 1
STEPS = 800
DT = 15.0

ROOT = Path("benchmarks/results-m17e")
ROOT.mkdir(parents=True, exist_ok=True)

# Original tangent-matrix observation scales.
H_FLOW_SCALE = 1.0
H_PRESSURE_SCALE = 1.0e4

# Noise standard deviations used in predictive analysis.
NOISE = np.array([0.01, 10.0])

# Physical scales of the initial-state coordinates.
STATE_SCALE = np.r_[
    np.full(N, 1.0e5),
    np.full(N+1, 10.0)
]

# Prior standard deviations in physical units.
PRIOR_SIGMA = np.r_[
    np.full(N, 1.0e4),
    np.full(N+1, 1.0)
]

# Prior standard deviations in state-finder coordinates.
PRIOR_SCALED = PRIOR_SIGMA / STATE_SCALE

# -------------------------------------------------------
# Exact time-varying tangent response.
# -------------------------------------------------------

H = np.loadtxt(
    "benchmarks/results-m17a/H_800.csv",
    delimiter=","
)

assert H.shape == (2*(STEPS+1), DIM), H.shape

# H is ordered as:
#   flow(t0), pressure(t0), flow(t1), pressure(t1), ...
#
# Convert to physical observation units.

H = H.reshape(STEPS+1, 2, DIM)

H[:, 0, :] *= H_FLOW_SCALE
H[:, 1, :] *= H_PRESSURE_SCALE

# -------------------------------------------------------
# Final-step state transition operator.
# -------------------------------------------------------

T = np.loadtxt(
    ROOT / "T_800.csv",
    delimiter=","
)

assert T.shape == (DIM, DIM)

# Interleaved independent physical coordinates:
#   inlet flow, internal flow 0, pressure 1, ...
#
# Convert to packed coordinates.

permutation = np.r_[
    np.arange(2, DIM, 2),
    np.arange(1, DIM, 2),
    [0]
]

assert len(np.unique(permutation)) == DIM

T = T[np.ix_(permutation, permutation)]

# Similarity transformation into state-finder coordinates.

T = (
    T * STATE_SCALE[None, :]
    / STATE_SCALE[:, None]
)

# -------------------------------------------------------
# Dominant left/right eigenpair.
# -------------------------------------------------------

values, left, right = eig(
    T,
    left=True,
    right=True
)

stable = np.where(np.abs(values) < 1.0)[0]

if len(stable) == 0:
    raise RuntimeError("No stable eigenvalues")

index = stable[np.argmax(np.abs(values[stable]))]

mu = values[index]
l = left[:, index]
r = right[:, index]

if abs(mu.imag) > 1e-10:
    raise RuntimeError(
        "Dominant eigenvalue is not sufficiently real"
    )

if np.max(np.abs(l.imag)) > 1e-8:
    raise RuntimeError("Complex dominant left mode")

if np.max(np.abs(r.imag)) > 1e-8:
    raise RuntimeError("Complex dominant right mode")

mu = float(mu.real)
l = l.real
r = r.real

inner = np.dot(l, r)

if abs(inner) < 1e-12:
    raise RuntimeError("Nearly defective eigenpair")

P = np.outer(r, l) / inner

projector_error = np.linalg.norm(P @ P - P)

if projector_error > 1e-10:
    raise RuntimeError("Invalid spectral projector")

# -------------------------------------------------------
# Physical boundary observation operator.
# -------------------------------------------------------

C = np.zeros((2, DIM))

# Inlet mass flow.
C[0, 2*N] = STATE_SCALE[2*N]

# Outlet pressure.
C[1, N-1] = STATE_SCALE[N-1]

# -------------------------------------------------------
# Initial perturbations.
# -------------------------------------------------------

rng = np.random.default_rng(20261013)

directions = {}

# Scale the right eigenvector so its largest
# prior-standardized component is approximately one.

right_prior = r / PRIOR_SCALED

directions["right_mode"] = (
    r / np.max(np.abs(right_prior))
)

# Independent random prior-scaled perturbations.

for j in range(5):
    z = rng.normal(size=DIM)

    # RMS prior-standardized amplitude = 1.
    z /= np.sqrt(np.mean(z*z))

    directions[f"random_{j+1}"] = PRIOR_SCALED*z

# Left-null perturbation:
# remove its dominant spectral component.

z = rng.normal(size=DIM)
x = PRIOR_SCALED*z

x = x - P @ x

if np.linalg.norm(x) < 1e-14:
    raise RuntimeError("Null projection failed")

# Normalize by prior-standardized RMS.
x /= np.sqrt(np.mean((x/PRIOR_SCALED)**2))

directions["left_null"] = x

# -------------------------------------------------------
# Error calculations.
# -------------------------------------------------------

times = np.arange(STEPS+1)*DT

check_minutes = [
    0, 1, 2, 5, 10, 20, 30, 40,
    50, 60, 80, 100, 150, 200
]

check_steps = [
    int(round(60*t/DT))
    for t in check_minutes
]

# Noise-normalized Euclidean error threshold.
TOLERANCE = 0.1

summary_rows = []
detail_rows = []

print("M1.7f: Full-trajectory modal validation")
print()
print(f"Dominant eigenvalue: {mu:.15f}")
print(f"Projector residual:  {projector_error:.3e}")
print(f"Noise tolerance:     {TOLERANCE}")
print()

for name, x0 in directions.items():

    # Exact time-varying tangent response.
    exact = np.einsum(
        "tij,j->ti",
        H,
        x0
    )

    # Frozen final-step rank-one prediction.
    amplitude = P @ x0

    reduced_state_observation = C @ amplitude

    powers = mu**np.arange(STEPS+1)

    reduced = (
        powers[:, None]
        * reduced_state_observation[None, :]
    )

    error = exact - reduced

    abs_flow = np.abs(error[:, 0])
    abs_pressure = np.abs(error[:, 1])

    noise_error = np.linalg.norm(
        error / NOISE[None, :],
        axis=1
    )

    exact_noise_norm = np.linalg.norm(
        exact / NOISE[None, :],
        axis=1
    )

    relative = (
        noise_error /
        np.maximum(exact_noise_norm, 1e-12)
    )

    # For each time, maximum error over all later times.
    future_max = np.maximum.accumulate(
        noise_error[::-1]
    )[::-1]

    eligible = np.where(
        future_max <= TOLERANCE
    )[0]

    if len(eligible):
        settling_step = int(eligible[0])
        settling_min = times[settling_step]/60
    else:
        settling_min = np.nan

    max_error = float(np.max(noise_error))

    print(f"Direction: {name}")
    print(
        f"  max noise-normalized error: "
        f"{max_error:.9e}"
    )
    print(
        f"  persistent settling time:  "
        f"{settling_min:.3f} min"
    )

    print(
        "  minute     flow_err_kg_s"
        "     pressure_err_Pa"
        "     noise_error"
        "     relative_error"
    )

    for k in check_steps:

        print(
            f"  {times[k]/60:6.1f}"
            f"  {abs_flow[k]:16.8e}"
            f"  {abs_pressure[k]:16.8e}"
            f"  {noise_error[k]:14.6e}"
            f"  {relative[k]:14.6e}"
        )

    print()

    summary_rows.append([
        name,
        max_error,
        settling_min,
        float(noise_error[80]),
        float(noise_error[200]),
        float(noise_error[400]),
        float(noise_error[800]),
    ])

    for k in range(STEPS+1):
        detail_rows.append([
            name,
            times[k]/60,
            exact[k, 0],
            exact[k, 1],
            reduced[k, 0],
            reduced[k, 1],
            abs_flow[k],
            abs_pressure[k],
            noise_error[k],
            relative[k],
        ])

# -------------------------------------------------------
# Save reproducible results.
# -------------------------------------------------------

with open(
    ROOT / "full_trajectory_summary.csv",
    "w"
) as f:

    f.write(
        "direction,max_noise_error,"
        "settling_min,"
        "noise_error_20min,"
        "noise_error_50min,"
        "noise_error_100min,"
        "noise_error_200min\n"
    )

    for row in summary_rows:
        f.write(",".join(map(str, row)) + "\n")

with open(
    ROOT / "full_trajectory_detail.csv",
    "w"
) as f:

    f.write(
        "direction,time_min,"
        "exact_flow_kg_s,"
        "exact_pressure_pa,"
        "reduced_flow_kg_s,"
        "reduced_pressure_pa,"
        "abs_flow_error_kg_s,"
        "abs_pressure_error_pa,"
        "noise_normalized_error,"
        "relative_error\n"
    )

    for row in detail_rows:
        f.write(",".join(map(str, row)) + "\n")

print("Outputs:")
print("  benchmarks/results-m17e/full_trajectory_summary.csv")
print("  benchmarks/results-m17e/full_trajectory_detail.csv")
print()
print("PASS M1.7f full-trajectory validation")
