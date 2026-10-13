#!/usr/bin/env python3
"""M1.7g: Time-dependent modal amplification and rank-one prediction.

Models:
    A: Frozen final-step eigenmode.
    B: Sampled instantaneous eigenvalue accumulation.
    C: Trajectory-aware rank-one sensitivity projection.

All errors are evaluated in physical boundary-observation units.
"""

import csv
from pathlib import Path

import numpy as np
from scipy.linalg import eig
from scipy.interpolate import PchipInterpolator

N = 100
DIM = 2*N + 1
STEPS = 800
DT = 15.0

ROOT = Path("benchmarks/results-m17e")
ROOT.mkdir(parents=True, exist_ok=True)

STATE_SCALE = np.r_[
    np.full(N, 1e5),
    np.full(N+1, 10.0)
]

PRIOR_SIGMA = np.r_[
    np.full(N, 1e4),
    np.full(N+1, 1.0)
]

PRIOR_SCALED = PRIOR_SIGMA / STATE_SCALE

NOISE = np.array([0.01, 10.0])

# ------------------------------------------------------
# Exact time-dependent boundary sensitivities.
# ------------------------------------------------------

H = np.loadtxt(
    "benchmarks/results-m17a/H_800.csv",
    delimiter=","
)

assert H.shape == (2*(STEPS+1), DIM)

H = H.reshape(STEPS+1, 2, DIM)

# Original observation scales from M1.6.
H[:, 0, :] *= 1.0
H[:, 1, :] *= 1e4

# H maps dimensionless state-finder coordinates
# to physical boundary observations.

# ------------------------------------------------------
# Dominant predictive direction from M1.7c.
# ------------------------------------------------------

with open(
    "benchmarks/results-m17a/"
    "dominant_predictive_mode.csv",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

v = np.array([
    float(row["standardized_mode"])
    for row in rows
])

assert v.shape == (DIM,)

v /= np.linalg.norm(v)

# ------------------------------------------------------
# Exact final-step modal projector.
# ------------------------------------------------------

permutation = np.r_[
    np.arange(2, DIM, 2),
    np.arange(1, DIM, 2),
    [0]
]

def load_scaled_operator(step):
    T = np.loadtxt(
        ROOT / f"T_{step}.csv",
        delimiter=","
    )

    assert T.shape == (DIM, DIM)

    T = T[np.ix_(permutation, permutation)]

    return (
        T * STATE_SCALE[None, :]
        / STATE_SCALE[:, None]
    )


def dominant_eigenpair(T):
    values, left, right = eig(
        T,
        left=True,
        right=True
    )

    stable = np.where(
        np.abs(values) < 1.0
    )[0]

    if len(stable) == 0:
        raise RuntimeError("No stable eigenvalue")

    index = stable[
        np.argmax(np.abs(values[stable]))
    ]

    mu = values[index]

    if abs(mu.imag) > 1e-10:
        raise RuntimeError(
            f"Complex dominant eigenvalue: {mu}"
        )

    l = left[:, index]
    r = right[:, index]

    if (
        np.max(np.abs(l.imag)) > 1e-8
        or np.max(np.abs(r.imag)) > 1e-8
    ):
        raise RuntimeError(
            "Complex dominant eigenvectors"
        )

    return float(mu.real), l.real, r.real


T_final = load_scaled_operator(800)

mu_final, l, r = dominant_eigenpair(T_final)

inner = np.dot(l, r)

if abs(inner) < 1e-12:
    raise RuntimeError("Nearly defective eigenpair")

P = np.outer(r, l) / inner

# Physical boundary observation operator.
C = np.zeros((2, DIM))

C[0, 2*N] = STATE_SCALE[2*N]
C[1, N-1] = STATE_SCALE[N-1]

# ------------------------------------------------------
# Sample instantaneous dominant eigenvalues.
# ------------------------------------------------------

sample_steps = np.array([
    20, 40, 80, 200, 400, 600, 800
])

sample_mu = []

for step in sample_steps:
    T = load_scaled_operator(int(step))
    mu, _, _ = dominant_eigenpair(T)

    if not 0.0 < mu < 1.0:
        raise RuntimeError(
            f"Invalid dominant eigenvalue at {step}"
        )

    sample_mu.append(mu)

sample_mu = np.array(sample_mu)

# Interpolate logarithmic eigenvalue increments.
#
# PCHIP is used to avoid excessive oscillation.
#
# For steps 1..19, use the earliest available
# eigenvalue as a constant extrapolation.
#
# This early extrapolation is a known approximation.

interpolator = PchipInterpolator(
    sample_steps,
    np.log(sample_mu),
    extrapolate=False
)

all_steps = np.arange(STEPS+1)

log_mu = np.empty(STEPS+1)

log_mu[0] = 0.0

for k in range(1, STEPS+1):
    if k < sample_steps[0]:
        log_mu[k] = np.log(sample_mu[0])
    else:
        log_mu[k] = float(interpolator(k))

# Accumulated logarithmic amplification.
log_amplification = np.cumsum(log_mu)

amplification_B = np.exp(log_amplification)

amplification_A = mu_final**all_steps

amplitude_ratio = (
    amplification_B / amplification_A
)

# ------------------------------------------------------
# Trajectory-aware rank-one response.
# ------------------------------------------------------

# Convert H into prior-standardized coordinates.
#
# z = x / PRIOR_SCALED
# y = H * diag(PRIOR_SCALED) * z

H_prior = H * PRIOR_SCALED[None, None, :]

# g_k = H_prior,k @ v.
g = np.einsum(
    "tij,j->ti",
    H_prior,
    v
)

# ------------------------------------------------------
# Construct test perturbations.
# ------------------------------------------------------

rng = np.random.default_rng(20261013)

directions = {}

# Final right eigenvector, normalized so the maximum
# prior-standardized component has magnitude one.

right_prior = r / PRIOR_SCALED

directions["right_mode"] = (
    r / np.max(np.abs(right_prior))
)

for j in range(5):
    z = rng.normal(size=DIM)
    z /= np.sqrt(np.mean(z*z))

    directions[f"random_{j+1}"] = PRIOR_SCALED*z

# Final frozen-time left-null perturbation.
z = rng.normal(size=DIM)
x = PRIOR_SCALED*z

x -= P @ x

x /= np.sqrt(
    np.mean((x/PRIOR_SCALED)**2)
)

directions["left_null"] = x

# ------------------------------------------------------
# Compare predictions.
# ------------------------------------------------------

def noise_norm(values):
    return np.linalg.norm(
        values / NOISE[None, :],
        axis=1
    )


def settling_time(errors, threshold=0.1):
    future_max = np.maximum.accumulate(
        errors[::-1]
    )[::-1]

    eligible = np.where(
        future_max <= threshold
    )[0]

    if not len(eligible):
        return np.nan

    return eligible[0]*DT/60.0


summary = []
details = []

check_minutes = [
    0, 5, 10, 20, 30, 40, 50,
    60, 80, 100, 150, 200
]

print("M1.7g: Time-dependent modal amplification")
print()
print(f"Final eigenvalue: {mu_final:.15f}")
print()
print("Sampled eigenvalues:")

for step, mu in zip(sample_steps, sample_mu):
    print(
        f"  t={step*DT/60:6.1f} min"
        f"  mu={mu:.15f}"
        f"  lambda={np.log(mu)/DT:.12e}"
    )

print()
print("Accumulated amplification correction:")

for minute in [5, 10, 20, 40, 60, 100, 200]:
    k = int(minute*60/DT)

    print(
        f"  t={minute:3d} min"
        f"  B/A={amplitude_ratio[k]:.9f}"
    )

print()

for name, x0 in directions.items():

    # Exact time-varying tangent response.
    exact = np.einsum(
        "tij,j->ti",
        H,
        x0
    )

    # Initial amplitude of final frozen eigenmode.
    modal_amplitude = P @ x0

    boundary_amplitude = C @ modal_amplitude

    # Model A: frozen amplification.
    A = (
        amplification_A[:, None]
        * boundary_amplitude[None, :]
    )

    # Model B: interpolated instantaneous eigenvalues.
    B = (
        amplification_B[:, None]
        * boundary_amplitude[None, :]
    )

    # Model C: exact trajectory-aware rank-one
    # sensitivity projection.
    z0 = x0 / PRIOR_SCALED

    coefficient = float(np.dot(v, z0))

    C_model = g * coefficient

    predictions = {
        "A_frozen": A,
        "B_variable_rate": B,
        "C_trajectory_rank1": C_model,
    }

    print(f"Direction: {name}")

    for model_name, predicted in predictions.items():

        error = exact - predicted

        E = noise_norm(error)

        relative = (
            E / np.maximum(
                noise_norm(exact),
                1e-12
            )
        )

        settling = settling_time(E)

        print(
            f"  {model_name:20s}"
            f" max_E={np.max(E):.6e}"
            f" settle_min={settling:8.3f}"
            f" E20={E[80]:.6e}"
            f" E50={E[200]:.6e}"
            f" E100={E[400]:.6e}"
            f" rel100={relative[400]:.6e}"
        )

        summary.append([
            name,
            model_name,
            np.max(E),
            settling,
            E[80],
            E[200],
            E[400],
            E[800],
            relative[400],
        ])

        for k in range(STEPS+1):
            details.append([
                name,
                model_name,
                k*DT/60,
                exact[k, 0],
                exact[k, 1],
                predicted[k, 0],
                predicted[k, 1],
                E[k],
                relative[k],
            ])

    print()

# ------------------------------------------------------
# Save results.
# ------------------------------------------------------

summary_path = ROOT / "m17g_summary.csv"

with summary_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "direction",
        "model",
        "max_noise_error",
        "settling_min",
        "E20",
        "E50",
        "E100",
        "E200",
        "relative_error_100min",
    ])

    writer.writerows(summary)

detail_path = ROOT / "m17g_detail.csv"

with detail_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "direction",
        "model",
        "time_min",
        "exact_flow",
        "exact_pressure",
        "predicted_flow",
        "predicted_pressure",
        "noise_error",
        "relative_error",
    ])

    writer.writerows(details)

amplification_path = ROOT / "m17g_amplification.csv"

with amplification_path.open("w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "time_min",
        "frozen_amplification",
        "variable_rate_amplification",
        "variable_over_frozen",
    ])

    for k in range(STEPS+1):
        writer.writerow([
            k*DT/60,
            amplification_A[k],
            amplification_B[k],
            amplitude_ratio[k],
        ])

print("Saved:")
print(f"  {summary_path}")
print(f"  {detail_path}")
print(f"  {amplification_path}")
print()
print("PASS M1.7g analysis")
