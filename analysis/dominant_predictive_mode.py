#!/usr/bin/env python3
"""M1.7c: Dominant predictive mode identification.

Extracts the dominant prior-standardized initial-state direction
governing late-time boundary predictions.

Also evaluates:
  - physical pressure and flow components
  - initial inlet/outlet flow perturbation proxies
  - exponential decay rate
  - alignment with the assimilation-informed subspace
  - posterior variance reduction in the predictive direction

The mode is a right singular vector of an output sensitivity
matrix, not necessarily a state-transition eigenvector.
"""

from pathlib import Path
import csv

import numpy as np

ROOT = Path("benchmarks/results-m17a")

H = np.loadtxt(
    ROOT / "H_800.csv",
    delimiter=",",
)

assert H.shape == (1602, 201)

DT = 15.0
N_CELLS = 100
LENGTH = 100000.0

# Physical prior standard deviations.
prior_sigma = np.r_[
    np.full(100, 10000.0),
    np.full(101, 1.0),
]

# C++ initial-state coordinate normalization.
state_scale = np.r_[
    np.full(100, 1.0e5),
    np.full(101, 10.0),
]

# Physical observation normalization used by H.
output_scale = np.array([1.0, 1.0e4])

# Assumed measurement uncertainties.
obs_sigma = np.array([0.01, 10.0])

# H is in scaled-output/scaled-state coordinates.
# Convert to physical-output/prior-standardized-state
# sensitivities.
A = H * (
    prior_sigma / state_scale
)[np.newaxis, :]

A[0::2] *= output_scale[0]
A[1::2] *= output_scale[1]

# Dimensionless sensitivity in units of observation SD.
K = A.copy()

K[0::2] /= obs_sigma[0]
K[1::2] /= obs_sigma[1]

# Select future times, not the assimilation interval.
future_minutes = (20, 30, 50, 100)

rows = []

for minute in future_minutes:
    step = int(round(minute * 60.0 / DT))

    rows.extend([
        2 * step,
        2 * step + 1,
    ])

F = K[rows]

# Normalize each selected output sensitivity row.
# This identifies common state-space directions without
# giving greater weight to earlier, larger responses.
row_norms = np.linalg.norm(F, axis=1)

if np.any(row_norms == 0.0):
    raise RuntimeError("Zero future sensitivity row")

F_normalized = F / row_norms[:, np.newaxis]

U, singular_values, Vt = np.linalg.svd(
    F_normalized,
    full_matrices=False,
)

mode = Vt[0].copy()

# Fix arbitrary SVD sign for reproducibility.
if np.sum(mode[:100]) < 0.0:
    mode *= -1.0

rank_one_fraction = (
    singular_values[0]**2
    / np.sum(singular_values**2)
)

# Convert to physical initial-state perturbation.
physical_mode = prior_sigma * mode

pressure_mode = physical_mode[:100]

# Coordinate order:
# 100 pressure unknowns, 100 internal flow unknowns,
# followed by initial inlet flow.
interior_flow_mode = physical_mode[100:200]
inlet_flow_mode = physical_mode[200]

pressure_energy = np.sum(mode[:100]**2)
flow_energy = np.sum(mode[100:]**2)

print("M1.7c: Dominant predictive mode")
print()
print(f"Rank-one energy fraction: {rank_one_fraction:.12f}")
print(f"Pressure standardized energy: {pressure_energy:.12f}")
print(f"Flow standardized energy:     {flow_energy:.12f}")

print()
print("Physical perturbation amplitudes:")
print(
    "Pressure RMS:",
    f"{np.sqrt(np.mean(pressure_mode**2)):.6f} Pa"
)
print(
    "Pressure min/max:",
    f"{pressure_mode.min():.6f}",
    f"{pressure_mode.max():.6f} Pa"
)
print(
    "Internal flow RMS:",
    f"{np.sqrt(np.mean(interior_flow_mode**2)):.9f} kg/s"
)
print(
    "Initial inlet flow:",
    f"{inlet_flow_mode:.9f} kg/s"
)

# The pressure coordinate at node 0 is prescribed by the
# boundary condition, not an initial-state unknown.
# Therefore, pressure_mode contains nodes 1..100.
#
# Flow coordinate positions must be checked against the
# C++ initial-state convention before assigning a physical
# interface index to each flow component.

mode_file = ROOT / "dominant_predictive_mode.csv"

with mode_file.open("w", newline="") as file:
    writer = csv.writer(file, lineterminator="\n")

    writer.writerow([
        "state_index",
        "component",
        "position_m",
        "standardized_mode",
        "physical_perturbation",
    ])

    dx = LENGTH / N_CELLS

    for i in range(100):
        writer.writerow([
            i,
            "pressure",
            (i + 1) * dx,
            mode[i],
            physical_mode[i],
        ])

    for i in range(100):
        writer.writerow([
            100 + i,
            "interior_flow",
            "",
            mode[100 + i],
            physical_mode[100 + i],
        ])

    writer.writerow([
        200,
        "inlet_flow",
        0.0,
        mode[200],
        physical_mode[200],
    ])

print()
print("Mode file:", mode_file)

# --------------------------------------------------
# Decay rate of the dominant mode
# --------------------------------------------------

# Project physical future-output sensitivities onto
# the same initial-state direction.
#
# For consistency, use observation-normalized outputs.
times = []
amplitudes = []

for step in range(80, 801):
    block = K[2*step:2*step+2]

    response = block @ mode

    amplitude = np.linalg.norm(response)

    if amplitude > 0.0:
        times.append(step * DT)
        amplitudes.append(amplitude)

times = np.asarray(times)
amplitudes = np.asarray(amplitudes)

# Fit the region from 30 to 100 minutes.
fit_mask = (
    (times >= 30.0 * 60.0)
    & (times <= 100.0 * 60.0)
)

slope, intercept = np.polyfit(
    times[fit_mask],
    np.log(amplitudes[fit_mask]),
    1,
)

tau = -1.0 / slope

print()
print("Long-time exponential decay:")
print(f"Decay eigenvalue estimate: {slope:.12e} 1/s")
print(f"Time constant:             {tau:.6f} s")
print(f"Time constant:             {tau/60.0:.6f} min")

# --------------------------------------------------
# Assimilation observability of predictive mode
# --------------------------------------------------

for window_min in (5, 10, 20):

    last_step = int(round(
        window_min * 60.0 / DT
    ))

    Ka = K[:2*(last_step+1)]

    Ua, sa, Vta = np.linalg.svd(
        Ka,
        full_matrices=True,
    )

    # Posterior covariance in prior-standardized
    # coordinates:
    #
    # C_z = V diag(1/(1+s_i^2)) V.T
    #
    # Nullspace directions have shrinkage factor 1.
    coefficients = Vta @ mode

    shrinkage = np.ones(201)

    shrinkage[:len(sa)] = (
        1.0 / (1.0 + sa*sa)
    )

    posterior_variance = np.sum(
        shrinkage * coefficients**2
    )

    posterior_std = np.sqrt(
        posterior_variance
    )

    # Fraction of mode energy within the span of
    # directions having >=90% variance reduction.
    information = sa*sa / (1.0 + sa*sa)

    informed = information >= 0.9

    informed_alignment = np.sum(
        coefficients[:len(sa)][informed]**2
    )

    print()
    print(f"Assimilation window: {window_min} min")
    print(
        "Predictive mode posterior SD ratio:",
        f"{posterior_std:.12e}"
    )
    print(
        "Predictive mode variance reduction:",
        f"{100*(1-posterior_variance):.6f}%"
    )
    print(
        "Alignment with >=90%-informed subspace:",
        f"{informed_alignment:.12f}"
    )

print()
print("PASS M1.7c dominant predictive mode")
