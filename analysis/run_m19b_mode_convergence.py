#!/usr/bin/env python3
"""M1.9b: Spatial convergence of physical observability eigenmodes."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path("benchmarks/results-m19a")
OUTPUT = Path("benchmarks/results-m19b")
OUTPUT.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = (100, 200, 400)
WINDOWS = ("full", "late")

# Physical domain length, meters.
LENGTH = 100_000.0

# Prior standard-deviation factors.
PRESSURE_PRIOR = 1.0e4
FLOW_PRIOR = 1.0

# Common comparison grid.
X = np.linspace(0.0, LENGTH, 1601)


def load_mode(window, n):
    path = SOURCE / f"{window}_n{n}_vectors.csv"

    vectors = np.loadtxt(path, delimiter=",")

    assert vectors.shape == (2*n + 1, 3), (
        f"Unexpected shape: {path}, {vectors.shape}"
    )

    v = vectors[:, 0]

    assert np.all(np.isfinite(v))
    assert abs(np.linalg.norm(v) - 1.0) < 1.0e-8

    dx = LENGTH / n

    xp = dx * np.arange(n + 1)
    xm = dx * (np.arange(n) + 0.5)

    pressure = np.concatenate((
        [0.0],
        PRESSURE_PRIOR * v[:n],
    ))

    flow = FLOW_PRIOR * v[n:2*n]
    inlet_flow = FLOW_PRIOR * v[2*n]

    return {
        "n": n,
        "pressure": np.interp(X, xp, pressure),
        "flow": np.interp(X, xm, flow),
        "inlet_flow": inlet_flow,
        "raw_pressure": pressure,
        "raw_flow": flow,
    }


def inner(a, b):
    return np.trapezoid(a*b, X) / LENGTH


def norm(a):
    return np.sqrt(inner(a, a))


def compare(a, b):
    na = norm(a)
    nb = norm(b)

    if na <= 0 or nb <= 0:
        raise ValueError("Zero field norm")

    correlation = inner(a, b) / (na*nb)
    correlation = float(np.clip(correlation, -1.0, 1.0))

    shape_error = np.sqrt(
        max(0.0, 2.0*(1.0-correlation))
    )

    return correlation, shape_error


def align(reference, candidate):
    # Use the combined dimensionless eigenvector geometry,
    # rather than adding pressure and flow in incompatible units.
    #
    # Pressure and flow are separately normalized before computing
    # the sign-selection score.
    score = 0.0

    for field in ("pressure", "flow"):
        a = reference[field]
        b = candidate[field]

        na = norm(a)
        nb = norm(b)

        if na > 0 and nb > 0:
            score += inner(a, b) / (na*nb)

    sign = 1.0 if score >= 0 else -1.0

    return {
        **candidate,
        "pressure": sign*candidate["pressure"],
        "flow": sign*candidate["flow"],
        "inlet_flow": sign*candidate["inlet_flow"],
        "raw_pressure": sign*candidate["raw_pressure"],
        "raw_flow": sign*candidate["raw_flow"],
    }


rows = []

for window in WINDOWS:
    modes = {
        n: load_mode(window, n)
        for n in RESOLUTIONS
    }

    # Align every mode against the finest-grid dominant mode.
    reference = modes[400]

    for n in (100, 200):
        modes[n] = align(reference, modes[n])

    for field, ylabel, filename in (
        ("pressure", "Pressure perturbation (Pa)",
         "pressure"),
        ("flow", "Mass-flow perturbation (kg/s)",
         "flow"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))

        for n in RESOLUTIONS:
            values = np.sqrt(n)*modes[n][field]

            ax.plot(
                X / 1000.0,
                values,
                label=f"n={n}",
            )

        ax.set(
            xlabel="Distance along pipe (km)",
            ylabel=f"sqrt(n) × {ylabel}",
            title=f"M1.9b {window}: dominant {filename} mode",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            OUTPUT / f"{window}_{filename}_modes.png",
            dpi=180,
        )
        plt.close(fig)

    for n1, n2 in ((100, 200), (200, 400), (100, 400)):
        for field in ("pressure", "flow"):
            corr, error = compare(
                modes[n1][field],
                modes[n2][field],
            )

            rows.append({
                "window": window,
                "field": field,
                "n_coarse": n1,
                "n_fine": n2,
                "correlation": corr,
                "relative_shape_error": error,
                "coarse_scaled_norm":
                    np.sqrt(n1)*norm(modes[n1][field]),
                "fine_scaled_norm":
                    np.sqrt(n2)*norm(modes[n2][field]),
            })

            print(
                f"{window:4s} {field:8s} "
                f"{n1:3d}->{n2:3d} "
                f"rho={corr:.12f} "
                f"shape_error={error:.6e}"
            )

with (OUTPUT / "mode_convergence.csv").open(
    "w", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote results to {OUTPUT}")
