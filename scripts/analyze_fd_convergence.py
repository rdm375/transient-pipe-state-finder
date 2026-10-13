#!/usr/bin/env python3

from pathlib import Path
import csv
import numpy as np

root = Path("benchmarks/results-m16b/fd-sweep")

steps = ["1e-3", "1e-4", "1e-5", "1e-6"]
horizons = [80, 100, 200, 400, 800]

matrices = {}
spectra = {}

summary = []

for step in steps:
    directory = root / f"h-{step}"

    print()
    print("=" * 72)
    print(f"Finite-difference step: {step}")
    print("=" * 72)

    for horizon in horizons:
        path = directory / f"H_{horizon}.csv"

        H = np.loadtxt(path, delimiter=",")

        expected = (2*(horizon+1), 201)

        if H.shape != expected:
            raise RuntimeError(
                f"{path}: unexpected shape {H.shape}"
            )

        if not np.all(np.isfinite(H)):
            raise RuntimeError(
                f"{path}: nonfinite entries"
            )

        matrices[(step, horizon)] = H

        singular_values = np.linalg.svd(
            H,
            compute_uv=False
        )

        # Include structural zeros for wide matrices.
        full_spectrum = np.zeros(201)

        full_spectrum[:len(singular_values)] = (
            singular_values
        )

        spectra[(step, horizon)] = full_spectrum

        sigma_max = full_spectrum[0]
        sigma_min = full_spectrum[-1]

        tolerance = (
            max(H.shape)
            * np.finfo(float).eps
            * sigma_max
        )

        rank = int(
            np.count_nonzero(
                singular_values > tolerance
            )
        )

        condition = (
            sigma_max/sigma_min
            if sigma_min > tolerance
            else float("inf")
        )

        summary.append({
            "fd_step": step,
            "horizon": horizon,
            "rank": rank,
            "nullity": 201-rank,
            "sigma_max": sigma_max,
            "sigma_min": sigma_min,
            "condition": condition
        })

        print(
            f"N={horizon:3d} "
            f"rank={rank:3d} "
            f"sigma_min={sigma_min:.9e} "
            f"condition={condition:.9e}"
        )

with (root / "fd_singular_values.csv").open(
    "w", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(summary[0])
    )

    writer.writeheader()
    writer.writerows(summary)


print()
print("=" * 72)
print("Sensitivity matrix convergence")
print("=" * 72)

# Relative Frobenius differences between consecutive
# finite-difference steps.
#
# Central differences have formal O(h^2) truncation
# error for sufficiently smooth functions.
#
# With successive steps differing by a factor of 10,
# the difference norms should decrease by roughly
# 100 until floating-point error dominates.

differences = {}

for horizon in horizons:
    print()
    print(f"Horizon {horizon}")

    for coarse, fine in zip(steps[:-1], steps[1:]):
        A = matrices[(coarse, horizon)]
        B = matrices[(fine, horizon)]

        numerator = np.linalg.norm(A-B, ord="fro")

        denominator = max(
            np.linalg.norm(B, ord="fro"),
            np.finfo(float).tiny
        )

        relative = numerator/denominator

        differences[(coarse, fine, horizon)] = (
            relative
        )

        print(
            f"  {coarse} -> {fine}: "
            f"relative_Frobenius_difference="
            f"{relative:.9e}"
        )

    d1 = differences[("1e-3", "1e-4", horizon)]
    d2 = differences[("1e-4", "1e-5", horizon)]
    d3 = differences[("1e-5", "1e-6", horizon)]

    print(
        "  difference ratios: "
        f"{d1/max(d2, 1e-300):.6g}, "
        f"{d2/max(d3, 1e-300):.6g}"
    )


print()
print("=" * 72)
print("Smallest singular-value convergence")
print("=" * 72)

for horizon in horizons:
    print()
    print(f"Horizon {horizon}")

    for step in steps:
        s = spectra[(step, horizon)]

        print(
            f"  h={step:>4}: "
            f"sigma_min={s[-1]:.12e}"
        )

    if horizon == 80:
        print(
            "  Structural nullity >= 39; "
            "smallest singular value is zero."
        )
        continue

    reference = spectra[("1e-5", horizon)][-1]

    for step in steps:
        value = spectra[(step, horizon)][-1]

        relative = abs(value-reference)/max(
            abs(reference),
            np.finfo(float).tiny
        )

        print(
            f"  h={step:>4}: "
            f"relative_difference_from_1e-5="
            f"{relative:.9e}"
        )


print()
print("=" * 72)
print("Nested-horizon singular-value monotonicity")
print("=" * 72)

# For H_long formed by appending rows to H_short:
#
# H_long.T @ H_long - H_short.T @ H_short
# is positive semidefinite.
#
# Therefore each singular value must be nondecreasing,
# in exact arithmetic, when sorted in descending order.
#
# Here we test the computed singular-value spectra.
#
# A tolerance is needed because the smallest values
# may be dominated by differentiation or SVD errors.

for step in steps:
    print()
    print(f"Finite-difference step {step}")

    for short, long in zip(
        horizons[:-1],
        horizons[1:]
    ):
        s_short = spectra[(step, short)]
        s_long = spectra[(step, long)]

        differences_s = s_long-s_short

        worst = float(np.min(differences_s))

        # Absolute tolerance relative to the largest
        # singular value. This is a numerical diagnostic,
        # not a physical observability threshold.
        tolerance = (
            1.0e-10
            * max(s_long[0], s_short[0])
        )

        violations = int(
            np.count_nonzero(
                differences_s < -tolerance
            )
        )

        print(
            f"  {short:3d} -> {long:3d}: "
            f"worst_delta={worst:.9e} "
            f"violations={violations}"
        )


print()
print(
    "Wrote:",
    root / "fd_singular_values.csv"
)
