#!/usr/bin/env python3
"""M2.5-A validation: numerical-rank sensitivity."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("benchmarks/results-m25")

CONFIGURATIONS = [
    "A_baseline",
    "B_inlet_pressure_x10",
    "C_outlet_flow_x10",
    "D_response_channels_x10",
    "E_input_channels_x0p1",
]

TOLERANCES = [
    1.0e-6,
    1.0e-8,
    1.0e-10,
    1.0e-12,
    1.0e-14,
]

records = []

for name in CONFIGURATIONS:

    spectra = {}

    for kind in (
        "joint",
        "fixed_boundary",
        "profiled",
    ):

        path = ROOT / f"{name}_{kind}_spectrum.csv"

        df = pd.read_csv(path)

        singular = df["singular_value"].to_numpy()

        if not np.isfinite(singular).all():
            raise ValueError(
                f"Nonfinite singular values: {path}"
            )

        spectra[kind] = singular

    for tolerance in TOLERANCES:

        record = {
            "configuration": name,
            "relative_tolerance": tolerance,
        }

        for kind, singular in spectra.items():

            threshold = tolerance * singular[0]

            record[f"{kind}_rank"] = int(
                np.count_nonzero(
                    singular > threshold
                )
            )

        records.append(record)


result = pd.DataFrame(records)

print()
print("M2.5-A NUMERICAL RANK SENSITIVITY")
print("=" * 80)

print(
    result.to_string(
        index=False,
    )
)

print()
print("D/E GLOBAL SCALING CHECK")
print("-" * 80)

for kind in (
    "joint",
    "fixed_boundary",
    "profiled",
):

    D = pd.read_csv(
        ROOT /
        f"D_response_channels_x10_{kind}_spectrum.csv"
    )["singular_value"].to_numpy()

    E = pd.read_csv(
        ROOT /
        f"E_input_channels_x0p1_{kind}_spectrum.csv"
    )["singular_value"].to_numpy()

    relative_error = np.linalg.norm(
        E - 10.0 * D
    ) / np.linalg.norm(E)

    print(
        f"{kind:16s}: "
        f"relative_error={relative_error:.12e}"
    )

    if relative_error > 1.0e-8:
        raise RuntimeError(
            f"D/E scaling check failed: {kind}"
        )

print()
print("Validation complete.")
