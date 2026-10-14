#!/usr/bin/env python3
"""M2.4-F2C nonlinear adversarial validation summary."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("benchmarks/results-m24")

SOURCE = ROOT / "adversarial_nonlinear_trajectories.csv"
OUTPUT = ROOT / "adversarial_validation_summary.csv"

df = pd.read_csv(SOURCE)

records = []

for (quantity, amplitude), group in df.groupby(
    ["quantity", "amplitude"],
    sort=True,
):
    group = group.sort_values("time_s")
    final = group.iloc[-1]

    pressure_error = group[
        "pressure_linearization_error_linf_Pa"
    ].max()

    flow_error = group[
        "flow_linearization_error_linf_kg_s"
    ].max()

    inventory_error = group[
        "inventory_linearization_error_kg"
    ].max()

    pressure_scale = max(
        group["pressure_linf_linear_Pa"].max(),
        1e-12,
    )

    flow_scale = max(
        group["flow_linf_linear_kg_s"].max(),
        1e-12,
    )

    inventory_scale = max(
        group["inventory_linear_kg"].abs().max(),
        1e-12,
    )

    record = {
        "quantity": quantity,
        "amplitude": amplitude,
        "pressure_relative_error": (
            pressure_error/pressure_scale
        ),
        "flow_relative_error": (
            flow_error/flow_scale
        ),
        "inventory_relative_error": (
            inventory_error/inventory_scale
        ),
        "final_pressure_nonlinear_Pa": (
            final.pressure_linf_nonlinear_Pa
        ),
        "final_pressure_linear_Pa": (
            final.pressure_linf_linear_Pa
        ),
        "final_flow_nonlinear_kg_s": (
            final.flow_linf_nonlinear_kg_s
        ),
        "final_flow_linear_kg_s": (
            final.flow_linf_linear_kg_s
        ),
        "final_inventory_nonlinear_kg": (
            final.inventory_nonlinear_kg
        ),
        "final_inventory_linear_kg": (
            final.inventory_linear_kg
        ),
        "measurement_norm_nonlinear": (
            final.measurement_norm_nonlinear
        ),
        "measurement_norm_linear": (
            final.measurement_norm_linear
        ),
        "max_conservation_defect_kg": (
            group.conservation_defect_kg.abs().max()
        ),
        "minimum_pressure_Pa": (
            group.minimum_pressure_Pa.min()
        ),
    }

    records.append(record)

summary = pd.DataFrame(records)

summary.to_csv(OUTPUT, index=False)

print()
print("M2.4-F2C NONLINEAR ADVERSARIAL VALIDATION")
print("=" * 90)

for quantity, group in summary.groupby("quantity"):
    print()
    print(quantity.upper())
    print("-" * 90)

    print(
        group[
            [
                "amplitude",
                "pressure_relative_error",
                "flow_relative_error",
                "inventory_relative_error",
                "measurement_norm_nonlinear",
                "measurement_norm_linear",
                "max_conservation_defect_kg",
            ]
        ].to_string(index=False)
    )

print()
print("MAXIMUM CONSERVATION DEFECT")
print(
    summary.max_conservation_defect_kg.max()
)

print()
print("MINIMUM PHYSICAL PRESSURE")
print(
    summary.minimum_pressure_Pa.min()
)

print()
print("Saved:", OUTPUT)

print()
print("Interpretation:")
print("  Relative linearization errors should decrease")
print("  approximately proportionally to amplitude.")
print("  The alpha=1 cases test finite-amplitude accuracy.")
print("  Measurement norms test nonlinear consistency.")
print("  Conservation defects test discrete mass balance.")

print()
print("M2.4-F2C MEASUREMENT NORMALIZATION CHECK")
print("=" * 72)

full_amplitude = summary[
    np.isclose(summary.amplitude, 1.0)
]

if len(full_amplitude) != 3:
    raise RuntimeError(
        "Expected three full-amplitude adversarial experiments"
    )

for row in full_amplitude.itertuples():
    discrepancy = abs(
        row.measurement_norm_linear - 1.0
    )

    print(
        f"{row.quantity:12s} "
        f"linear={row.measurement_norm_linear:.9g} "
        f"nonlinear={row.measurement_norm_nonlinear:.9g} "
        f"linear_constraint_error={discrepancy:.3e}"
    )

    if discrepancy > 1.0e-5:
        raise RuntimeError(
            f"Measurement normalization mismatch: {row.quantity}"
        )

print()
print("PASS: Tangent measurement norms match F2B constraints.")
