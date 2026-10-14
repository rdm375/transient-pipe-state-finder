#!/usr/bin/env python3
"""M2.4-E: Inventory reliability and conservation diagnostics."""

from pathlib import Path
import numpy as np
import pandas as pd

source = Path("benchmarks/results-m24/mode_decay.csv")
output = Path("benchmarks/results-m24/inventory_reliability.csv")

df = pd.read_csv(source)

required = [
    "inventory_error_kg",
    "inventory_change_error_kg",
    "mass_conservation_defect_kg",
]

for column in required:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")
    if not np.isfinite(df[column]).all():
        raise ValueError(f"Nonfinite values: {column}")

tolerances = [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01]

records = []

print("\nM2.4-E INVENTORY RELIABILITY")
print("=" * 78)

for mode, group in df.groupby("mode"):
    group = group.sort_values("time_s")

    time = group["time_s"].to_numpy()
    error = group["inventory_error_kg"].to_numpy()
    change = group["inventory_change_error_kg"].to_numpy()
    defect = group["mass_conservation_defect_kg"].to_numpy()

    absolute_error = np.abs(error)

    future_max = np.maximum.accumulate(
        absolute_error[::-1]
    )[::-1]

    print(f"\nMode {mode}")
    print(f"  Initial inventory error: {error[0]:.9g} kg")
    print(f"  Final inventory error:   {error[-1]:.9g} kg")
    print(f"  Peak inventory error:    {absolute_error.max():.9g} kg")
    print(f"  Inventory-change error:  {change[-1]:.9g} kg")
    print(f"  Maximum mass defect:     {np.abs(defect).max():.9g} kg/step")

    for tolerance in tolerances:
        valid = np.flatnonzero(future_max <= tolerance)

        settling_time = (
            float(time[valid[0]])
            if valid.size
            else np.nan
        )

        records.append({
            "mode": mode,
            "tolerance_kg": tolerance,
            "settling_time_s": settling_time,
            "initial_inventory_error_kg": error[0],
            "final_inventory_error_kg": error[-1],
            "peak_inventory_error_kg": absolute_error.max(),
            "final_inventory_change_error_kg": change[-1],
            "max_mass_defect_kg_per_step": np.abs(defect).max(),
        })

result = pd.DataFrame(records)
result.to_csv(output, index=False)

print("\nINVENTORY SETTLING TIMES (seconds)")
print("=" * 78)

table = result.pivot(
    index="mode",
    columns="tolerance_kg",
    values="settling_time_s"
)

print(table.to_string())

print("\nCONSERVATION SUMMARY")
print("=" * 78)

max_defect = np.abs(
    df["mass_conservation_defect_kg"]
).max()

print(f"Maximum defect: {max_defect:.12e} kg/step")

print("\nSaved:", output)
