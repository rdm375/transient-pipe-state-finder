#!/usr/bin/env python3
"""M2.4: Settling times for weakly observable initial-state modes."""

from pathlib import Path
import numpy as np
import pandas as pd

source = Path("benchmarks/results-m24/mode_decay.csv")
output = Path("benchmarks/results-m24/reliability_horizon.csv")

df = pd.read_csv(source)

# Illustrative engineering tolerances, not yet validated
# operational acceptance criteria.
tolerances = {
    "pressure_linf_Pa": [
        1000.0,
        100.0,
        10.0,
    ],
    "flow_linf_kg_s": [
        0.1,
        0.01,
        0.001,
    ],
}

records = []

for mode, group in df.groupby("mode"):
    group = group.sort_values("time_s")

    time = group["time_s"].to_numpy()

    for quantity, limits in tolerances.items():
        error = group[quantity].to_numpy()

        # Maximum error over the remaining simulation.
        future_max = np.maximum.accumulate(error[::-1])[::-1]

        for tolerance in limits:
            valid = np.flatnonzero(future_max <= tolerance)

            settling_time = (
                float(time[valid[0]])
                if valid.size
                else np.nan
            )

            records.append({
                "mode": mode,
                "quantity": quantity,
                "tolerance": tolerance,
                "settling_time_s": settling_time,
                "initial_error": error[0],
                "final_error": error[-1],
                "peak_error": np.max(error),
                "peak_time_s": time[np.argmax(error)],
            })

result = pd.DataFrame(records)
result.to_csv(output, index=False)

print("\nM2.4 OPERATIONAL SETTLING TIMES")
print("=" * 75)

for quantity in tolerances:
    print(f"\n{quantity}")
    subset = result[result.quantity == quantity]

    table = subset.pivot(
        index="mode",
        columns="tolerance",
        values="settling_time_s"
    )

    print(table.to_string())

print("\nPeak amplification diagnostics")
print("-" * 75)

for mode, group in df.groupby("mode"):
    print(f"\nMode {mode}")

    for quantity in tolerances:
        values = group[quantity].to_numpy()

        ratio = (
            values.max() / values[0]
            if values[0] > 0
            else np.nan
        )

        print(
            f"  {quantity}: "
            f"peak/initial={ratio:.6g}"
        )

print(f"\nSaved: {output}")
print(
    "NOTE: Settling times are established only within "
    "the 600-second simulation window."
)
