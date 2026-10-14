#!/usr/bin/env python3
"""M2.5-C1: Nonlinear replay of three joint uncertainty directions."""

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path("benchmarks/results-m25")

EXECUTABLE = Path(
    "build/state_finder_m25_joint_nonlinear_replay"
)

CASES = (
    ("outlet_pressure", "pressure", 20),
    ("inlet_flow", "flow", 0),
    ("inventory", "inventory", 0),
)

EPSILON = 1.0
INITIAL_RADIUS = 1.0
BOUNDARY_RADIUS = 1.0
NX = 41
DIM = 163

directions = np.load(
    ROOT / "joint_operational_directions.npz"
)

sensitivities = pd.read_csv(
    ROOT / "operational_sensitivity.csv"
)

def gradient(quantity, index):
    rows = sensitivities[
        (sensitivities.time_s == 600.0)
        & (sensitivities.quantity == quantity)
        & (sensitivities["index"] == index)
    ].sort_values("column")

    if not np.array_equal(
        rows.column.to_numpy(),
        np.arange(DIM)
    ):
        raise RuntimeError("Incomplete gradient")

    return rows.sensitivity.to_numpy(dtype=float)


process = subprocess.Popen(
    [str(EXECUTABLE)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

header = process.stdout.readline().strip()

if header != "READY 163":
    raise RuntimeError(
        f"Unexpected oracle header: {header}"
    )

records = []

for name, quantity, index in CASES:
    key = f"{name}_eps1_rb1"

    direction = directions[key]

    line = ",".join(
        format(float(x), ".17g")
        for x in direction
    )

    process.stdin.write(line + "\n")
    process.stdin.flush()

    response = process.stdout.readline().strip()

    if not response.startswith("OK,"):
        raise RuntimeError(
            f"{name}: nonlinear replay failed: {response}"
        )

    values = [
        float(x)
        for x in response.split(",")[1:]
    ]

    (
        nonlinear_pressure,
        nonlinear_flow,
        nonlinear_inventory,
        nonlinear_measurement_norm,
        min_pressure,
        max_mass_defect,
        compatibility,
    ) = values

    nonlinear_values = {
        "outlet_pressure": nonlinear_pressure,
        "inlet_flow": nonlinear_flow,
        "inventory": nonlinear_inventory,
    }

    linear = float(
        gradient(quantity,index) @ direction
    )

    nonlinear = nonlinear_values[name]

    records.append({
        "objective": name,
        "linear_prediction": linear,
        "nonlinear_deviation": nonlinear,
        "absolute_difference": nonlinear-linear,
        "relative_difference": (
            (nonlinear-linear)/linear
            if linear != 0.0
            else float("nan")
        ),
        "nonlinear_measurement_norm":
            nonlinear_measurement_norm,
        "initial_radius":
            float(np.linalg.norm(direction[:NX])),
        "boundary_radius":
            float(np.linalg.norm(direction[NX:])),
        "minimum_pressure_Pa":
            min_pressure,
        "maximum_mass_defect_kg":
            max_mass_defect,
        "initial_compatibility_Pa":
            compatibility,
        "strict_measurement_feasible":
            nonlinear_measurement_norm <= EPSILON,
    })

process.stdin.write("QUIT\n")
process.stdin.flush()
process.stdin.close()

process.wait(timeout=30)

if process.returncode != 0:
    raise RuntimeError(
        process.stderr.read()
    )

df = pd.DataFrame(records)

print()
print("M2.5-C1 NONLINEAR JOINT-DIRECTION REPLAY")
print("=" * 100)

print(df.to_string(index=False))

output = ROOT / "nonlinear_joint_replay.csv"

df.to_csv(output,index=False)

print()
print("Saved:",output)
