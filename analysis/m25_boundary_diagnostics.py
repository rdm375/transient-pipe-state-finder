#!/usr/bin/env python3
"""M2.5-B3: Diagnose joint operational uncertainty directions."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("benchmarks")
M24 = ROOT / "results-m24"
M25 = ROOT / "results-m25"

NX = 41
NT = 61
DIM = NX + 2 * NT

CHANNELS = (
    "inlet_pressure",
    "inlet_flow",
    "outlet_pressure",
    "outlet_flow",
)

OBJECTIVES = (
    ("outlet_pressure", "pressure", 20),
    ("inlet_flow", "flow", 0),
    ("inventory", "inventory", 0),
)

J = np.loadtxt(
    M24 / "joint_measurement_jacobian.csv",
    delimiter=",",
    skiprows=1,
)

assert J.shape == (4 * NT, DIM)

operational = pd.read_csv(
    M25 / "operational_sensitivity.csv"
)

directions = np.load(
    M25 / "joint_operational_directions.npz"
)

channel_records = []
boundary_records = []
compensation_records = []
time_records = []

print("\nM2.5-B3 BOUNDARY UNCERTAINTY DIAGNOSTICS")
print("=" * 78)

for name, quantity, index in OBJECTIVES:

    key = f"{name}_eps1_rb1"
    d = directions[key]

    dx = d[:NX]
    db = d[NX:]

    rx = J[:, :NX] @ dx
    rb = J[:, NX:] @ db
    residual = rx + rb

    selected = operational[
        (operational.time_s == 600.0)
        & (operational.quantity == quantity)
        & (operational["index"] == index)
    ].sort_values("column")

    assert np.array_equal(
        selected.column.to_numpy(),
        np.arange(DIM),
    )

    g = selected.sensitivity.to_numpy()

    initial_effect = float(g[:NX] @ dx)
    boundary_effect = float(g[NX:] @ db)

    nx = float(np.linalg.norm(rx))
    nb = float(np.linalg.norm(rb))
    nr = float(np.linalg.norm(residual))

    cosine = (
        float(rx @ rb) / (nx * nb)
        if nx > 0 and nb > 0
        else float("nan")
    )

    compensation_records.append({
        "objective": name,
        "initial_response_norm": nx,
        "boundary_response_norm": nb,
        "combined_response_norm": nr,
        "response_cosine": cosine,
        "initial_operational_effect": initial_effect,
        "boundary_operational_effect": boundary_effect,
        "total_operational_effect": initial_effect + boundary_effect,
    })

    matrix = residual.reshape(NT, 4)
    total_energy = float(residual @ residual)

    print(f"\n{name}")
    print("-" * 78)
    print(f"Initial-state parameter norm: {np.linalg.norm(dx):.9g}")
    print(f"Boundary parameter norm:      {np.linalg.norm(db):.9g}")
    print(f"Initial measurement norm:     {nx:.9g}")
    print(f"Boundary measurement norm:    {nb:.9g}")
    print(f"Combined measurement norm:    {nr:.9g}")
    print(f"Response cosine:              {cosine:.9g}")
    print(f"Initial operational effect:   {initial_effect:.12g}")
    print(f"Boundary operational effect:  {boundary_effect:.12g}")
    print(f"Total operational effect:     {initial_effect + boundary_effect:.12g}")

    print("\nMeasurement channel contributions:")

    for j, channel in enumerate(CHANNELS):
        values = matrix[:, j]
        energy = float(values @ values)
        peak = int(np.argmax(np.abs(values)))

        fraction = energy / total_energy

        channel_records.append({
            "objective": name,
            "channel": channel,
            "residual_norm": float(np.linalg.norm(values)),
            "fraction_squared_norm": fraction,
            "peak_time_s": 10 * peak,
            "peak_absolute_residual": float(abs(values[peak])),
        })

        print(
            f"  {channel:18s} "
            f"energy={fraction:9.5%} "
            f"peak={10 * peak:4d}s"
        )

    inlet = db[:NT] * 1.0e4
    outlet = db[NT:] * 1.0

    print("\nBoundary history perturbations:")

    for channel, values, units in (
        ("inlet_pressure", inlet, "Pa"),
        ("outlet_flow", outlet, "kg/s"),
    ):
        peak = int(np.argmax(np.abs(values)))

        boundary_records.append({
            "objective": name,
            "channel": channel,
            "units": units,
            "l2_norm": float(np.linalg.norm(values)),
            "maximum_absolute": float(abs(values[peak])),
            "peak_time_s": 10 * peak,
            "first_sample": float(values[0]),
            "last_sample": float(values[-1]),
        })

        print(
            f"  {channel:18s} "
            f"max={abs(values[peak]):.9g} {units} "
            f"at {10 * peak}s"
        )

    for k in range(NT):
        record = {
            "objective": name,
            "time_s": 10 * k,
            "inlet_pressure_perturbation_Pa": float(inlet[k]),
            "outlet_flow_perturbation_kg_s": float(outlet[k]),
            "measurement_residual_norm": float(
                np.linalg.norm(matrix[k])
            ),
        }

        for j, channel in enumerate(CHANNELS):
            record[f"{channel}_residual"] = float(matrix[k, j])

        time_records.append(record)

pd.DataFrame(channel_records).to_csv(
    M25 / "boundary_diagnostic_channels.csv",
    index=False,
)

pd.DataFrame(boundary_records).to_csv(
    M25 / "boundary_diagnostic_histories.csv",
    index=False,
)

pd.DataFrame(compensation_records).to_csv(
    M25 / "boundary_diagnostic_compensation.csv",
    index=False,
)

pd.DataFrame(time_records).to_csv(
    M25 / "boundary_diagnostic_timeseries.csv",
    index=False,
)

print("\nM2.5-B3 DIAGNOSTICS COMPLETE")
