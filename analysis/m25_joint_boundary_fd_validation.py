#!/usr/bin/env python3
"""M2.5-C1: Independent nonlinear finite-difference validation."""

from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path("benchmarks/results-m25")
EXE = Path("build/state_finder_m25_joint_nonlinear_replay")

NX = 41
STEPS = 60
DIM = 163
NOBS = 244

STEP_SIZES = (1e-2, 1e-3, 1e-4)

PIN_START = 41
MOUT_START = 102

# Existing M24 Jacobian has a header:
# x0,x1,...,x162
jacobian_df = pd.read_csv(
    "benchmarks/results-m24/joint_measurement_jacobian.csv"
)

assert jacobian_df.columns.tolist() == [
    f"x{i}" for i in range(DIM)
]

J = jacobian_df.to_numpy(dtype=float)

assert J.shape == (NOBS, DIM)
assert np.all(np.isfinite(J))

# Verify direct prescribed-boundary derivatives.
for k in range(STEPS + 1):
    assert np.isclose(
        J[4*k, PIN_START+k], 1.0, atol=1e-12
    )
    assert np.isclose(
        J[4*k+3, MOUT_START+k], 1.0, atol=1e-12
    )

print("Jacobian shape and boundary ordering PASS")

# Terminal operational sensitivities.
sens = pd.read_csv(
    ROOT / "operational_sensitivity.csv"
)

def gradient(quantity, index):
    rows = sens[
        (sens.time_s == 600.0)
        & (sens.quantity == quantity)
        & (sens["index"] == index)
    ].sort_values("column")

    columns = rows.column.to_numpy(dtype=int)

    assert np.array_equal(
        columns, np.arange(DIM)
    )

    return rows.sensitivity.to_numpy(dtype=float)

G = np.vstack([
    gradient("pressure", 20),
    gradient("flow", 0),
    gradient("inventory", 0),
])

# Eight independent test directions.
directions = {}

for name, column in [
    ("pin_t0", 41),
    ("pin_t300", 71),
    ("pin_t600", 101),
    ("mout_t0", 102),
    ("mout_t300", 132),
    ("mout_t600", 162),
]:
    v = np.zeros(DIM)
    v[column] = 1.0
    directions[name] = v

rng = np.random.default_rng(20261014)

v = np.zeros(DIM)
v[NX:] = rng.standard_normal(DIM-NX)
v /= np.linalg.norm(v)
directions["mixed_boundary"] = v

v = rng.standard_normal(DIM)
v /= np.linalg.norm(v)
directions["mixed_joint"] = v

# Persistent nonlinear forward-model oracle.
proc = subprocess.Popen(
    [str(EXE)],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

header = proc.stdout.readline().strip()

if header != "READY 163":
    raise RuntimeError(
        f"Unexpected oracle header: {header}"
    )

def query(direction, measurement=False):
    prefix = "MEASURE," if measurement else ""

    payload = prefix + ",".join(
        format(float(x), ".17g")
        for x in direction
    )

    proc.stdin.write(payload + "\n")
    proc.stdin.flush()

    response = proc.stdout.readline().strip()

    expected = "MEASURE," if measurement else "OK,"

    if not response.startswith(expected):
        raise RuntimeError(
            f"Nonlinear oracle error: {response}"
        )

    values = np.fromstring(
        response[len(expected):],
        sep=","
    )

    expected_size = NOBS if measurement else 7

    if values.size != expected_size:
        raise RuntimeError(
            f"Unexpected response length: {values.size}"
        )

    if not np.all(np.isfinite(values)):
        raise RuntimeError(
            "Nonfinite nonlinear response"
        )

    return values

records = []

try:
    for name, direction in directions.items():
        tangent_measurements = J @ direction
        tangent_operations = G @ direction

        for h in STEP_SIZES:
            plus = h*direction
            minus = -h*direction

            y_plus = query(plus, measurement=True)
            y_minus = query(minus, measurement=True)

            fd_measurements = (
                y_plus-y_minus
            )/(2*h)

            op_plus = query(plus)
            op_minus = query(minus)

            fd_operations = (
                op_plus[:3]-op_minus[:3]
            )/(2*h)

            measurement_error = (
                fd_measurements-tangent_measurements
            )

            operation_error = (
                fd_operations-tangent_operations
            )

            records.append({
                "direction": name,
                "h": h,
                "measurement_max_abs_error":
                    float(np.max(np.abs(measurement_error))),
                "measurement_relative_l2_error":
                    float(
                        np.linalg.norm(measurement_error)
                        /max(
                            np.linalg.norm(tangent_measurements),
                            1e-14
                        )
                    ),
                "operational_max_abs_error":
                    float(np.max(np.abs(operation_error))),
                "operational_relative_l2_error":
                    float(
                        np.linalg.norm(operation_error)
                        /max(
                            np.linalg.norm(tangent_operations),
                            1e-14
                        )
                    ),
                "pressure_abs_error":
                    float(abs(operation_error[0])),
                "inlet_flow_abs_error":
                    float(abs(operation_error[1])),
                "inventory_abs_error":
                    float(abs(operation_error[2])),
            })

        print(f"Completed: {name}", flush=True)

finally:
    if proc.poll() is None:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()

    proc.wait(timeout=30)

    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.read()
        )

df = pd.DataFrame(records)

output = ROOT / "joint_boundary_fd_validation.csv"

df.to_csv(output, index=False)

print()
print("M2.5-C1 INDEPENDENT NONLINEAR DERIVATIVE VALIDATION")
print("=" * 90)

print(df.to_string(index=False))

print()
print("Best measurement relative errors:")
print(
    df.groupby("direction")[
        "measurement_relative_l2_error"
    ].min().to_string()
)

print()
print("Best operational relative errors:")
print(
    df.groupby("direction")[
        "operational_relative_l2_error"
    ].min().to_string()
)

print()
print("Saved:", output)
