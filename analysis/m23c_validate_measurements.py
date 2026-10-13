#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path

root = Path("benchmarks/results-m23c")

with (root / "measurement_metadata.json").open() as f:
    metadata = json.load(f)

with (root / "measurements.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == metadata["sample_count"] == 61

sigma_flow = metadata["inlet_flow_sigma_kg_s"]
sigma_pressure = metadata["outlet_pressure_sigma_Pa"]

flow_noise = []
pressure_noise = []

for k, row in enumerate(rows):
    time = float(row["time_s"])

    assert math.isclose(
        time, 10.0*k, abs_tol=1e-10
    )

    flow_truth = float(row["inlet_flow_truth_kg_s"])
    flow_observed = float(row["inlet_flow_observed_kg_s"])

    pressure_truth = float(row["outlet_pressure_truth_Pa"])
    pressure_observed = float(row["outlet_pressure_observed_Pa"])

    df = float(row["inlet_flow_noise_kg_s"])
    dp = float(row["outlet_pressure_noise_Pa"])

    assert math.isclose(
        flow_observed-flow_truth,
        df,
        abs_tol=1e-12
    )

    assert math.isclose(
        pressure_observed-pressure_truth,
        dp,
        abs_tol=1e-8
    )

    flow_noise.append(df)
    pressure_noise.append(dp)


def statistics(values):
    count = len(values)
    mean = sum(values)/count
    rms = math.sqrt(
        sum(value*value for value in values)/count
    )
    return mean, rms


flow_mean, flow_rms = statistics(flow_noise)
pressure_mean, pressure_rms = statistics(pressure_noise)

print("M2.3-C measurement validation")
print(f"samples={len(rows)}")
print(f"flow_mean_kg_s={flow_mean:.12g}")
print(f"flow_rms_kg_s={flow_rms:.12g}")
print(f"pressure_mean_Pa={pressure_mean:.12g}")
print(f"pressure_rms_Pa={pressure_rms:.12g}")

# These are deliberately broad statistical sanity bounds,
# not exact expectations for a finite Gaussian sample.
assert flow_rms > 0.5*sigma_flow
assert flow_rms < 1.5*sigma_flow

assert pressure_rms > 0.5*sigma_pressure
assert pressure_rms < 1.5*sigma_pressure

print("PASS M2.3-C measurement validation")
