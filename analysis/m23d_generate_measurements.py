#!/usr/bin/env python3

import csv
import hashlib
import json
import math
import random
from pathlib import Path

REFERENCE = Path(
    "benchmarks/results-m23a/reference_boundary.csv"
)

OUTPUT = Path("benchmarks/results-m23d")
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 20261013

SIGMA = {
    "inlet_pressure": 1000.0,
    "inlet_flow": 0.1,
    "outlet_pressure": 1000.0,
    "outlet_flow": 0.1,
}

CHANNELS = [
    ("inlet_pressure", "inlet_pressure_Pa", "Pa"),
    ("inlet_flow", "inlet_flow_kg_s", "kg_s"),
    ("outlet_pressure", "outlet_pressure_Pa", "Pa"),
    ("outlet_flow", "outlet_flow_kg_s", "kg_s"),
]

rng = random.Random(SEED)

with REFERENCE.open(newline="") as f:
    reference = list(csv.DictReader(f))

assert len(reference) == 61

fields = ["time_s"]

for name, _, unit in CHANNELS:
    fields.extend([
        f"{name}_truth_{unit}",
        f"{name}_observed_{unit}",
        f"{name}_noise_{unit}",
    ])

measurements = []

for k, sample in enumerate(reference):
    time = float(sample["time_s"])

    assert math.isclose(
        time, 10.0*k, abs_tol=1e-9
    )

    row = {"time_s": time}

    for name, reference_column, unit in CHANNELS:
        truth = float(sample[reference_column])

        noise = rng.gauss(0.0, SIGMA[name])

        row[f"{name}_truth_{unit}"] = truth
        row[f"{name}_observed_{unit}"] = truth + noise
        row[f"{name}_noise_{unit}"] = noise

    measurements.append(row)

csv_path = OUTPUT / "measurements.csv"

with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(measurements)

reference_sha256 = hashlib.sha256(
    REFERENCE.read_bytes()
).hexdigest()

metadata = {
    "experiment": "M2.3-D",
    "description": "Four-channel noisy boundary measurements",
    "reference": str(REFERENCE),
    "reference_sha256": reference_sha256,
    "sample_count": len(measurements),
    "sampling_interval_s": 10.0,
    "duration_s": 600.0,
    "random_seed": SEED,
    "random_engine": "Python random.Random (MT19937)",
    "normal_distribution": "Python random.gauss",
    "noise_model": "Independent Gaussian",
    "noise_standard_deviations": SIGMA,
    "channels": [name for name, _, _ in CHANNELS],
}

metadata_path = OUTPUT / "measurement_metadata.json"

metadata_path.write_text(
    json.dumps(metadata, indent=2) + "\n"
)

for name, _, unit in CHANNELS:
    values = [
        row[f"{name}_noise_{unit}"]
        for row in measurements
    ]

    mean = sum(values) / len(values)

    rms = math.sqrt(
        sum(v*v for v in values) / len(values)
    )

    print(
        f"{name}: "
        f"sigma={SIGMA[name]:.8g}, "
        f"mean={mean:.8g}, "
        f"rms={rms:.8g}"
    )

print(f"samples={len(measurements)}")
print(f"reference_sha256={reference_sha256}")
print("PASS M2.3-D four-channel measurement generation")
