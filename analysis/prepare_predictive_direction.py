#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np

source = Path(
    "benchmarks/results-m17a/"
    "dominant_predictive_mode.csv"
)

destination = Path(
    "benchmarks/results-m17e/"
    "predictive_direction.txt"
)

with source.open(newline="") as f:
    rows = list(csv.DictReader(f))

indices = np.array([
    int(row["state_index"])
    for row in rows
])

v = np.array([
    float(row["standardized_mode"])
    for row in rows
])

assert v.shape == (201,)
assert np.array_equal(indices, np.arange(201))
assert np.all(np.isfinite(v))

v /= np.linalg.norm(v)

prior_sigma = np.r_[
    np.full(100, 1e4),
    np.full(101, 1.0)
]

state_scale = np.r_[
    np.full(100, 1e5),
    np.full(101, 10.0)
]

direction = (
    prior_sigma / state_scale
) * v

destination.parent.mkdir(
    parents=True,
    exist_ok=True
)

np.savetxt(
    destination,
    direction,
    fmt="%.17e"
)

print(f"dimension={len(direction)}")
print(f"predictive_mode_norm={np.linalg.norm(v):.17e}")
print(f"direction_norm={np.linalg.norm(direction):.17e}")
print(f"output={destination}")
print("PASS M1.7h direction preparation")
