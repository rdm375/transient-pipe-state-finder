#!/usr/bin/env python3
"""M1.7a: Time-resolved sensitivity and information accumulation."""

from pathlib import Path
import csv

import numpy as np

ROOT = Path("benchmarks/results-m17a")

H = np.loadtxt(ROOT / "H_800.csv", delimiter=",")

assert H.shape == (1602, 201)

prior = np.r_[
    np.full(100, 10000.0 / 1.0e5),
    np.full(101, 1.0 / 10.0),
]

obs = np.empty(H.shape[0])
obs[0::2] = 0.01
obs[1::2] = 10.0 / 1.0e4

K = H * prior[np.newaxis, :] / obs[:, np.newaxis]

# Per-time-step prior-whitened sensitivity magnitude.
K_time = K.reshape(801, 2, 201)

sensitivity_norm = np.linalg.norm(
    K_time, axis=(1, 2)
)

# Compute information dimension at selected horizons.
horizons = (
    0, 1, 2, 5, 10, 20, 40, 60,
    80, 100, 150, 200, 400, 800
)

rows = []

for horizon in horizons:
    Kh = K[:2 * (horizon + 1)]

    singular_values = np.linalg.svd(
        Kh, compute_uv=False
    )

    lam = singular_values**2

    effective_dimension = np.sum(
        lam / (1.0 + lam)
    )

    rows.append({
        "step": horizon,
        "time_s": 15.0 * horizon,
        "time_min": 0.25 * horizon,
        "effective_dimension": effective_dimension,
        "sensitivity_norm": sensitivity_norm[horizon],
    })

    print(
        f"step={horizon:3d} "
        f"time={0.25*horizon:7.2f} min "
        f"d_eff={effective_dimension:12.6f} "
        f"sensitivity={sensitivity_norm[horizon]:12.6e}"
    )

output = ROOT / "information_decay.csv"

with output.open("w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

# Save full time-resolved sensitivity norms.
norm_output = ROOT / "sensitivity_norms.csv"

with norm_output.open("w", newline="") as file:
    writer = csv.writer(
        file,
        lineterminator="\n",
    )
    writer.writerow(["step", "time_s", "sensitivity_norm"])

    for k, value in enumerate(sensitivity_norm):
        writer.writerow([k, 15.0*k, value])

print()
print("PASS M1.7a information-decay analysis")
