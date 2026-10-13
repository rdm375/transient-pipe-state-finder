#!/usr/bin/env python3
"""Plot M1.7c dominant predictive mode."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("benchmarks/results-m17a")

data = np.genfromtxt(
    ROOT / "dominant_predictive_mode.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8",
)

pressure = data[
    data["component"] == "pressure"
]

flow = data[
    data["component"] == "interior_flow"
]

inlet = data[
    data["component"] == "inlet_flow"
]

x_pressure = pressure["position_m"] / 1000.0

# Flow coordinate placement is provisional until
# verified against the C++ state ordering.
x_flow = np.arange(100)

plt.figure(figsize=(10, 5))

plt.plot(
    x_pressure,
    pressure["physical_perturbation"],
    marker=".",
)

plt.xlabel("Distance along pipeline (km)")
plt.ylabel("Initial pressure perturbation (Pa)")
plt.title("Dominant Predictive Mode: Pressure Component")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    ROOT / "dominant_mode_pressure.png",
    dpi=180,
)

plt.close()

plt.figure(figsize=(10, 5))

plt.plot(
    x_flow,
    flow["physical_perturbation"],
    marker=".",
    label="Flow components",
)

plt.axhline(
    inlet["physical_perturbation"][0],
    linestyle="--",
    label="Initial inlet flow component",
)

plt.xlabel("Flow component index")
plt.ylabel("Initial mass-flow perturbation (kg/s)")
plt.title("Dominant Predictive Mode: Flow Component")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    ROOT / "dominant_mode_flow.png",
    dpi=180,
)

plt.close()

print("PASS M1.7c predictive-mode plots")
