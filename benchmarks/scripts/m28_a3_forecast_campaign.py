#!/usr/bin/env python3
"""M28-A3: statistical forecast validation using saved M27 solutions."""

import csv
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXECUTABLE = ROOT / "build/state_finder_m28_forecast"

INPUT = ROOT / "benchmarks/results-m27/statistical-c"
OUTPUT = ROOT / "benchmarks/results-m28/forecast-a3"

HORIZONS = (600, 1200, 1800)
SCENARIOS = ("c1", "c2")
REPLICATES = range(32)


def read_terminal(path):
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        first = next(rows)
        last = first

        for row in rows:
            last = row

    return first, last


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not EXECUTABLE.is_file():
        raise RuntimeError(f"Missing executable: {EXECUTABLE}")

    records = []

    for scenario in SCENARIOS:
        for replicate in REPLICATES:
            name = f"r{replicate:03d}_original"

            solution = (
                INPUT / scenario / "solutions" / f"{name}.csv"
            )

            if not solution.is_file():
                raise FileNotFoundError(solution)

            directory = OUTPUT / scenario
            directory.mkdir(parents=True, exist_ok=True)

            prefix = directory / f"{name}.csv"

            completed = subprocess.run(
                [str(EXECUTABLE), str(solution), str(prefix)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            (directory / f"{name}.log").write_text(
                completed.stdout + completed.stderr
            )

            for horizon in HORIZONS:
                result = directory / f"{name}_{horizon}s.csv"

                first, last = read_terminal(result)

                initial_error = float(
                    first["inventory_error_kg"]
                )

                final_error = float(
                    last["inventory_error_kg"]
                )

                amplification = (
                    abs(final_error) / abs(initial_error)
                    if abs(initial_error) > 1.0e-8
                    else math.nan
                )

                records.append(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "horizon_s": horizon,
                        "initial_inventory_error_kg":
                            initial_error,
                        "final_inventory_error_kg":
                            final_error,
                        "absolute_initial_error_kg":
                            abs(initial_error),
                        "absolute_final_error_kg":
                            abs(final_error),
                        "amplification": amplification,
                        "terminal_max_pressure_error_pa":
                            float(
                                last["maximum_pressure_error_pa"]
                            ),
                        "terminal_reference_mass_defect_kg":
                            float(
                                last["reference_mass_defect_kg"]
                            ),
                        "terminal_estimated_mass_defect_kg":
                            float(
                                last["estimated_mass_defect_kg"]
                            ),
                    }
                )

            print(f"Completed {scenario}/{name}", flush=True)

    summary = OUTPUT / "forecast_summary.csv"

    with summary.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(records[0])
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"\nWrote {len(records)} forecast records")
    print(summary)


if __name__ == "__main__":
    main()
