#!/usr/bin/env python3
"""M28-A4: paired original/calibrated forecast comparison."""

import csv
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXECUTABLE = ROOT / "build/state_finder_m28_forecast"
INPUT = ROOT / "benchmarks/results-m27/statistical-c"
OUTPUT = ROOT / "benchmarks/results-m28/forecast-a4"

SCENARIOS = ("c1", "c2")
ESTIMATORS = ("original", "calibrated")
HORIZONS = (600, 1200, 1800)
REPLICATES = range(32)


def read_history(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise RuntimeError(f"Empty forecast: {path}")

    for row in rows:
        for key, value in row.items():
            if not math.isfinite(float(value)):
                raise RuntimeError(
                    f"Nonfinite {key} in {path}"
                )

    return rows


def rmse(values):
    return math.sqrt(
        statistics.mean(x * x for x in values)
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    if not EXECUTABLE.is_file():
        raise FileNotFoundError(EXECUTABLE)

    records = []

    for scenario in SCENARIOS:
        for replicate in REPLICATES:
            for estimator in ESTIMATORS:
                name = f"r{replicate:03d}_{estimator}"

                solution = (
                    INPUT / scenario / "solutions"
                    / f"{name}.csv"
                )

                if not solution.is_file():
                    raise FileNotFoundError(solution)

                directory = OUTPUT / scenario
                directory.mkdir(parents=True, exist_ok=True)

                prefix = directory / f"{name}.csv"

                # The C++ driver refuses to overwrite outputs.
                # Reuse existing complete results when present.
                expected = [
                    directory / f"{name}_{h}s.csv"
                    for h in HORIZONS
                ]

                if not all(p.is_file() for p in expected):
                    if any(p.exists() for p in expected):
                        raise RuntimeError(
                            f"Partial output exists for {name}; "
                            "inspect before rerunning"
                        )

                    completed = subprocess.run(
                        [
                            str(EXECUTABLE),
                            str(solution),
                            str(prefix),
                        ],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        check=True,
                    )

                    (directory / f"{name}.log").write_text(
                        completed.stdout + completed.stderr
                    )

                for horizon in HORIZONS:
                    path = (
                        directory /
                        f"{name}_{horizon}s.csv"
                    )

                    history = read_history(path)

                    first = history[0]
                    last = history[-1]

                    if abs(
                        float(last["forecast_time_s"]) - horizon
                    ) > 1.0e-9:
                        raise RuntimeError(
                            f"Incorrect forecast horizon: {path}"
                        )

                    initial_error = float(
                        first["inventory_error_kg"]
                    )

                    final_error = float(
                        last["inventory_error_kg"]
                    )

                    maximum_pressure_error = max(
                        float(row["maximum_pressure_error_pa"])
                        for row in history
                    )

                    maximum_mass_defect = max(
                        max(
                            abs(float(
                                row["reference_mass_defect_kg"]
                            )),
                            abs(float(
                                row["estimated_mass_defect_kg"]
                            )),
                        )
                        for row in history
                    )

                    records.append({
                        "scenario": scenario,
                        "replicate": replicate,
                        "estimator": estimator,
                        "horizon_s": horizon,
                        "initial_inventory_error_kg":
                            initial_error,
                        "final_inventory_error_kg":
                            final_error,
                        "absolute_final_inventory_error_kg":
                            abs(final_error),
                        "terminal_max_pressure_error_pa":
                            float(
                                last["maximum_pressure_error_pa"]
                            ),
                        "maximum_pressure_error_pa":
                            maximum_pressure_error,
                        "maximum_mass_defect_kg":
                            maximum_mass_defect,
                    })

                print(
                    f"Completed {scenario}/{name}",
                    flush=True,
                )

    summary_path = OUTPUT / "forecast_summary.csv"

    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(records[0]),
        )
        writer.writeheader()
        writer.writerows(records)

    grouped = defaultdict(dict)

    for row in records:
        key = (
            row["scenario"],
            row["replicate"],
            row["horizon_s"],
        )
        grouped[key][row["estimator"]] = row

    paired = []

    for key, estimators in sorted(grouped.items()):
        if set(estimators) != set(ESTIMATORS):
            raise RuntimeError(
                f"Incomplete estimator pair: {key}"
            )

        original = estimators["original"]
        calibrated = estimators["calibrated"]

        original_error = float(
            original["final_inventory_error_kg"]
        )

        calibrated_error = float(
            calibrated["final_inventory_error_kg"]
        )

        paired.append({
            "scenario": key[0],
            "replicate": key[1],
            "horizon_s": key[2],
            "original_error_kg": original_error,
            "calibrated_error_kg": calibrated_error,
            "absolute_error_difference_kg":
                abs(calibrated_error) - abs(original_error),
            "squared_error_difference_kg2":
                calibrated_error**2 - original_error**2,
            "terminal_pressure_error_difference_pa":
                float(
                    calibrated[
                        "terminal_max_pressure_error_pa"
                    ]
                ) -
                float(
                    original[
                        "terminal_max_pressure_error_pa"
                    ]
                ),
        })

    paired_path = OUTPUT / "paired_comparison.csv"

    with paired_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(paired[0]),
        )
        writer.writeheader()
        writer.writerows(paired)

    print()
    print(
        "scenario horizon n "
        "original_rmse calibrated_rmse "
        "original_mae calibrated_mae "
        "mean_paired_abs_difference "
        "calibrated_improved"
    )

    for scenario in SCENARIOS:
        for horizon in HORIZONS:
            subset = [
                row for row in paired
                if row["scenario"] == scenario
                and row["horizon_s"] == horizon
            ]

            original = [
                row["original_error_kg"]
                for row in subset
            ]

            calibrated = [
                row["calibrated_error_kg"]
                for row in subset
            ]

            differences = [
                row["absolute_error_difference_kg"]
                for row in subset
            ]

            improved = sum(
                difference < 0
                for difference in differences
            )

            print(
                scenario,
                horizon,
                len(subset),
                f"{rmse(original):.6f}",
                f"{rmse(calibrated):.6f}",
                f"{statistics.mean(map(abs, original)):.6f}",
                f"{statistics.mean(map(abs, calibrated)):.6f}",
                f"{statistics.mean(differences):+.6f}",
                improved,
            )

    print()
    print(f"Summary: {summary_path}")
    print(f"Paired comparison: {paired_path}")
    print(f"Total forecast runs: {len(records) // 3}")
    print(f"Total horizon records: {len(records)}")


if __name__ == "__main__":
    main()
