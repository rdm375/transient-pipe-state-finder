#!/usr/bin/env python3
"""M27-C3: nonlinear reconstruction convergence audit."""

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / (
    "benchmarks/results-m27/statistical-c/"
    "reconstruction_results.csv"
)

OUTPUT = ROOT / (
    "benchmarks/results-m27/convergence-c3"
)

ORIGINAL = ROOT / (
    "build/state_finder_m27_correlated_reconstruction"
)

CALIBRATED = ROOT / (
    "build/state_finder_m27_calibrated_reconstruction"
)

ITERATIONS = (48, 96, 192)
REPLICATE = 0

OUTPUT.mkdir(parents=True, exist_ok=True)

with SOURCE.open(newline="") as stream:
    baseline = list(csv.DictReader(stream))

results = []

for scenario in ("c1", "c2"):

    input_file = ROOT / (
        "benchmarks/results-m27/statistical-c/"
        f"{scenario}/inputs/r{REPLICATE:03d}.csv"
    )

    for model in ("original", "calibrated"):

        previous = next(
            row for row in baseline
            if row["scenario"] == scenario
            and row["model"] == model
            and int(row["replicate"]) == REPLICATE
        )

        results.append({
            "scenario": scenario,
            "model": model,
            "iterations_budget": 24,
            "iterations_completed": previous["iterations"],
            "objective": previous["final_objective"],
            "gradient_norm": previous["gradient_norm"],
            "inventory_error_kg":
                previous["terminal_inventory_error_kg"],
            "converged": previous["converged"],
            "reason": previous["reason"],
        })

        for budget in ITERATIONS:

            stem = f"{scenario}_{model}_iter{budget}"

            log = OUTPUT / f"{stem}.log"
            solution = OUTPUT / f"{stem}.csv"

            if model == "original":
                command = [
                    str(ORIGINAL),
                    str(input_file),
                    str(budget),
                    str(solution),
                ]
            else:
                command = [
                    str(CALIBRATED),
                    str(input_file),
                    str(budget),
                    "2.0",
                    "0.0",
                    str(solution),
                ]

            if log.exists() != solution.exists():
                raise RuntimeError(
                    f"Incomplete prior run: {stem}"
                )

            if not log.exists():
                print(
                    f"Running {stem}",
                    flush=True,
                )

                with log.open("x") as stream:
                    completed = subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )

                if completed.returncode != 0:
                    raise RuntimeError(
                        f"Solver failed: {log}"
                    )

                if not solution.exists():
                    raise RuntimeError(
                        f"Missing solution: {solution}"
                    )

            values = {}

            for line in log.read_text().splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    values[key] = value

            results.append({
                "scenario": scenario,
                "model": model,
                "iterations_budget": budget,
                "iterations_completed": values["iterations"],
                "objective": values["final_objective"],
                "gradient_norm": values["gradient_norm"],
                "inventory_error_kg":
                    values["terminal_inventory_error_kg"],
                "converged": values["converged"],
                "reason": values["reason"],
            })

print()
print("M27-C3 CONVERGENCE AUDIT")
print()

print(
    f"{'Case':18s} {'Budget':>7s} "
    f"{'Objective':>13s} {'Gradient':>12s} "
    f"{'Inventory kg':>14s} {'Converged':>10s}"
)

for row in results:

    label = (
        f"{row['scenario']}/"
        f"{row['model']}"
    )

    print(
        f"{label:18s} "
        f"{row['iterations_budget']:>7d} "
        f"{float(row['objective']):13.6f} "
        f"{float(row['gradient_norm']):12.5g} "
        f"{float(row['inventory_error_kg']):+14.6f} "
        f"{str(row['converged']):>10s}"
    )

summary = OUTPUT / "convergence_summary.csv"

if summary.exists():
    raise RuntimeError(
        f"Refusing to overwrite {summary}"
    )

with summary.open("x", newline="") as stream:
    writer = csv.DictWriter(
        stream,
        fieldnames=list(results[0]),
    )
    writer.writeheader()
    writer.writerows(results)

print()
print(f"Saved {summary}")
