"""M27 native covariance-weighted Gauss-Newton iteration study.

Independent runs from the same nominal initial state.
No calibration priors or statistical regularization.
"""

from pathlib import Path
import csv
import subprocess

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "benchmarks/results-m27"

EXECUTABLE = (
    ROOT / "build/state_finder_m27_correlated_reconstruction"
)

MEASUREMENTS = (
    RESULTS
    / "correlated_measurement_realization_sigma0p5.csv"
)

ITERATION_LIMITS = (6, 12, 24)

SUMMARY = RESULTS / "native_iteration_sweep.csv"


def parse_output(output):
    result = {}

    for line in output.splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def main():
    if SUMMARY.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing summary: {SUMMARY}"
        )

    if not EXECUTABLE.is_file():
        raise FileNotFoundError(EXECUTABLE)

    if not MEASUREMENTS.is_file():
        raise FileNotFoundError(MEASUREMENTS)

    rows = []

    for iterations in ITERATION_LIMITS:
        label = f"native_gn_{iterations:03d}"

        log_path = RESULTS / f"{label}.log"
        solution_path = RESULTS / f"{label}_solution.csv"

        for path in (log_path, solution_path):
            if path.exists():
                raise FileExistsError(
                    f"Refusing to overwrite: {path}"
                )

        print(
            f"\nRunning M27 native GN: "
            f"max_iterations={iterations}",
            flush=True,
        )

        completed = subprocess.run(
            [
                str(EXECUTABLE),
                str(MEASUREMENTS),
                str(iterations),
                str(solution_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            print(completed.stdout)
            print(completed.stderr)

            raise RuntimeError(
                f"Native reconstruction failed "
                f"for {iterations} iterations"
            )

        metrics = parse_output(completed.stdout)

        solution = pd.read_csv(solution_path)

        if len(solution) != 163:
            raise RuntimeError(
                "Unexpected solution dimension"
            )

        if not np.array_equal(
            solution["coordinate"].to_numpy(),
            np.arange(163),
        ):
            raise RuntimeError(
                "Unexpected coordinate ordering"
            )

        final_objective = float(
            metrics["final_objective"]
        )

        verified_objective = float(
            metrics["verified_final_objective"]
        )

        if not np.isclose(
            final_objective,
            verified_objective,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(
                "Final objective verification failed"
            )

        row = {
            "max_iterations": iterations,
            "iterations": int(metrics["iterations"]),
            "converged": int(metrics["converged"]),
            "initial_objective": float(
                metrics["initial_objective"]
            ),
            "final_objective": final_objective,
            "gradient_norm": float(
                metrics["gradient_norm"]
            ),
            "objective_evaluations": int(
                metrics["objective_evaluations"]
            ),
            "gradient_evaluations": int(
                metrics["gradient_evaluations"]
            ),
            "pcg_iterations": int(
                metrics["pcg_iterations"]
            ),
            "operator_calls": int(
                metrics["operator_calls"]
            ),
            "damping_retries": int(
                metrics["damping_retries"]
            ),
            "scaled_solution_norm": float(
                metrics["scaled_solution_norm"]
            ),
            "terminal_inventory_true_kg": float(
                metrics["terminal_inventory_true_kg"]
            ),
            "terminal_inventory_estimated_kg": float(
                metrics["terminal_inventory_estimated_kg"]
            ),
            "terminal_inventory_error_kg": float(
                metrics["terminal_inventory_error_kg"]
            ),
            "terminal_inventory_absolute_error_kg": float(
                metrics["terminal_inventory_absolute_error_kg"]
            ),
            "terminal_inventory_relative_error": float(
                metrics["terminal_inventory_relative_error"]
            ),
            "terminal_inventory_relative_error_pct": float(
                metrics["terminal_inventory_relative_error_pct"]
            ),
            "terminal_inventory_absolute_relative_error_pct": float(
                metrics[
                    "terminal_inventory_absolute_relative_error_pct"
                ]
            ),
            "minimum_pressure_pa": float(
                metrics["minimum_pressure_pa"]
            ),
            "maximum_mass_defect_kg": float(
                metrics["maximum_mass_defect_kg"]
            ),
            "elapsed_seconds": float(
                metrics["elapsed_seconds"]
            ),
            "reason": metrics["reason"],
            "solution_file": solution_path.name,
            "log_file": log_path.name,
        }

        log_path.write_text(
            completed.stdout + completed.stderr
        )

        rows.append(row)

        print(
            f"  objective:       "
            f"{row['final_objective']:.9f}"
        )
        print(
            f"  gradient norm:   "
            f"{row['gradient_norm']:.6e}"
        )
        print(
            f"  inventory error: "
            f"{row['terminal_inventory_error_kg']:.6f} kg"
        )
        print(
            f"  PCG iterations:  "
            f"{row['pcg_iterations']}"
        )
        print(
            f"  elapsed:         "
            f"{row['elapsed_seconds']:.3f} s"
        )
        print(
            f"  converged:       "
            f"{bool(row['converged'])}"
        )

    with SUMMARY.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nM27 native iteration sweep complete")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved summary: {SUMMARY}")


if __name__ == "__main__":
    main()
