#!/usr/bin/env python3
"""M2.3-F: Exact four-channel reconstruction and linepack validation."""

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path


CHANNELS = (
    ("inlet_pressure", "Pa"),
    ("inlet_flow", "kg_s"),
    ("outlet_pressure", "Pa"),
    ("outlet_flow", "kg_s"),
)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_exact_measurements(source, destination):
    rows = read_csv(source)
    assert len(rows) == 61, f"Expected 61 samples, got {len(rows)}"

    for row in rows:
        for channel, unit in CHANNELS:
            truth = f"{channel}_truth_{unit}"
            observed = f"{channel}_observed_{unit}"
            noise = f"{channel}_noise_{unit}"

            assert truth in row
            assert observed in row
            assert noise in row

            row[observed] = row[truth]
            row[noise] = "0"

    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Confirm all four histories are exactly copied.
    check = read_csv(destination)

    for row in check:
        for channel, unit in CHANNELS:
            assert (
                float(row[f"{channel}_observed_{unit}"])
                == float(row[f"{channel}_truth_{unit}"])
            )

    print(f"PASS: {len(rows) * 4} exact measurements")
    return rows


def rmse(a, b):
    assert len(a) == len(b) and a
    return math.sqrt(
        sum((x - y)**2 for x, y in zip(a, b)) / len(a)
    )


def compare_boundary(rows, reconstructed):
    assert len(rows) == len(reconstructed)

    result = {}

    for channel, unit in CHANNELS:
        truth = [
            float(row[f"{channel}_truth_{unit}"])
            for row in rows
        ]

        estimate = [
            float(row[f"{channel}_reconstructed_{unit}"])
            for row in reconstructed
        ]

        result[f"{channel}_rmse_{unit}"] = rmse(truth, estimate)

    return result


def compare_inventory(reference_path, reconstructed_path):
    reference = read_csv(reference_path)
    reconstructed = read_csv(reconstructed_path)

    assert len(reference) == len(reconstructed)

    print("\nReference inventory columns:")
    print(", ".join(reference[0].keys()))

    # Accept common inventory column names.
    candidates = (
        "linepack_kg",
        "inventory_kg",
        "mass_kg",
    )

    common = [
        key for key in candidates
        if key in reference[0] and key in reconstructed[0]
    ]

    if not common:
        return {
            "status": "reference_inventory_column_requires_mapping",
            "reference_columns": list(reference[0].keys()),
        }

    key = common[0]

    truth = [float(row[key]) for row in reference]
    estimate = [float(row[key]) for row in reconstructed]

    for i, (a, b) in enumerate(zip(reference, reconstructed)):
        assert math.isclose(
            float(a["time_s"]),
            float(b["time_s"]),
            abs_tol=1e-8,
        ), f"Timestamp mismatch at row {i}"

    truth_change = truth[-1] - truth[0]
    estimate_change = estimate[-1] - estimate[0]

    return {
        "status": "compared",
        "inventory_rmse_kg": rmse(truth, estimate),
        "initial_inventory_error_kg": estimate[0] - truth[0],
        "final_inventory_error_kg": estimate[-1] - truth[-1],
        "reference_inventory_change_kg": truth_change,
        "reconstructed_inventory_change_kg": estimate_change,
        "inventory_change_error_kg": estimate_change - truth_change,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executable",
        default="build/state_finder_m23e_reconstruction",
    )
    args = parser.parse_args()

    root = Path("benchmarks/results-m23f")
    root.mkdir(parents=True, exist_ok=True)

    noisy = Path("benchmarks/results-m23d/measurements.csv")
    exact = root / "exact_measurements.csv"
    output = root / "exact-reconstruction"

    rows = write_exact_measurements(noisy, exact)

    command = [
        str(Path(args.executable).resolve()),
        str(exact),
        str(output),
    ]

    print("\nRunning exact four-channel reconstruction...")
    subprocess.run(command, check=True)

    with (output / "reconstruction_summary.json").open() as stream:
        summary = json.load(stream)

    boundary = read_csv(output / "reconstructed_boundary.csv")

    boundary_errors = compare_boundary(rows, boundary)

    reference_inventory = Path(
        "benchmarks/results-m23a/reference_inventory.csv"
    )

    inventory_errors = compare_inventory(
        reference_inventory,
        output / "reconstructed_inventory.csv",
    )

    report = {
        "experiment": "M2.3-F exact four-channel reconstruction",
        "channels": [channel for channel, _ in CHANNELS],
        "measurement_count": 4 * len(rows),
        "optimization": summary,
        "boundary_errors": boundary_errors,
        "inventory_comparison": inventory_errors,
    }

    report_path = root / "validation_report.json"

    with report_path.open("w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print("\nM2.3-F VALIDATION REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print(f"\nSaved: {report_path}")

    assert summary["measurement_count"] == 244
    assert summary["variable_count"] == 163
    assert summary["max_mass_defect_kg_s"] < 1e-5

    print("\nPASS: Four-channel accounting and conservation")
    print(
        "NOTE: Convergence and state-recovery accuracy "
        "must be assessed separately."
    )


if __name__ == "__main__":
    main()
