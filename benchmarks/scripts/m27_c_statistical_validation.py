#!/usr/bin/env python3
"""M27-C: paired statistical validation of nonlinear reconstruction.

C1: 32 independent temporally correlated noise realizations, zero bias.
C2: The same noise realizations, plus random constant pressure biases.

The C++ drivers interpret their CSV input as normalized measurement
perturbations. In C2 this includes noise AND injected calibration bias.

All results are finite-iteration estimates (24 GN iterations), unless
a solver reports convergence. This script does not change the solvers.
"""

import argparse
import csv
import hashlib
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

N_TIME = 61
N_CHANNELS = 4
DT = 10.0
TAU = 120.0
SIGMA_NOISE = 0.5
SIGMA_CAL_KPA = 2.0
RHO_CAL = 0.0
ITERATIONS = 24

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "benchmarks/results-m27/statistical-c"
BASE_EXE = ROOT / "build/state_finder_m27_correlated_reconstruction"
CAL_EXE = ROOT / "build/state_finder_m27_calibrated_reconstruction"

FIELDS = [
    "scenario", "replicate", "noise_seed", "bias_seed",
    "true_inlet_bias_kpa", "true_outlet_bias_kpa",
    "model", "status", "returncode", "iterations",
    "converged", "reason", "final_objective",
    "gradient_norm", "terminal_inventory_error_kg",
    "maximum_mass_defect_kg", "elapsed_seconds",
    "estimated_inlet_bias_kpa", "estimated_outlet_bias_kpa",
    "noise_sha256", "log_file", "solution_file",
]


def write_csv_new(path, header, rows):
    """Never silently overwrite an existing experiment artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def noise_realization(seed):
    """Stationary AR(1) with covariance sigma^2 exp(-|dt|/tau).

    Each channel has its own independent Gaussian innovation stream.
    The four channels are interleaved by time.
    """
    rng = random.Random(seed)
    phi = math.exp(-DT / TAU)
    innovation_scale = SIGMA_NOISE * math.sqrt(1.0 - phi * phi)

    previous = [0.0] * N_CHANNELS
    samples = []
    innovations = []

    for k in range(N_TIME):
        for channel in range(N_CHANNELS):
            z = rng.gauss(0.0, 1.0)
            value = (
                SIGMA_NOISE * z if k == 0
                else phi * previous[channel] + innovation_scale * z
            )
            previous[channel] = value
            samples.append(value)
            innovations.append(z)

    return samples, innovations


def calibration_bias(seed):
    rng = random.Random(seed)
    return (
        SIGMA_CAL_KPA * rng.gauss(0.0, 1.0),
        SIGMA_CAL_KPA * rng.gauss(0.0, 1.0),
    )


def input_measurements(noise, scenario, biases):
    values = list(noise)

    if scenario == "c2":
        inlet, outlet = biases
        for k in range(N_TIME):
            values[4 * k] += inlet / 10.0
            values[4 * k + 2] += outlet / 10.0

    return values


def prepare_input(path, values, innovations):
    rows = [
        (index, format(value, ".17g"), format(z, ".17g"))
        for index, (value, z) in enumerate(zip(values, innovations))
    ]

    if not path.exists():
        write_csv_new(
            path,
            ["measurement_index", "noise_normalized", "whitened_noise"],
            rows,
        )
    else:
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            actual = list(reader)

        expected = [
            ["measurement_index", "noise_normalized", "whitened_noise"],
            *[[str(a), b, c] for a, b, c in rows],
        ]

        if actual != expected:
            raise RuntimeError(f"Existing input differs: {path}")

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_log(path):
    result = {}

    for line in path.read_text().splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()

    required = [
        "iterations", "converged", "reason", "final_objective",
        "gradient_norm", "terminal_inventory_error_kg",
        "maximum_mass_defect_kg", "elapsed_seconds",
    ]

    missing = [key for key in required if key not in result]
    if missing:
        raise RuntimeError(f"{path}: missing {missing}")

    for key in (
        "final_objective", "gradient_norm",
        "terminal_inventory_error_kg",
        "maximum_mass_defect_kg", "elapsed_seconds",
    ):
        if not math.isfinite(float(result[key])):
            raise RuntimeError(f"{path}: nonfinite {key}")

    return result


def run_one(scenario, replicate, model, input_path, sha, biases,
            noise_seed, bias_seed):
    directory = OUTPUT / scenario
    stem = f"r{replicate:03d}_{model}"

    log_path = directory / "logs" / f"{stem}.log"
    solution_path = directory / "solutions" / f"{stem}.csv"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    solution_path.parent.mkdir(parents=True, exist_ok=True)

    if model == "original":
        command = [
            str(BASE_EXE), str(input_path),
            str(ITERATIONS), str(solution_path),
        ]
    else:
        command = [
            str(CAL_EXE), str(input_path),
            str(ITERATIONS), str(SIGMA_CAL_KPA),
            str(RHO_CAL), str(solution_path),
        ]

    if log_path.exists() != solution_path.exists():
        raise RuntimeError(
            f"Incomplete prior run: {log_path} / {solution_path}. "
            "Inspect it before retrying."
        )

    if not log_path.exists():
        # Exclusive creation prevents accidental log replacement.
        with log_path.open("x") as stream:
            completed = subprocess.run(
                command, stdout=stream,
                stderr=subprocess.STDOUT, check=False,
                cwd=ROOT,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Solver failed (code {completed.returncode}): "
                f"{log_path}"
            )

        if not solution_path.exists():
            raise RuntimeError(
                f"Solver returned success without solution: {log_path}"
            )

    parsed = parse_log(log_path)

    expected_dimensions = 163 if model == "original" else 165
    with solution_path.open(newline="") as stream:
        solution_rows = list(csv.reader(stream))

    if (
        len(solution_rows) != expected_dimensions + 1
        or solution_rows[0] !=
        ["coordinate", "reconstructed_scaled_value"]
    ):
        raise RuntimeError(f"Unexpected solution shape: {solution_path}")

    return {
        "scenario": scenario,
        "replicate": replicate,
        "noise_seed": noise_seed,
        "bias_seed": bias_seed if scenario == "c2" else "",
        "true_inlet_bias_kpa": biases[0] if scenario == "c2" else 0.0,
        "true_outlet_bias_kpa": biases[1] if scenario == "c2" else 0.0,
        "model": model,
        "status": "ok",
        "returncode": 0,
        "iterations": parsed["iterations"],
        "converged": parsed["converged"],
        "reason": parsed["reason"],
        "final_objective": parsed["final_objective"],
        "gradient_norm": parsed["gradient_norm"],
        "terminal_inventory_error_kg":
            parsed["terminal_inventory_error_kg"],
        "maximum_mass_defect_kg":
            parsed["maximum_mass_defect_kg"],
        "elapsed_seconds": parsed["elapsed_seconds"],
        "estimated_inlet_bias_kpa":
            parsed.get("calibration_inlet_kpa", ""),
        "estimated_outlet_bias_kpa":
            parsed.get("calibration_outlet_kpa", ""),
        "noise_sha256": sha,
        "log_file": str(log_path.relative_to(ROOT)),
        "solution_file": str(solution_path.relative_to(ROOT)),
    }


def percentile(sorted_values, fraction):
    index = (len(sorted_values) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    weight = index - lower
    return (
        (1.0 - weight) * sorted_values[lower]
        + weight * sorted_values[upper]
    )


def bootstrap_paired(original, calibrated, seed=273000, draws=10000):
    """95% percentile interval for mean paired absolute-error change."""
    rng = random.Random(seed)
    differences = [
        abs(c) - abs(o)
        for o, c in zip(original, calibrated)
    ]
    n = len(differences)
    means = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(draws)
    )
    return (
        statistics.mean(differences),
        percentile(means, 0.025),
        percentile(means, 0.975),
    )


def summarize(records, expected_count):
    for scenario in ("c1", "c2"):
        subset = [
            row for row in records
            if row["scenario"] == scenario
        ]
        by_model = {
            model: {
                int(row["replicate"]): row
                for row in subset if row["model"] == model
            }
            for model in ("original", "calibrated")
        }

        common = sorted(
            set(by_model["original"])
            & set(by_model["calibrated"])
        )

        if len(common) != expected_count:
            raise RuntimeError(
                f"{scenario}: only {len(common)}/{expected_count} "
                "complete paired realizations"
            )

        errors = {
            model: [
                float(by_model[model][i][
                    "terminal_inventory_error_kg"
                ])
                for i in common
            ]
            for model in ("original", "calibrated")
        }

        print(f"\n{scenario.upper()}: {len(common)} paired realizations")

        for model, values in errors.items():
            mean = statistics.mean(values)
            rmse = math.sqrt(
                statistics.mean(e * e for e in values)
            )
            mae = statistics.mean(abs(e) for e in values)
            median_ae = statistics.median(abs(e) for e in values)
            q = sorted(values)

            converged = sum(
                str(by_model[model][i]["converged"]).lower()
                in ("1", "true")
                for i in common
            )

            max_defect = max(
                float(by_model[model][i][
                    "maximum_mass_defect_kg"
                ])
                for i in common
            )

            print(
                f"  {model:10s}: "
                f"bias={mean:+.3f} kg  "
                f"RMSE={rmse:.3f} kg  "
                f"MAE={mae:.3f} kg  "
                f"medianAE={median_ae:.3f} kg  "
                f"q05={percentile(q, .05):+.3f}  "
                f"q95={percentile(q, .95):+.3f}  "
                f"converged={converged}/{len(common)}  "
                f"max_mass_defect={max_defect:.3g} kg"
            )

        delta, low, high = bootstrap_paired(
            errors["original"], errors["calibrated"]
        )

        wins = sum(
            abs(c) < abs(o)
            for o, c in zip(
                errors["original"], errors["calibrated"]
            )
        )

        print(
            "  calibrated - original mean absolute error: "
            f"{delta:+.3f} kg "
            f"(paired bootstrap 95% CI [{low:+.3f}, {high:+.3f}]); "
            f"calibrated wins {wins}/{len(common)}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count", type=int, default=32,
        help="Realizations per scenario (default: 32)",
    )
    args = parser.parse_args()

    if not 1 <= args.count <= 32:
        parser.error("--count must be between 1 and 32")

    for exe in (BASE_EXE, CAL_EXE):
        if not exe.is_file():
            parser.error(f"Missing executable: {exe}")

    print(
        f"M27-C: {args.count} realizations per scenario, "
        f"{4 * args.count} total solves",
        flush=True,
    )

    records = []

    for scenario in ("c1", "c2"):
        directory = OUTPUT / scenario

        for replicate in range(args.count):
            noise_seed = 271000 + replicate
            bias_seed = 272000 + replicate

            noise, innovations = noise_realization(noise_seed)
            biases = calibration_bias(bias_seed)

            values = input_measurements(
                noise, scenario, biases
            )

            input_path = (
                directory / "inputs"
                / f"r{replicate:03d}.csv"
            )

            sha = prepare_input(
                input_path, values, innovations
            )

            for model in ("original", "calibrated"):
                print(
                    f"[{scenario} {replicate+1:02d}/"
                    f"{args.count:02d}] {model}",
                    flush=True,
                )

                row = run_one(
                    scenario, replicate, model,
                    input_path, sha, biases,
                    noise_seed, bias_seed,
                )

                records.append(row)

    summary_path = OUTPUT / "reconstruction_results.csv"

    # Allow an earlier pilot to be extended to 32 realizations.
    # Refuse to overwrite a completed summary.
    if summary_path.exists():
        with summary_path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))

        if existing != [
            {key: str(row.get(key, "")) for key in FIELDS}
            for row in records
        ]:
            raise RuntimeError(
                f"Existing summary differs: {summary_path}. "
                "Preserve or relocate it before generating a new one."
            )
    else:
        write_csv_new(
            summary_path,
            FIELDS,
            [[row.get(key, "") for key in FIELDS]
             for row in records],
        )

    summarize(records, args.count)

    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
