#!/usr/bin/env python3

import csv
import os
import re
import subprocess
from pathlib import Path

import numpy as np

OUTPUT = Path("benchmarks/results-m19a")
OUTPUT.mkdir(parents=True, exist_ok=True)

EXECUTABLE = "./build/state_finder_inprocess_lanczos"

ROWS = []

env = os.environ.copy()
env["OPENBLAS_NUM_THREADS"] = "2"
env["OMP_NUM_THREADS"] = "2"

def extract(text, key):
    match = re.search(
        rf"^{re.escape(key)}=(.+)$",
        text,
        re.MULTILINE
    )

    if not match:
        raise RuntimeError(f"Missing {key}")

    return match.group(1).strip()

for n in (100, 200, 400):
    for start_step, window in (
        (0, "full"),
        (80, "late"),
    ):
        name = f"{window}_n{n}"
        prefix = OUTPUT / name

        print(
            f"\n===== n={n}, window={window} =====",
            flush=True
        )

        command = [
            "/usr/bin/time",
            "-f",
            "peak_memory_kb=%M",
            EXECUTABLE,
            str(start_step),
            str(prefix),
            str(n),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env
        )

        (OUTPUT / f"{name}.log").write_text(
            result.stdout + "\n" + result.stderr
        )

        print(result.stdout, flush=True)

        if result.returncode:
            print(result.stderr)
            raise RuntimeError(
                f"Scaling run failed: {name}"
            )

        eigenpairs = np.loadtxt(
            str(prefix) + "_eigenpairs.csv",
            delimiter=",",
            skiprows=1
        )

        assert eigenpairs.shape == (3, 4)

        eigenvalues = eigenpairs[:, 1]

        # Reported for descriptive purposes, not yet
        # a continuum-independent effective rank.
        positive = np.maximum(eigenvalues, 0.0)
        total_top3 = positive.sum()

        top3_fraction = (
            positive[0] / total_top3
            if total_top3 > 0 else float("nan")
        )

        row = {
            "cells": n,
            "state_dimension": 2*n+1,
            "window": window,
            "start_step": start_step,
            "trajectory_seconds": float(
                extract(result.stdout, "trajectory_seconds")
            ),
            "operator_calls": int(
                extract(result.stdout, "operator_calls")
            ),
            "operator_seconds": float(
                extract(result.stdout, "operator_seconds")
            ),
            "total_seconds": float(
                extract(result.stdout, "total_seconds")
            ),
            "peak_memory_kb": int(
                extract(result.stderr, "peak_memory_kb")
            ),
            "lambda1": eigenvalues[0],
            "lambda2": eigenvalues[1],
            "lambda3": eigenvalues[2],
            "lambda2_over_lambda1":
                eigenvalues[1]/eigenvalues[0],
            "lambda3_over_lambda1":
                eigenvalues[2]/eigenvalues[0],
            "lambda1_fraction_of_top3":
                top3_fraction,
            "max_reported_ritz_residual":
                eigenpairs[:, 3].max(),
        }

        ROWS.append(row)

        print(
            f"lambda1={row['lambda1']:.9e} "
            f"lambda2/lambda1="
            f"{row['lambda2_over_lambda1']:.9e} "
            f"time={row['total_seconds']:.3f}s "
            f"calls={row['operator_calls']}",
            flush=True
        )

with (OUTPUT / "scaling_summary.csv").open(
    "w", newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=ROWS[0].keys(),
        lineterminator="\n"
    )

    writer.writeheader()
    writer.writerows(ROWS)

print(
    "\nPASS M1.9a spatial scaling sweep",
    flush=True
)
