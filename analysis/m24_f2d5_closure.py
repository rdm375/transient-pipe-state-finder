#!/usr/bin/env python3
"""M2.4-F2D-5: strict-feasibility correction and scientific closure.

Selects the best F2D-4 replay-validated candidate for each observable.
Uses radial contraction with nonlinear oracle evaluations to establish
strict numerical feasibility, then writes reproducibility artifacts.

Scope:
  - Fixed terminal observables and indices from F2B/F2D.
  - Existing C++ nonlinear oracle (not independent physics validation).
  - Feasible adversarial witnesses, not certified global optima.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m24_f2d_nonlinear_optimize import Oracle, ROOT, LABELS, N


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def git_output(*args):
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_best():
    results = pd.read_csv(ROOT / "f2d4_replay_results.csv")
    vectors = pd.read_csv(ROOT / "f2d4_validated_directions.csv")

    best = []

    for quantity in LABELS:
        eligible = results[
            (results.quantity == quantity)
            & (results.replay_pass.astype(str).str.lower() == "true")
        ]

        if eligible.empty:
            raise RuntimeError(
                f"No replay-validated candidate for {quantity}"
            )

        row = eligible.loc[eligible.terminal_abs.idxmax()]
        source = str(row.source)
        sign = int(row.sign)

        subset = vectors[
            (vectors.source == source)
            & (vectors.quantity == quantity)
            & (vectors.sign == sign)
        ].sort_values("coordinate")

        if len(subset) != N:
            raise RuntimeError(
                f"Missing coordinates: {source}/{quantity}/{sign}"
            )

        if not np.array_equal(
            subset.coordinate.to_numpy(), np.arange(N)
        ):
            raise RuntimeError(
                f"Invalid coordinate order: {quantity}"
            )

        x = subset.value.to_numpy(dtype=float)

        if not np.isfinite(x).all():
            raise RuntimeError(f"Nonfinite direction: {quantity}")

        best.append((quantity, source, sign, row, x))

    return best


def evaluate(oracle, x, q, sign, bound):
    values, r2, pmin, defect, gradients = oracle.evaluate(x)

    objective = float(values[q])
    r2 = float(r2)
    pmin = float(pmin)
    defect = float(defect)
    initial2 = float(x @ x)

    return {
        "terminal_signed": objective,
        "terminal_abs": abs(objective),
        "signed_ratio": sign * objective / bound,
        "absolute_ratio": abs(objective) / bound,
        "initial_squared": initial2,
        "measurement_squared": r2,
        "initial_norm": float(np.sqrt(initial2)),
        "measurement_norm": float(np.sqrt(max(0.0, r2))),
        "min_pressure_Pa": pmin,
        "conservation_defect_kg": defect,
        "objective_gradient": np.asarray(
            gradients[q], dtype=float
        ),
        "measurement_gradient": np.asarray(
            gradients[3], dtype=float
        ),
    }


def feasible(data, margin, conservation_tol):
    return (
        np.isfinite(data["initial_squared"])
        and np.isfinite(data["measurement_squared"])
        and data["initial_squared"] <= 1.0 - margin
        and 0.0 <= data["measurement_squared"] <= 1.0 - margin
        and np.isfinite(data["min_pressure_Pa"])
        and data["min_pressure_Pa"] > 0.0
        and np.isfinite(data["conservation_defect_kg"])
        and 0.0 <= data["conservation_defect_kg"]
        <= conservation_tol
    )


def contract(
    oracle, x, q, sign, bound, margin, conservation_tol
):
    """Find a feasible radial contraction, then bisect its boundary.

    No monotonicity assumption is made about the nonlinear measurement
    constraint. The final point is always explicitly oracle-validated.
    """

    original = evaluate(oracle, x, q, sign, bound)

    if feasible(original, margin, conservation_tol):
        return x.copy(), original, 1.0, 0

    # Try progressively stronger contractions until a feasible
    # point is found. A feasible point is checked, not assumed.
    scales = [
        1.0 - 10.0**(-k) for k in range(12, 1, -1)
    ]
    scales += [
        0.99, 0.98, 0.95, 0.90, 0.80,
        0.60, 0.40, 0.20, 0.10, 0.0
    ]

    scales = sorted(set(scales), reverse=True)

    high = 1.0
    low = None
    low_data = None
    evaluations = 0

    for scale in scales:
        try:
            data = evaluate(
                oracle, scale * x, q, sign, bound
            )
            evaluations += 1
        except RuntimeError:
            continue

        if feasible(data, margin, conservation_tol):
            low = scale
            low_data = data
            break

        high = scale

    if low is None:
        raise RuntimeError(
            "No feasible radial contraction found"
        )

    # Refine between a feasible lower scale and the nearest
    # previously checked infeasible upper scale.
    # This is a local boundary search, not a proof of monotonicity.
    for _ in range(45):
        mid = 0.5 * (low + high)

        if mid == low or mid == high:
            break

        try:
            data = evaluate(
                oracle, mid * x, q, sign, bound
            )
            evaluations += 1
            ok = feasible(
                data, margin, conservation_tol
            )
        except RuntimeError:
            ok = False

        if ok:
            low = mid
            low_data = data
        else:
            high = mid

    corrected = low * x

    # Final evaluation is mandatory.
    final = evaluate(
        oracle, corrected, q, sign, bound
    )
    evaluations += 1

    if not feasible(final, margin, conservation_tol):
        raise RuntimeError(
            "Final contracted direction is not feasible"
        )

    return corrected, final, low, evaluations


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("build/state_finder_m24_f2d_oracle"),
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=1e-8,
        help="Strict interior margin on squared constraints",
    )

    parser.add_argument(
        "--conservation-tol",
        type=float,
        default=1e-6,
    )

    parser.add_argument(
        "--max-relative-loss",
        type=float,
        default=1e-5,
        help="Maximum permitted relative objective reduction",
    )

    args = parser.parse_args()

    if not (0 < args.margin < 1):
        parser.error("--margin must be between 0 and 1")

    if args.conservation_tol <= 0:
        parser.error("--conservation-tol must be positive")

    if args.max_relative_loss < 0:
        parser.error("--max-relative-loss must be nonnegative")

    candidates = load_best()

    oracle_path = args.oracle.resolve()

    if not oracle_path.is_file():
        raise FileNotFoundError(oracle_path)

    results = []
    directions = []
    failures = []

    oracle = Oracle(args.oracle)

    try:
        for quantity, source, sign, saved, x in candidates:
            q = LABELS.index(quantity)
            bound = float(saved.F2B_linear_bound)

            original = evaluate(
                oracle, x, q, sign, bound
            )

            corrected, final, scale, neval = contract(
                oracle,
                x,
                q,
                sign,
                bound,
                args.margin,
                args.conservation_tol,
            )

            # Compare against independently replayed original,
            # not just a stored CSV objective.
            original_abs = original["terminal_abs"]
            final_abs = final["terminal_abs"]

            relative_loss = (
                (original_abs - final_abs) / original_abs
                if original_abs > 0 else float("inf")
            )

            sign_ok = sign * final["terminal_signed"] > 0
            exceedance_ok = final["absolute_ratio"] > 1.0
            loss_ok = relative_loss <= args.max_relative_loss

            strictly_feasible = feasible(
                final,
                args.margin,
                args.conservation_tol,
            )

            passed = all((
                strictly_feasible,
                sign_ok,
                exceedance_ok,
                loss_ok,
            ))

            # Diagnostic stationarity: contraction generally
            # moves a point away from the active boundary.
            # Do not misrepresent the original KKT residual
            # as a KKT certificate for the corrected point.
            result = {
                "quantity": quantity,
                "source": source,
                "sign": sign,
                "pass": passed,
                "contraction_scale": scale,
                "contraction_evaluations": neval,
                "original_terminal_abs": original_abs,
                "corrected_terminal_abs": final_abs,
                "F2B_linear_bound": bound,
                "original_ratio": original["absolute_ratio"],
                "corrected_ratio": final["absolute_ratio"],
                "relative_objective_loss": relative_loss,
                "initial_norm": final["initial_norm"],
                "measurement_norm": final["measurement_norm"],
                "initial_squared_margin": (
                    1.0 - final["initial_squared"]
                ),
                "measurement_squared_margin": (
                    1.0 - final["measurement_squared"]
                ),
                "min_pressure_Pa": final["min_pressure_Pa"],
                "conservation_defect_kg": (
                    final["conservation_defect_kg"]
                ),
                "strictly_feasible": strictly_feasible,
                "sign_ok": sign_ok,
                "exceedance_ok": exceedance_ok,
                "objective_loss_ok": loss_ok,
            }

            results.append(result)

            directions.extend(
                {
                    "quantity": quantity,
                    "source": source,
                    "sign": sign,
                    "coordinate": j,
                    "value": float(value),
                }
                for j, value in enumerate(corrected)
            )

            print(
                f"{quantity:9s} "
                f"PASS={passed!s:5s} "
                f"scale={scale:.12g} "
                f"|v|={final['initial_norm']:.12g} "
                f"|r|={final['measurement_norm']:.12g} "
                f"ratio={final['absolute_ratio']:.12g} "
                f"loss={relative_loss:.3e}",
                flush=True,
            )

            if not passed:
                failures.append(quantity)

    finally:
        oracle.close()

    ROOT.mkdir(parents=True, exist_ok=True)

    results_path = ROOT / "f2d5_closure_results.csv"
    directions_path = ROOT / "f2d5_strict_directions.csv"
    manifest_path = ROOT / "f2d5_manifest.json"

    pd.DataFrame(results).to_csv(
        results_path, index=False, float_format="%.17g"
    )

    pd.DataFrame(directions).to_csv(
        directions_path, index=False, float_format="%.17g"
    )

    script_path = Path(__file__).resolve()

    inputs = [
        ROOT / "f2d4_replay_results.csv",
        ROOT / "f2d4_validated_directions.csv",
    ]

    manifest = {
        "experiment": "M2.4-F2D-5",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_level": (
            "Strict-feasibility correction and saved-direction "
            "replay using the existing nonlinear C++ oracle"
        ),
        "independent_physics_implementation": False,
        "global_optimality_certified": False,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status_porcelain": git_output(
            "status", "--porcelain"
        ),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "oracle_path": str(oracle_path),
        "oracle_sha256": sha256(oracle_path),
        "script_sha256": sha256(script_path),
        "inputs_sha256": {
            str(p): sha256(p) for p in inputs
        },
        "strict_squared_constraint_margin": args.margin,
        "conservation_tolerance_kg": (
            args.conservation_tol
        ),
        "maximum_relative_objective_loss": (
            args.max_relative_loss
        ),
        "passed": not failures,
        "failed_quantities": failures,
        "results_sha256": sha256(results_path),
        "directions_sha256": sha256(directions_path),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print()
    print("F2D-5 SCIENTIFIC CLOSURE")
    print("-----------------------")
    print(f"Results:    {results_path}")
    print(f"Directions: {directions_path}")
    print(f"Manifest:   {manifest_path}")

    if failures:
        raise SystemExit(
            "FAIL: closure criteria not satisfied for "
            + ", ".join(failures)
        )

    print("PASS: all three strictly feasible witnesses validated")


if __name__ == "__main__":
    main()
