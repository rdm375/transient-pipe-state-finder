#!/usr/bin/env python3
"""M2.4-F2D-4: replay saved nonlinear adversarial candidates.

Re-evaluates saved directions using the persistent C++ nonlinear oracle.
This is an independent replay, not an independent physics implementation.

The oracle's conservation diagnostic is reproduced, not independently
rederived here. Global optimality is not assessed.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m24_f2d_nonlinear_optimize import Oracle, ROOT, LABELS, N


SOURCES = (
    (
        "f2d1",
        "f2d_nonlinear_local_optima.csv",
        "f2d_nonlinear_local_optima_directions.csv",
    ),
    (
        "f2d2",
        "f2d2_trust_constr_optima.csv",
        "f2d2_trust_constr_directions.csv",
    ),
    (
        "f2d3",
        "f2d3_warmstart_optima.csv",
        "f2d3_warmstart_directions.csv",
    ),
)


def boolean(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def direction(df, quantity, sign):
    selected = df[
        (df.quantity == quantity) &
        (df.sign == sign)
    ].sort_values("coordinate")

    indices = selected.coordinate.to_numpy()

    if len(selected) != N or not np.array_equal(
        indices, np.arange(N)
    ):
        raise ValueError(
            f"Incomplete direction: {quantity}, sign={sign}"
        )

    values = selected.value.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Nonfinite direction")

    return values


def load_candidates():
    candidates = []

    for source, summary_name, directions_name in SOURCES:
        summary_path = ROOT / summary_name
        directions_path = ROOT / directions_name

        if not summary_path.is_file():
            print(f"SKIP {source}: summary missing")
            continue

        if not directions_path.is_file():
            print(f"SKIP {source}: directions missing")
            continue

        summary = pd.read_csv(summary_path)
        directions = pd.read_csv(directions_path)

        for _, row in summary.iterrows():
            if "feasible" in row.index:
                if not boolean(row["feasible"]):
                    continue
            elif (
                "initial_violation" in row.index
                and "measurement_violation" in row.index
            ):
                # F2D-3 uses diagnostic violations rather than
                # a separate feasibility column. Replay will
                # independently enforce the strict constraints.
                if (
                    float(row.initial_violation) > 1e-8
                    or float(row.measurement_violation) > 1e-8
                ):
                    continue
            else:
                raise ValueError(
                    f"Unknown feasibility schema: {source}"
                )

            quantity = str(row.quantity)
            sign = int(row.sign)

            if quantity not in LABELS or sign not in (-1, 1):
                raise ValueError(
                    f"Invalid candidate: {source}, "
                    f"{quantity}, {sign}"
                )

            x = direction(directions, quantity, sign)

            candidates.append(
                (source, quantity, sign, row, x)
            )

    return candidates


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--oracle",
        type=Path,
        default=Path("build/state_finder_m24_f2d_oracle"),
    )

    parser.add_argument(
        "--constraint-tol",
        type=float,
        default=1e-10,
        help="Tolerance on squared uncertainty constraints",
    )

    parser.add_argument(
        "--objective-rtol",
        type=float,
        default=1e-9,
    )

    parser.add_argument(
        "--objective-atol",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--conservation-tol",
        type=float,
        default=1e-6,
    )

    args = parser.parse_args()

    candidates = load_candidates()

    if not candidates:
        raise SystemExit("No feasible saved candidates found")

    bounds = pd.read_csv(
        ROOT / "exact_operational_bounds.csv"
    )

    linear = {}

    for quantity in LABELS:
        selected = bounds[
            np.isclose(bounds.measurement_bound, 1.0) &
            np.isclose(bounds.time_s, 600.0) &
            (bounds.quantity == quantity)
        ]

        if len(selected) != 1:
            raise ValueError(
                f"Expected exactly one F2B bound: {quantity}"
            )

        linear[quantity] = float(
            selected.iloc[0].exact_bound
        )

    oracle = Oracle(args.oracle)

    rows = []
    validated_directions = []

    try:
        for source, quantity, sign, saved, x in candidates:
            q = LABELS.index(quantity)

            values, r2, pmin, defect, _ = oracle.evaluate(x)

            objective = float(values[q])
            absolute = abs(objective)
            initial_squared = float(x @ x)
            measurement_squared = float(r2)

            initial_norm = float(np.sqrt(initial_squared))
            measurement_norm = float(
                np.sqrt(max(measurement_squared, 0.0))
            )

            ratio = absolute / linear[quantity]

            # F2D-1 and F2D-2 store terminal_abs.
            # F2D-3 stores the signed normalized score.
            if "terminal_abs" in saved.index:
                saved_absolute = float(saved.terminal_abs)
            elif "score" in saved.index:
                saved_absolute = (
                    abs(float(saved.score)) * linear[quantity]
                )
            else:
                raise ValueError(
                    f"No saved objective: {source}, {quantity}"
                )

            objective_error = abs(
                absolute - saved_absolute
            )

            objective_limit = (
                args.objective_atol +
                args.objective_rtol * max(
                    absolute, saved_absolute
                )
            )

            objective_agrees = (
                objective_error <= objective_limit
            )

            initial_ok = (
                initial_squared <=
                1.0 + args.constraint_tol
            )

            measurement_ok = (
                measurement_squared <=
                1.0 + args.constraint_tol
            )

            pressure_ok = (
                np.isfinite(pmin) and pmin > 0.0
            )

            conservation_ok = (
                np.isfinite(defect) and
                defect <= args.conservation_tol
            )

            sign_ok = (
                sign * objective >= 0.0
            )

            passed = all((
                objective_agrees,
                initial_ok,
                measurement_ok,
                pressure_ok,
                conservation_ok,
                sign_ok,
            ))

            record = dict(
                source=source,
                quantity=quantity,
                sign=sign,
                replay_pass=passed,
                objective_agrees=objective_agrees,
                initial_ok=initial_ok,
                measurement_ok=measurement_ok,
                pressure_ok=pressure_ok,
                conservation_ok=conservation_ok,
                sign_ok=sign_ok,
                initial_norm=initial_norm,
                measurement_norm=measurement_norm,
                initial_squared_violation=max(
                    0.0, initial_squared - 1.0
                ),
                measurement_squared_violation=max(
                    0.0, measurement_squared - 1.0
                ),
                terminal_signed=objective,
                terminal_abs=absolute,
                saved_terminal_abs=saved_absolute,
                objective_absolute_difference=objective_error,
                F2B_linear_bound=linear[quantity],
                nonlinear_to_F2B_ratio=ratio,
                min_pressure_Pa=float(pmin),
                max_conservation_defect_kg=float(defect),
            )

            rows.append(record)

            if passed:
                validated_directions.extend(
                    dict(
                        source=source,
                        quantity=quantity,
                        sign=sign,
                        coordinate=j,
                        value=float(value),
                    )
                    for j, value in enumerate(x)
                )

            print(
                f"{source:5s} {quantity:9s} "
                f"sign={sign:+d} "
                f"PASS={passed!s:5s} "
                f"|v|={initial_norm:.12g} "
                f"|r|={measurement_norm:.12g} "
                f"ratio={ratio:.12g} "
                f"delta={objective_error:.3e}",
                flush=True,
            )

    finally:
        oracle.close()

    ROOT.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(rows)

    results.to_csv(
        ROOT / "f2d4_replay_results.csv",
        index=False,
    )

    pd.DataFrame(
        validated_directions,
        columns=[
            "source",
            "quantity",
            "sign",
            "coordinate",
            "value",
        ],
    ).to_csv(
        ROOT / "f2d4_validated_directions.csv",
        index=False,
    )

    print("\nBEST REPLAY-VALIDATED CANDIDATES")

    best = {}

    for quantity in LABELS:
        selected = results[
            (results.quantity == quantity) &
            (results.replay_pass)
        ]

        if selected.empty:
            print(f"{quantity}: NO VALIDATED CANDIDATE")
            continue

        winner = selected.loc[
            selected.terminal_abs.idxmax()
        ]

        best[quantity] = {
            "source": str(winner.source),
            "sign": int(winner.sign),
            "terminal_abs": float(winner.terminal_abs),
            "linear_bound": float(winner.F2B_linear_bound),
            "ratio": float(
                winner.nonlinear_to_F2B_ratio
            ),
        }

        print(
            f"{quantity:9s} "
            f"{winner.terminal_abs:.12g} "
            f"ratio={winner.nonlinear_to_F2B_ratio:.12g} "
            f"source={winner.source} "
            f"sign={int(winner.sign):+d}"
        )

    summary = {
        "experiment": "M2.4-F2D-4",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_level": (
            "Saved-direction replay using the existing "
            "C++ nonlinear oracle; not independent physics"
        ),
        "global_optimality_certified": False,
        "constraint_tolerance_squared": args.constraint_tol,
        "objective_relative_tolerance": args.objective_rtol,
        "objective_absolute_tolerance": args.objective_atol,
        "conservation_tolerance_kg": args.conservation_tol,
        "candidates_checked": len(results),
        "candidates_passed": int(
            results.replay_pass.sum()
        ),
        "best_validated": best,
    }

    with (
        ROOT / "f2d4_replay_summary.json"
    ).open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    failed = results[~results.replay_pass]

    print(
        f"\nReplay checks: "
        f"{len(results) - len(failed)}/{len(results)} passed"
    )

    if not failed.empty:
        print("\nFAILED CANDIDATES")
        print(
            failed[
                [
                    "source",
                    "quantity",
                    "sign",
                    "objective_agrees",
                    "initial_ok",
                    "measurement_ok",
                    "pressure_ok",
                    "conservation_ok",
                    "sign_ok",
                ]
            ].to_string(index=False)
        )

    if len(best) != len(LABELS):
        raise SystemExit(
            "FAIL: not every quantity has a validated candidate"
        )

    print("\nF2D-4 replay completed.")


if __name__ == "__main__":
    main()
