#!/usr/bin/env python3
"""
M2.4-F2B: Exact constrained operational uncertainty optimization.

Find worst-case linearized operational errors subject to:

    ||v||_2       <= r_x
    ||J_x v||_2   <= epsilon_y

Boundary histories are fixed.

The optimization is convex. Numerical results are subject
to solver tolerances and conditioning.
"""

from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


ROOT = Path("benchmarks/results-m24")

J_PATH = ROOT / "joint_measurement_jacobian.csv"
G_PATH = ROOT / "operational_sensitivity.csv"
F2_PATH = ROOT / "measurement_conditioned_certificate.csv"

OUTPUT_PATH = ROOT / "exact_operational_bounds.csv"
DIRECTIONS_PATH = ROOT / "adversarial_initial_states.csv"

NX = 41
TIME = 600.0

RADIUS = 1.0
MEASUREMENT_BOUNDS = [1.0, 5.0, 15.0]

TOLERANCES = {
    "pressure": 100.0,
    "flow": 0.01,
    "inventory": 10.0,
}

OUTPUT_ROWS = {
    "pressure": 21,
    "flow": 22,
    "inventory": 1,
}


def load_jacobian():
    try:
        J = np.loadtxt(J_PATH, delimiter=",")
    except ValueError:
        J = np.loadtxt(
            J_PATH,
            delimiter=",",
            skiprows=1,
        )

    if J.shape != (244, 163):
        raise ValueError(
            f"Unexpected Jacobian shape: {J.shape}"
        )

    if not np.isfinite(J).all():
        raise ValueError("Nonfinite Jacobian")

    return J[:, :NX]


def load_sensitivity(quantity, time):
    df = pd.read_csv(G_PATH)

    subset = df[
        np.isclose(df.time_s, time) &
        (df.quantity == quantity)
    ]

    rows = OUTPUT_ROWS[quantity]

    G = np.full((rows, NX), np.nan)

    for row in subset.itertuples(index=False):
        i = int(row.index)
        j = int(row.column)

        if not (0 <= i < rows and 0 <= j < NX):
            raise ValueError("Invalid sensitivity index")

        if np.isfinite(G[i, j]):
            raise ValueError("Duplicate sensitivity entry")

        G[i, j] = row.sensitivity

    if not np.isfinite(G).all():
        raise ValueError(
            f"Incomplete {quantity} sensitivity"
        )

    return G


def solve_worst_case(g, Jx, epsilon):
    """
    Maximize g @ v over the intersection of two L2 balls.

    Central symmetry implies:

        max |g @ v| = max g @ v
    """

    v = cp.Variable(NX)

    problem = cp.Problem(
        cp.Maximize(g @ v),
        [
            cp.norm(v, 2) <= RADIUS,
            cp.norm(Jx @ v, 2) <= epsilon,
        ],
    )

    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
        max_iter=500,
        verbose=False,
    )

    if problem.status not in (
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    ):
        raise RuntimeError(
            f"Optimization failed: {problem.status}"
        )

    if v.value is None:
        raise RuntimeError("Missing optimizer solution")

    direction = np.asarray(v.value).reshape(-1)

    initial_norm = np.linalg.norm(direction)
    measurement_norm = np.linalg.norm(Jx @ direction)

    # Check constraint satisfaction independently.
    if initial_norm > RADIUS * (1.0 + 1e-6):
        raise RuntimeError(
            "Initial uncertainty constraint violated"
        )

    if measurement_norm > epsilon * (1.0 + 1e-6):
        raise RuntimeError(
            "Measurement constraint violated"
        )

    return {
        "bound": float(g @ direction),
        "direction": direction,
        "initial_norm": initial_norm,
        "measurement_norm": measurement_norm,
        "status": problem.status,
    }


def main():
    Jx = load_jacobian()

    sensitivities = {
        quantity: load_sensitivity(quantity, TIME)
        for quantity in TOLERANCES
    }

    f2 = pd.read_csv(F2_PATH)

    records = []
    directions = []

    print()
    print("M2.4-F2B EXACT OPERATIONAL BOUNDS")
    print("=" * 78)
    print(f"Time: {TIME:g} s")
    print(f"Initial-state dimension: {NX}")
    print(f"Initial uncertainty radius: {RADIUS}")
    print(f"Measurement Jacobian: {Jx.shape}")

    for epsilon in MEASUREMENT_BOUNDS:

        print()
        print(f"Measurement residual bound: {epsilon:g}")
        print("-" * 78)

        for quantity, tolerance in TOLERANCES.items():

            G = sensitivities[quantity]

            best = None
            best_index = None

            # Find the worst spatial location.
            for index, g in enumerate(G):

                result = solve_worst_case(
                    g, Jx, epsilon
                )

                if (
                    best is None or
                    result["bound"] > best["bound"]
                ):
                    best = result
                    best_index = index

            conservative = f2[
                np.isclose(
                    f2.measurement_bound,
                    epsilon,
                ) &
                np.isclose(
                    f2.time_s,
                    TIME,
                )
            ]

            if len(conservative) != 1:
                raise ValueError(
                    "Missing F2 comparison row"
                )

            f2_bound = float(
                conservative.iloc[0][
                    f"{quantity}_bound"
                ]
            )

            exact_bound = best["bound"]

            record = {
                "measurement_bound": epsilon,
                "time_s": TIME,
                "quantity": quantity,
                "worst_index": best_index,
                "exact_bound": exact_bound,
                "f2_upper_bound": f2_bound,
                "upper_to_exact_ratio": (
                    f2_bound / exact_bound
                    if exact_bound > 0
                    else np.nan
                ),
                "normalized_error": (
                    exact_bound / tolerance
                ),
                "initial_norm": best["initial_norm"],
                "measurement_norm": best["measurement_norm"],
                "solver_status": best["status"],
            }

            records.append(record)

            for j, value in enumerate(best["direction"]):
                directions.append({
                    "measurement_bound": epsilon,
                    "quantity": quantity,
                    "time_s": TIME,
                    "worst_index": best_index,
                    "initial_coordinate": j,
                    "scaled_perturbation": value,
                })

            print(
                f"{quantity:12s} "
                f"exact={exact_bound:14.7g} "
                f"F2_upper={f2_bound:14.7g} "
                f"ratio={f2_bound/exact_bound:8.4f} "
                f"normalized={exact_bound/tolerance:10.4f}"
            )

            print(
                f"  worst_index={best_index} "
                f"||v||={best['initial_norm']:.9g} "
                f"||Jv||={best['measurement_norm']:.9g} "
                f"status={best['status']}"
            )

    pd.DataFrame(records).to_csv(
        OUTPUT_PATH,
        index=False,
    )

    pd.DataFrame(directions).to_csv(
        DIRECTIONS_PATH,
        index=False,
    )

    print()
    print("Saved:")
    print(" ", OUTPUT_PATH)
    print(" ", DIRECTIONS_PATH)

    print()
    print("Classification:")
    print("  Numerically optimized linearized worst-case bounds")
    print("  Fixed boundary histories")
    print("  Retrospective measurement conditioning")
    print("  Experimental uncertainty assumptions")
    print("  Not a nonlinear guarantee")


if __name__ == "__main__":
    main()
