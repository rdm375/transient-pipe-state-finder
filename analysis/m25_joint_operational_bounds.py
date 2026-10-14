#!/usr/bin/env python3
"""M2.5-B2: Joint measurement-conditioned operational bounds.

Linearized optimization over initial-state and boundary-history
perturbations, using the four-channel M24 measurement Jacobian
and M25 joint operational sensitivities.

The radii are exploratory bounds in scaled optimization coordinates.
They are not calibrated statistical confidence regions.
"""

from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd


M24 = Path("benchmarks/results-m24")
M25 = Path("benchmarks/results-m25")

NX = 41
NB = 122
DIM = NX + NB

TIME = 600.0

EPSILONS = (1.0, 5.0, 15.0)
BOUNDARY_RADII = (0.0, 1.0, 5.0)

INITIAL_RADIUS = 1.0

SOLVER = "CLARABEL"

OBJECTIVES = (
    ("outlet_pressure", "pressure", 20, "Pa"),
    ("inlet_flow", "flow", 0, "kg/s"),
    ("inventory", "inventory", 0, "kg"),
)

FEASIBILITY_RTOL = 1.0e-6
FEASIBILITY_ATOL = 1.0e-7


def load_measurement_jacobian():
    J = np.loadtxt(
        M24 / "joint_measurement_jacobian.csv",
        delimiter=",",
        skiprows=1,
    )

    if J.shape != (244, DIM):
        raise ValueError(
            f"Unexpected measurement Jacobian shape: {J.shape}"
        )

    if not np.isfinite(J).all():
        raise ValueError("Nonfinite measurement Jacobian")

    return J


def load_operational_gradients():
    df = pd.read_csv(
        M25 / "operational_sensitivity.csv"
    )

    gradients = {}

    for name, quantity, index, units in OBJECTIVES:

        selected = df[
            (df["time_s"] == TIME)
            & (df["quantity"] == quantity)
            & (df["index"] == index)
        ]

        selected = selected.sort_values("column")

        columns = selected["column"].to_numpy()

        if not np.array_equal(
            columns,
            np.arange(DIM),
        ):
            raise RuntimeError(
                f"Incomplete gradient: {name}"
            )

        g = selected["sensitivity"].to_numpy(
            dtype=float
        )

        if not np.isfinite(g).all():
            raise ValueError(
                f"Nonfinite operational gradient: {name}"
            )

        gradients[name] = (g, units)

    return gradients


def solve_bound(J, g, epsilon, boundary_radius):
    delta = cp.Variable(DIM)

    constraints = [
        cp.norm(J @ delta, 2) <= epsilon,
        cp.norm(delta[:NX], 2) <= INITIAL_RADIUS,
    ]

    if boundary_radius == 0.0:
        constraints.append(
            delta[NX:] == 0.0
        )
    else:
        constraints.append(
            cp.norm(delta[NX:], 2) <= boundary_radius
        )

    problem = cp.Problem(
        cp.Maximize(g @ delta),
        constraints,
    )

    problem.solve(
        solver=SOLVER,
        verbose=False,
        tol_gap_abs=1.0e-9,
        tol_gap_rel=1.0e-9,
        tol_feas=1.0e-9,
        max_iter=500,
    )

    if problem.status != cp.OPTIMAL:
        raise RuntimeError(
            f"Optimization status: {problem.status}"
        )

    if delta.value is None:
        raise RuntimeError(
            "Optimizer returned no direction"
        )

    direction = np.asarray(
        delta.value,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(direction).all():
        raise RuntimeError(
            "Nonfinite optimizer direction"
        )

    measurement_norm = float(
        np.linalg.norm(J @ direction)
    )

    initial_norm = float(
        np.linalg.norm(direction[:NX])
    )

    boundary_norm = float(
        np.linalg.norm(direction[NX:])
    )

    def check(value, bound, label):
        tolerance = (
            FEASIBILITY_ATOL
            + FEASIBILITY_RTOL * max(1.0, bound)
        )

        if value > bound + tolerance:
            raise RuntimeError(
                f"{label} infeasible: "
                f"{value} > {bound}"
            )

    check(
        measurement_norm,
        epsilon,
        "Measurement constraint",
    )

    check(
        initial_norm,
        INITIAL_RADIUS,
        "Initial-state radius",
    )

    check(
        boundary_norm,
        boundary_radius,
        "Boundary-history radius",
    )

    objective = float(g @ direction)

    if not np.isclose(
        objective,
        problem.value,
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise RuntimeError(
            "Objective mismatch"
        )

    return {
        "objective": objective,
        "measurement_norm": measurement_norm,
        "initial_norm": initial_norm,
        "boundary_norm": boundary_norm,
        "direction": direction,
        "status": problem.status,
    }


def main():
    M25.mkdir(
        parents=True,
        exist_ok=True,
    )

    J = load_measurement_jacobian()
    gradients = load_operational_gradients()

    summary = []
    directions = []

    print()
    print("M2.5-B2 JOINT OPERATIONAL BOUNDS")
    print("=" * 78)
    print("Measurement Jacobian:", J.shape)
    print("Joint dimension:", DIM)
    print("Terminal time:", TIME, "s")
    print("Solver:", SOLVER)

    for name, _, _, units in OBJECTIVES:
        g, _ = gradients[name]

        print()
        print(name, f"({units})")
        print("-" * 78)

        for epsilon in EPSILONS:
            for radius in BOUNDARY_RADII:

                result = solve_bound(
                    J,
                    g,
                    epsilon,
                    radius,
                )

                key = (
                    f"{name}_eps{epsilon:g}"
                    f"_rb{radius:g}"
                )

                summary.append({
                    "objective": name,
                    "units": units,
                    "epsilon": epsilon,
                    "initial_radius": INITIAL_RADIUS,
                    "boundary_radius": radius,
                    "bound": result["objective"],
                    "measurement_norm":
                        result["measurement_norm"],
                    "initial_norm":
                        result["initial_norm"],
                    "boundary_norm":
                        result["boundary_norm"],
                    "solver_status":
                        result["status"],
                    "direction_key": key,
                })

                directions.append(
                    (key, result["direction"])
                )

                print(
                    f"eps={epsilon:4g} "
                    f"rb={radius:4g} "
                    f"bound={result['objective']:.12g} "
                    f"||Jd||={result['measurement_norm']:.6g} "
                    f"||dx||={result['initial_norm']:.6g} "
                    f"||db||={result['boundary_norm']:.6g}"
                )

    result_df = pd.DataFrame(summary)

    # Write only after all optimizations and checks succeed.
    result_df.to_csv(
        M25 / "joint_operational_bounds.csv",
        index=False,
    )

    np.savez_compressed(
        M25 / "joint_operational_directions.npz",
        **dict(directions),
    )

    print()
    print("=" * 78)
    print("COMPARISON WITH FIXED BOUNDARIES")
    print("=" * 78)

    for name, _, _, units in OBJECTIVES:
        subset = result_df[
            result_df.objective == name
        ]

        for epsilon in EPSILONS:
            rows = subset[
                subset.epsilon == epsilon
            ].set_index("boundary_radius")

            baseline = float(rows.loc[0.0, "bound"])

            for radius in (1.0, 5.0):
                bound = float(
                    rows.loc[radius, "bound"]
                )

                ratio = (
                    bound / baseline
                    if baseline > 0.0
                    else float("nan")
                )

                print(
                    f"{name:18s} "
                    f"eps={epsilon:4g} "
                    f"rb={radius:3g} "
                    f"ratio={ratio:.8f}"
                )

    print()
    print("Saved:")
    print(" ", M25 / "joint_operational_bounds.csv")
    print(" ", M25 / "joint_operational_directions.npz")


if __name__ == "__main__":
    main()
