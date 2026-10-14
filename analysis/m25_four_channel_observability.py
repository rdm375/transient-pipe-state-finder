#!/usr/bin/env python3
"""M2.5-A: Four-channel measurement-accuracy observability study.

Uses the existing M24 joint measurement Jacobian.

No forward-model recomputation is required.

Measurement scales are treated as relative weighting assumptions,
not automatically as calibrated instrument standard deviations.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE = Path(
    "benchmarks/results-m24/joint_measurement_jacobian.csv"
)

OUTPUT = Path("benchmarks/results-m25")

NX = 41
NB = 122
ROWS = 244
COLS = NX + NB

RTOL = 1.0e-10

CHANNELS = (
    "inlet_pressure",
    "inlet_flow",
    "outlet_pressure",
    "outlet_flow",
)

BASELINE_SCALES = np.array(
    [1.0e4, 1.0, 1.0e4, 1.0],
    dtype=float,
)

# Multipliers applied to the baseline measurement scales.
#
# A multiplier of 10 reduces that channel's weight by 10.
# A multiplier of 0.1 increases its weight by 10.
#
# These are sensitivity scenarios, not calibrated sensor models.

CONFIGURATIONS = {
    "A_baseline": (1.0, 1.0, 1.0, 1.0),
    "B_inlet_pressure_x10": (10.0, 1.0, 1.0, 1.0),
    "C_outlet_flow_x10": (1.0, 1.0, 1.0, 10.0),
    "D_response_channels_x10": (1.0, 10.0, 10.0, 1.0),
    "E_input_channels_x0p1": (0.1, 1.0, 1.0, 0.1),
}


def load_jacobian():
    J = np.loadtxt(
        SOURCE,
        delimiter=",",
        skiprows=1,
    )

    if J.shape != (ROWS, COLS):
        raise ValueError(
            f"Expected {(ROWS, COLS)}, got {J.shape}"
        )

    if not np.isfinite(J).all():
        raise ValueError("Nonfinite Jacobian entries")

    return J


def singular_report(matrix):
    singular = np.linalg.svd(
        matrix,
        compute_uv=False,
    )

    threshold = RTOL * singular[0]

    rank = int(
        np.count_nonzero(singular > threshold)
    )

    smallest = float(singular[-1])

    condition = (
        float(singular[0] / smallest)
        if smallest > 0.0
        else float("inf")
    )

    return singular, rank, condition


def profile_boundary(J):
    A = J[:, :NX]
    B = J[:, NX:]

    U, singular, _ = np.linalg.svd(
        B,
        full_matrices=False,
    )

    threshold = RTOL * singular[0]

    rank = int(
        np.count_nonzero(singular > threshold)
    )

    Q = U[:, :rank]

    # Orthogonal projection away from the resolved
    # boundary-history sensitivity subspace.
    profiled = A - Q @ (Q.T @ A)

    return profiled, rank


def save_spectrum(name, matrix):
    singular, rank, condition = singular_report(matrix)

    spectrum = pd.DataFrame({
        "index": np.arange(len(singular)),
        "singular_value": singular,
        "relative_singular_value": (
            singular / singular[0]
        ),
    })

    spectrum.to_csv(
        OUTPUT / f"{name}.csv",
        index=False,
    )

    return {
        "rank": rank,
        "largest": float(singular[0]),
        "smallest": float(singular[-1]),
        "condition": condition,
    }


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    J0 = load_jacobian()

    records = []

    print()
    print("M2.5-A FOUR-CHANNEL OBSERVABILITY")
    print("=" * 72)

    print("Jacobian:", J0.shape)
    print("Initial variables:", NX)
    print("Boundary variables:", NB)
    print("Observation times:", ROWS // 4)
    print("Numerical rank tolerance:", RTOL)

    for name, multipliers in CONFIGURATIONS.items():

        multipliers = np.asarray(
            multipliers,
            dtype=float,
        )

        if np.any(multipliers <= 0.0):
            raise ValueError(
                "Measurement multipliers must be positive"
            )

        # Existing Jacobian rows are already normalized
        # by BASELINE_SCALES.
        #
        # Changing scale s -> multiplier*s requires
        # dividing the corresponding Jacobian rows
        # by that multiplier.

        row_weights = np.tile(
            1.0 / multipliers,
            ROWS // 4,
        )

        J = row_weights[:, None] * J0

        A = J[:, :NX]

        profiled, boundary_rank = profile_boundary(J)

        joint_stats = save_spectrum(
            f"{name}_joint_spectrum",
            J,
        )

        fixed_stats = save_spectrum(
            f"{name}_fixed_boundary_spectrum",
            A,
        )

        profiled_stats = save_spectrum(
            f"{name}_profiled_spectrum",
            profiled,
        )

        # Orthogonal projection must not increase
        # the Frobenius norm.
        norm_fixed = np.linalg.norm(A, "fro")
        norm_profiled = np.linalg.norm(
            profiled,
            "fro",
        )

        if norm_profiled > norm_fixed * (1.0 + 1e-10):
            raise RuntimeError(
                f"Projection increased norm: {name}"
            )

        record = {
            "configuration": name,
            "inlet_pressure_multiplier": multipliers[0],
            "inlet_flow_multiplier": multipliers[1],
            "outlet_pressure_multiplier": multipliers[2],
            "outlet_flow_multiplier": multipliers[3],
            "joint_rank": joint_stats["rank"],
            "boundary_rank": boundary_rank,
            "fixed_initial_rank": fixed_stats["rank"],
            "profiled_initial_rank": profiled_stats["rank"],
            "joint_largest": joint_stats["largest"],
            "joint_smallest": joint_stats["smallest"],
            "fixed_initial_smallest": fixed_stats["smallest"],
            "profiled_initial_largest": profiled_stats["largest"],
            "profiled_initial_smallest": profiled_stats["smallest"],
            "profiled_initial_condition": profiled_stats["condition"],
            "fixed_frobenius_norm": norm_fixed,
            "profiled_frobenius_norm": norm_profiled,
        }

        records.append(record)

        print()
        print(name)
        print("-" * 72)
        print("Multipliers:", multipliers)
        print("Joint rank:", joint_stats["rank"])
        print("Boundary rank:", boundary_rank)
        print("Fixed initial rank:", fixed_stats["rank"])
        print("Profiled initial rank:", profiled_stats["rank"])
        print(
            "Weakest profiled singular value:",
            f'{profiled_stats["smallest"]:.12e}',
        )
        print(
            "Profiled condition number:",
            f'{profiled_stats["condition"]:.12e}',
        )

    summary = pd.DataFrame(records)

    summary.to_csv(
        OUTPUT / "four_channel_observability_summary.csv",
        index=False,
    )

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    print(
        summary[
            [
                "configuration",
                "joint_rank",
                "boundary_rank",
                "profiled_initial_rank",
                "profiled_initial_smallest",
                "profiled_initial_condition",
            ]
        ].to_string(index=False)
    )

    print()
    print("Saved results to:", OUTPUT)


if __name__ == "__main__":
    main()
