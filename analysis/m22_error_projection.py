#!/usr/bin/env python3
"""M2.2b: Project M2.1 reconstruction errors onto information modes."""

import csv
import math
from pathlib import Path

ROOT = Path("benchmarks")
M21 = ROOT / "results-m21"
M22 = ROOT / "results-m22"

N = 100
DIM = 2 * N + 1
PRESSURE_SCALE = 1.0e5
FLOW_SCALE = 10.0

METHODS = ("unregularized", "tikhonov", "hybrid")
THRESHOLDS = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)


def read_matrix(path):
    with path.open(newline="") as f:
        return [
            [float(value) for value in row]
            for row in csv.reader(f)
        ]


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def norm2(a):
    return dot(a, a)


def state_error(method):
    path = M21 / f"gn_{method}_initial_state.csv"

    pressure = [None] * N
    flow = [None] * N
    inlet = None

    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            field = row["field"]
            index = int(row["index"])
            error = float(row["recovered_error"])

            if field == "pressure_pa":
                if index == 0:
                    # Prescribed inlet pressure is not a state DOF.
                    continue

                if not 1 <= index <= N:
                    raise ValueError("Invalid pressure index")

                pressure[index - 1] = error / PRESSURE_SCALE

            elif field == "flow_kg_s":
                if not 0 <= index < N:
                    raise ValueError("Invalid flow index")

                flow[index] = error / FLOW_SCALE

            elif field == "inlet_flow_kg_s":
                if inlet is not None:
                    raise ValueError("Duplicate inlet flow")

                inlet = error / FLOW_SCALE

            else:
                raise ValueError(f"Unexpected field: {field}")

    if (
        any(x is None for x in pressure)
        or any(x is None for x in flow)
        or inlet is None
    ):
        raise ValueError(f"Incomplete state: {path}")

    result = pressure + flow + [inlet]

    if len(result) != DIM or not all(map(math.isfinite, result)):
        raise ValueError("Invalid state error vector")

    return result


def main():
    M22.mkdir(parents=True, exist_ok=True)

    with (M22 / "spectrum.csv").open(newline="") as f:
        eigenvalues = [
            float(row["eigenvalue"])
            for row in csv.DictReader(f)
        ]

    with (M22 / "eigenvectors.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(eigenvalues) != DIM or len(rows) != DIM:
        raise ValueError("Unexpected eigensystem dimensions")

    for i, row in enumerate(rows):
        if int(row["coordinate"]) != i:
            raise ValueError("Unexpected eigenvector coordinate")

    # V[mode][coordinate]
    V = [
        [
            float(rows[i][f"mode_{j}"])
            for i in range(DIM)
        ]
        for j in range(DIM)
    ]

    K = read_matrix(M22 / "K.csv")

    if not K or any(len(row) != DIM for row in K):
        raise ValueError("Invalid sensitivity matrix")

    lambda_max = eigenvalues[-1]

    if not math.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError("Invalid maximum eigenvalue")

    # Verify orthonormality before using the eigensystem.
    max_orthogonality_error = 0.0

    for i in range(DIM):
        for j in range(i + 1):
            expected = 1.0 if i == j else 0.0

            error = abs(dot(V[i], V[j]) - expected)

            max_orthogonality_error = max(
                max_orthogonality_error, error
            )

    if max_orthogonality_error > 1e-9:
        raise ValueError(
            "Eigenvector orthogonality check failed: "
            f"{max_orthogonality_error}"
        )

    summary = []
    modal_rows = []

    print(f"state_dimension={DIM}")
    print(f"observation_dimension={len(K)}")
    print(f"maximum_eigenvalue={lambda_max:.12e}")
    print(
        "maximum_orthogonality_error="
        f"{max_orthogonality_error:.12e}"
    )

    for method in METHODS:
        e = state_error(method)

        coefficients = [dot(v, e) for v in V]

        total = norm2(e)
        modal_total = math.fsum(a*a for a in coefficients)

        parseval_error = abs(total - modal_total) / max(
            total, 1e-300
        )

        if parseval_error > 1e-9:
            raise ValueError(
                f"Parseval check failed for {method}: "
                f"{parseval_error}"
            )

        # Direct observation sensitivity: ||K e||^2.
        Ke = [dot(row, e) for row in K]
        visible2 = norm2(Ke)

        # Spectral identity: e^T G e = sum lambda_j a_j^2.
        spectral_visible2 = math.fsum(
            lam*a*a
            for lam, a in zip(eigenvalues, coefficients)
        )

        identity_error = abs(
            visible2 - spectral_visible2
        ) / max(visible2, 1e-300)

        if identity_error > 1e-7:
            raise ValueError(
                f"Information identity failed for {method}: "
                f"{identity_error}"
            )

        for j, (lam, a) in enumerate(
            zip(eigenvalues, coefficients)
        ):
            modal_rows.append({
                "method": method,
                "mode": j,
                "eigenvalue": lam,
                "coefficient": a,
                "state_error_energy": a*a,
                "observation_error_energy": lam*a*a,
            })

        for threshold in THRESHOLDS:
            cutoff = threshold*lambda_max

            weak2 = math.fsum(
                a*a
                for lam, a in zip(eigenvalues, coefficients)
                if lam <= cutoff
            )

            strong2 = math.fsum(
                a*a
                for lam, a in zip(eigenvalues, coefficients)
                if lam > cutoff
            )

            rank = sum(
                lam > cutoff for lam in eigenvalues
            )

            summary.append({
                "method": method,
                "threshold": threshold,
                "rank": rank,
                "nullity": DIM-rank,
                "scaled_error_norm": math.sqrt(total),
                "weak_error_fraction": weak2/total,
                "strong_error_fraction": strong2/total,
                "linearized_observation_norm": math.sqrt(visible2),
                "parseval_relative_error": parseval_error,
                "information_identity_relative_error": identity_error,
            })

        print(f"\nmethod={method}")
        print(f"scaled_error_norm={math.sqrt(total):.12e}")
        print(
            "linearized_observation_norm="
            f"{math.sqrt(visible2):.12e}"
        )
        print(
            "information_identity_relative_error="
            f"{identity_error:.12e}"
        )

        for threshold in THRESHOLDS:
            item = next(
                x for x in summary
                if x["method"] == method
                and x["threshold"] == threshold
            )

            print(
                f"threshold={threshold:.0e} "
                f"rank={item['rank']} "
                f"weak_fraction="
                f"{item['weak_error_fraction']:.8f}"
            )

    summary_path = M22 / "error_projection_summary.csv"
    modal_path = M22 / "error_projection_modes.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)

    with modal_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(modal_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(modal_rows)

    print(f"\nsummary_csv={summary_path}")
    print(f"modal_csv={modal_path}")
    print("PASS M2.2b error projection")


if __name__ == "__main__":
    main()
