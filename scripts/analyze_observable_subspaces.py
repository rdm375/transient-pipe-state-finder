#!/usr/bin/env python3
"""Analyze analytic initial-state observability subspaces."""

from pathlib import Path
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path("benchmarks/results-m16b-analytic")
OUTPUT = ROOT / "subspace-analysis"

OUTPUT.mkdir(parents=True, exist_ok=True)

N = 100
D = 2 * N + 1

HORIZONS = (80, 100, 200, 400, 800)

THRESHOLDS = (1e-6, 1e-8, 1e-10, 1e-12)

# Relative to the largest singular value.
REFERENCE_THRESHOLD = 1e-10


def load_matrix(horizon):
    path = ROOT / f"H_{horizon}.csv"

    H = np.loadtxt(path, delimiter=",")

    assert H.shape == (2 * (horizon + 1), D)
    assert np.isfinite(H).all()

    return H


def full_svd(H):
    # full_matrices=True is important for H_80:
    # it has 201 columns but only 162 rows.
    return np.linalg.svd(H, full_matrices=True)


def effective_rank(s, sigma_max, threshold):
    return int(np.count_nonzero(s > threshold * sigma_max))


def component_energy(v):
    pressure = float(np.dot(v[:N], v[:N]))
    flow = float(np.dot(v[N:2*N], v[N:2*N]))
    inlet = float(v[-1] ** 2)

    return pressure, flow, inlet


def frequency_summary(v):
    """Normalized high-frequency energy and dominant DCT frequency.

    Uses an orthonormal DCT-II implemented via NumPy's FFT.
    Frequency indices 0..N-1 describe increasingly oscillatory
    spatial patterns. This is a descriptive diagnostic, not a
    physical modal decomposition.
    """
    x = np.asarray(v, dtype=float)
    size = x.size

    extended = np.concatenate((x, x[::-1]))
    transform = np.fft.rfft(extended)

    coefficients = np.real(
        transform[:size] *
        np.exp(-1j * np.pi * np.arange(size) / (2 * size))
    )

    coefficients[0] /= np.sqrt(2.0)

    energy = coefficients**2
    total = float(energy.sum())

    if total == 0:
        return 0.0, 0

    high_frequency = float(
        energy[size // 2:].sum() / total
    )

    dominant = int(np.argmax(energy))

    return high_frequency, dominant


def projector_distance(Va, Vb):
    """Frobenius distance between equal-rank orthogonal projectors."""
    overlap = Va.T @ Vb

    value = (
        2 * Va.shape[1] -
        2 * np.linalg.norm(overlap, "fro") ** 2
    )

    return np.sqrt(max(0.0, value))


def principal_angles(Va, Vb):
    """Principal angles in degrees between two column subspaces."""
    overlap = Va.T @ Vb

    singular_values = np.linalg.svd(
        overlap, compute_uv=False
    )

    singular_values = np.clip(singular_values, 0.0, 1.0)

    return np.degrees(np.arccos(singular_values))


def save_spectrum_plot(spectra):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for horizon, s in spectra.items():
        # Plot only explicitly returned singular values.
        # The remaining 39 singular values for H_80
        # correspond to its structural column nullspace.
        normalized = s / s[0]

        ax.semilogy(
            np.arange(1, len(s) + 1),
            np.maximum(normalized, 1e-25),
            label=f"{horizon} steps",
        )

    for threshold in THRESHOLDS:
        ax.axhline(
            threshold,
            linestyle=":",
            linewidth=0.8,
        )

    ax.set_xlabel("Singular-value index")
    ax.set_ylabel("Singular value / largest singular value")
    ax.set_title("Analytic initial-state sensitivity spectra")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT / "singular_spectra.png", dpi=180)
    plt.close(fig)


def save_mode_plot(vectors, indices, title, filename):
    fig, axes = plt.subplots(
        len(indices), 2,
        figsize=(11, 2.5 * len(indices)),
        squeeze=False,
    )

    for row, index in enumerate(indices):
        v = vectors[:, index]

        axes[row, 0].plot(
            np.arange(1, N + 1),
            v[:N],
        )

        axes[row, 1].plot(
            np.arange(N) + 0.5,
            v[N:2*N],
        )

        axes[row, 0].set_ylabel(f"Mode {index + 1}")
        axes[row, 0].grid(alpha=0.3)
        axes[row, 1].grid(alpha=0.3)

    axes[0, 0].set_title("Pressure components")
    axes[0, 1].set_title("Internal-flow components")

    axes[-1, 0].set_xlabel("Pressure node")
    axes[-1, 1].set_xlabel("Flow face")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=180)
    plt.close(fig)


def main():
    decompositions = {}
    spectra = {}

    print("=== EFFECTIVE RANK ===")

    rank_rows = []

    for horizon in HORIZONS:
        H = load_matrix(horizon)

        U, s, Vt = full_svd(H)

        decompositions[horizon] = (s, Vt)
        spectra[horizon] = s

        ranks = [
            effective_rank(s, s[0], threshold)
            for threshold in THRESHOLDS
        ]

        rank_rows.append([horizon, *ranks])

        print(
            f"steps={horizon:3d} "
            f"ranks={ranks} "
            f"sigma_max={s[0]:.12e}"
        )

    with open(
        OUTPUT / "effective_ranks.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "horizon",
            *[f"rank_{t:.0e}" for t in THRESHOLDS],
        ])

        writer.writerows(rank_rows)

    save_spectrum_plot(spectra)

    print("\n=== OBSERVABLE SUBSPACE COMPARISON ===")

    s80, Vt80 = decompositions[80]
    s800, Vt800 = decompositions[800]

    comparison_rows = []

    for threshold in THRESHOLDS:
        r80 = effective_rank(s80, s80[0], threshold)
        r800 = effective_rank(s800, s800[0], threshold)

        V80 = Vt80[:r80, :].T
        V800 = Vt800[:r800, :].T

        angles = principal_angles(V80, V800)

        max_angle = float(np.max(angles))
        mean_angle = float(np.mean(angles))

        distance = (
            projector_distance(V80, V800)
            if r80 == r800
            else float("nan")
        )

        comparison_rows.append([
            threshold,
            r80,
            r800,
            max_angle,
            mean_angle,
            distance,
        ])

        print(
            f"threshold={threshold:.0e} "
            f"rank80={r80} "
            f"rank800={r800} "
            f"max_angle={max_angle:.6e} deg "
            f"mean_angle={mean_angle:.6e} deg "
            f"projector_distance={distance:.6e}"
        )

    with open(
        OUTPUT / "subspace_comparison.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "threshold",
            "rank_80",
            "rank_800",
            "max_principal_angle_deg",
            "mean_principal_angle_deg",
            "projector_frobenius_distance",
        ])

        writer.writerows(comparison_rows)

    print("\n=== 800-STEP MODE STRUCTURE ===")

    r = effective_rank(
        s800, s800[0], REFERENCE_THRESHOLD
    )

    # Selected modes spanning the resolved and unresolved
    # portions of the spectrum.
    indices = sorted(set([
        0,
        1,
        9,
        min(29, r - 1),
        max(0, r - 1),
        min(D - 1, r),
        min(D - 1, r + 10),
        D - 1,
    ]))

    mode_rows = []

    for index in indices:
        v = Vt800[index, :]

        pressure, flow, inlet = component_energy(v)

        pressure_hf, pressure_freq = frequency_summary(
            v[:N]
        )

        flow_hf, flow_freq = frequency_summary(
            v[N:2*N]
        )

        sigma = (
            s800[index]
            if index < len(s800)
            else 0.0
        )

        mode_rows.append([
            index + 1,
            sigma,
            pressure,
            flow,
            inlet,
            pressure_hf,
            flow_hf,
            pressure_freq,
            flow_freq,
        ])

        print(
            f"mode={index+1:3d} "
            f"sigma={sigma:.5e} "
            f"pressure={pressure:.4f} "
            f"flow={flow:.4f} "
            f"inlet={inlet:.4f} "
            f"pressure_HF={pressure_hf:.4f} "
            f"flow_HF={flow_hf:.4f}"
        )

    with open(
        OUTPUT / "mode_structure.csv",
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "mode",
            "singular_value",
            "pressure_energy",
            "internal_flow_energy",
            "inlet_flow_energy",
            "pressure_high_frequency_fraction",
            "flow_high_frequency_fraction",
            "pressure_dominant_frequency",
            "flow_dominant_frequency",
        ])

        writer.writerows(mode_rows)

    save_mode_plot(
        Vt800.T,
        indices,
        "Selected right singular vectors, 800 steps",
        "selected_modes_800.png",
    )

    # Save the resolved basis for subsequent reconstruction.
    #
    # Shape: 201 x r.
    V_resolved = Vt800[:r, :].T

    np.save(
        OUTPUT / "resolved_basis_800.npy",
        V_resolved,
    )

    np.save(
        OUTPUT / "singular_values_800.npy",
        s800,
    )

    print(
        f"\nResolved basis: {V_resolved.shape}"
    )

    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()
