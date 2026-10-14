# Reproducibility and clean-checkout build

## Prerequisites

- Linux, CMake 3.20+, C++20 compiler, and LAPACK development libraries.
- Sibling checkout of `isothermal-pipe-sim`, or set `PIPE_SIM_SOURCE_DIR` explicitly.
- Python >=3.11 for analysis; the package declares NumPy, SciPy, and JAX dependencies. The separate `requirements.txt` also lists pandas and matplotlib. Dependency specifications are **not yet locked**.

## Build and test

```bash
cmake -S . -B build -DPIPE_SIM_SOURCE_DIR="$(realpath ../isothermal-pipe-sim)"
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

This is a documented build procedure, not a claim of a fresh-machine validation. A clean-checkout CI job remains to be implemented.

## Provenance to capture for each reference experiment

Record the state-finder Git commit and dirty-tree status, the forward-simulator Git commit, CMake configuration, compiler and flags, LAPACK implementation, Python/package versions, input files, random seeds, numerical tolerances, and SHA256 checksums of results.

The forward-simulator revision is **not yet pinned** in this repository. Until it is, two clones of the state-finder repository can produce different builds against different sibling simulator revisions.

## Research data

Benchmark directories contain both canonical reference data and experiment outputs. Do not regenerate tracked reference CSV files during a routine test without checking their diffs. Compare floating-point outputs using documented physical and numerical tolerances, not unexamined byte-for-byte equality.

## Validation hierarchy

1. Build and unit tests.
2. Conservation, parity, finite-difference, and adjoint-identity checks.
3. Synthetic inverse-problem recovery and observability studies.
4. Nonlinear replay and constraint-feasibility checks.
5. Independent model validation and production acceptance (not yet established).
