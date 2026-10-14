# Transient Pipe State Finder

Research software for physics-constrained reconstruction and uncertainty analysis of the initial pressure and mass-flow state of a single isothermal gas pipeline from boundary observations.

**Status:** research implementation; **not** a production-ready StateFinder release. The CMake project version `0.1.0` is not a release certification. See [release acceptance criteria](docs/RELEASE_CANDIDATE_CHECKLIST.md).

## Mathematical problem

Given prescribed inlet pressure and outlet mass flow over an observation interval, estimate the unknown initial internal state using measured inlet mass flow and outlet pressure. The forward model is a staggered-grid, theta-method discretization with boundary half-cells. Its numerical conservation defect is evaluated separately from the physical difference between inlet and outlet mass flows.

The inverse problem is generally ill-conditioned or non-identifiable: small observation residuals do not imply accurate reconstruction of every internal-state coordinate. The research therefore examines sensitivity operators, information spectra, regularization, and the uncertainty of predicted operational quantities.

Read [Mathematical foundations](docs/MATHEMATICAL_FOUNDATIONS.md) for definitions and derivations at an advanced undergraduate mathematics level.

## Implemented research areas

- Forward-model parity and mass-conservation validation.
- Initial-state optimization and regularized reconstruction.
- Tangent-linear and adjoint sensitivity verification; matrix-free analysis.
- Observability, information spectra, and modal/predictive analysis.
- Synthetic measurement and exact-state-recovery experiments (M2.3).
- Linearized operational uncertainty bounds and nonlinear feasible adversarial witnesses (M2.4).

These are **research capabilities**, not a guarantee that all production requirements are satisfied. In particular, M2.4 nonlinear witnesses are not certificates of global optimality.

## Build and validation

The build currently expects the sibling repository `../isothermal-pipe-sim` (override with `-DPIPE_SIM_SOURCE_DIR=/path/to/isothermal-pipe-sim`). CMake 3.20+, a C++20 compiler, and LAPACK are required. Python analysis scripts have additional dependencies.

```bash
cmake -S . -B build -DPIPE_SIM_SOURCE_DIR="$(realpath ../isothermal-pipe-sim)"
cmake --build build -j 4
ctest --test-dir build --output-on-failure
```

See [Reproducibility](docs/REPRODUCIBILITY.md) for dependency and benchmark provenance instructions. This build command has not been independently verified on a fresh checkout.

## Project roadmap

See [Development roadmap](docs/DEVELOPMENT_ROADMAP.md). Research milestones and production release gates are intentionally distinct.

## License

GNU Affero General Public License v3.0 **only** (`AGPL-3.0-only`), without a linking exception.
