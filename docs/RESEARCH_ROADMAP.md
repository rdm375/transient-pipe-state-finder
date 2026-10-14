# State Finder Research Roadmap

Status reviewed after M28-A4.

This roadmap distinguishes implemented research infrastructure,
completed numerical experiments, and capabilities that still
require scientific or operational validation.

## Phase 1 — Forward and derivative foundations

Status: Implemented and regression-tested.

- Shared C++ isothermal pipeline forward solver.
- Implicit time stepping and conservation diagnostics.
- Tangent-linear and discrete-adjoint operations.
- Derivative verification and matrix-free products.

Remaining:
- Continue numerical conditioning and roundoff investigations.
- Maintain parity with the pinned forward-solver revision.

## Phase 2 — Inverse problem and observability

Status: Substantial research infrastructure implemented.

- Scaled least-squares reconstruction.
- Regularization and uncertainty-weighted objectives.
- Initial-state and predictive observability studies.
- Information spectra and matrix-free analysis.
- Nonlinear constrained operational sensitivity experiments.

Remaining:
- Reconcile weakly observable directions with operational
  forecast sensitivities.
- Distinguish statistical regularization from solver
  preconditioning and damping.
- Quantify nonlinear approximation limits.

## Phase 3 / M27 — Correlated nonlinear reconstruction

Status: Experimental campaign completed;
numerical-analysis questions remain open.

- Correlated observation noise.
- Original and calibration-aware estimators.
- 32 paired realizations in each of two scenarios.
- Optimizer, dense-versus-PCG, and spectrum diagnostics.

Outstanding:
- Nonlinear optimizer convergence.
- Realistic initialization instead of truth-based baseline.
- Weak-direction behavior and numerical conditioning.
- Calibration parameter identifiability.

## Phase 4 / M28 — Operational forecasting

Status: A1–A4 completed for controlled synthetic scenarios.

- A1: Exact trajectory continuation regression.
- A2: Single reconstructed-state forecast.
- A3: 64 original-estimator forecasts.
- A4: 128 paired original/calibrated forecasts.
- Forecast inventory and pressure diagnostics.
- Near-roundoff numerical mass conservation.

Outstanding:
- Forecast-oriented tangent/adjoint sensitivity.
- Error-mode decomposition and decay analysis.
- Uncertain future pressure and demand histories.
- Realistic initialization and operating scenarios.
- Predictive uncertainty intervals and coverage.
- Independent validation.

## Phase 5 — Robust operational state estimation

Status: Not demonstrated.

Candidate milestones:
- Operationally available initialization.
- Joint state and boundary uncertainty propagation.
- Forecast-relevant regularization and estimation.
- Calibration uncertainty and sensor-fault robustness.
- Multiple operating regimes and perturbation classes.
- Verified prediction intervals and coverage tests.

## Release engineering

Status: Research software; not production-ready.

- Clean-checkout reproducibility and dependency pinning.
- CI and regression automation.
- Stable interfaces and input validation.
- Experiment provenance and retained reference datasets.
- User-facing diagnostics and documentation.
- Explicit release acceptance criteria.

## Immediate priorities

1. Close and preserve M28-A1–A4 artifacts.
2. Revisit the unfinished M27 numerical-analysis questions.
3. Develop forecast-oriented tangent/adjoint sensitivities.
4. Introduce uncertain future boundary histories.
5. Test reconstruction from operationally plausible initial guesses.

Do not equate completion of an experiment with validation of a
production estimator.
