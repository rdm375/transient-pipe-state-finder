# M27 — Phase 3 Closure Report

**Status:** Closed as finite-iteration numerical and statistical validation  
**Date:** 2026-10-14  
**Branch:** feature/shared-trajectory-matrix-free

## 1. Scope

Phase 3 implemented nonlinear reconstruction of transient pipeline
states and boundary histories from temporally correlated measurements.

Two estimators were evaluated:

- Original correlated-measurement reconstruction: 163 variables.
- Pressure-calibration-aware reconstruction: 165 variables.

The forward model remains the existing isothermal-pipe-sim solver.

## 2. Statistical Validation: M27-C1 and M27-C2

The campaign used 32 independent measurement-noise realizations
per scenario, with paired original and calibrated reconstructions.

Each reconstruction used a maximum of 24 Gauss–Newton iterations.

| Scenario | Estimator | Inventory bias (kg) | RMSE (kg) | MAE (kg) |
|---|---|---:|---:|---:|
| C1: zero calibration bias | Original | +24.844 | 241.213 | 198.833 |
| C1: zero calibration bias | Calibrated | +23.682 | 225.467 | 185.689 |
| C2: injected calibration bias | Original | +95.294 | 306.310 | 267.714 |
| C2: injected calibration bias | Calibrated | +85.756 | 285.852 | 250.901 |

Paired absolute-error comparisons:

- C1: calibrated minus original MAE = -13.144 kg;
  paired bootstrap 95% CI [-19.871, -7.177] kg.
- C2: calibrated minus original MAE = -16.813 kg;
  paired bootstrap 95% CI [-24.034, -10.027] kg.

The calibrated estimator had lower absolute inventory error
in 25/32 C1 realizations and 27/32 C2 realizations.

These results support an improvement for the tested
finite-iteration procedures. They do not establish superiority
of fully converged estimators.

## 3. Calibration Recovery

For C1, the true calibration offsets were zero.

Estimated calibration RMSE:

- Inlet: 0.4819 kPa.
- Outlet: 0.4178 kPa.

For C2, the injected calibration offsets were sampled
independently with 2 kPa standard deviation.

Estimated calibration-error RMSE:

- Inlet: 2.1584 kPa.
- Outlet: 1.8249 kPa.

Calibration offsets were not reliably recovered.
Improved inventory estimation must not be interpreted
as successful identification of physical sensor biases.

## 4. Optimization Convergence

All 128 statistical reconstructions terminated at the
24-iteration limit without satisfying the gradient tolerance.

The subsequent M27-C3 audit extended selected reconstructions
to 48, 96, and 192 iterations.

The objectives continued to decrease, but no tested case
satisfied the configured stationarity criterion.

Estimator rankings for individual realizations changed
with iteration budget.

Consequently, the statistical campaign characterizes
a fixed-budget reconstruction procedure.

## 5. Linear-Solver Diagnostics

M27-C4 added optimizer diagnostics.

M27-C5 tested the existing joint-regularization
preconditioner without changing the objective.

For the selected C1 original-estimator realization,
preconditioning reduced PCG iterations only modestly.

M27-C6 introduced an optional dense Cholesky reference solver.

At 192 iterations:

| Metric | Matrix-free PCG | Dense Cholesky |
|---|---:|---:|
| Final objective | 52.418381 | 50.671739 |
| Gradient norm | 0.088392 | 0.189511 |
| Damping retries | 61 | 0 |
| Operator calls | 36,242 | 31,680 |
| Runtime (s) | 32.44 | 28.07 |
| Inventory error (kg) | +105.41 | +93.19 |

Dense Cholesky improved objective reduction and avoided
linear-solver damping retries in this experiment.

Neither solver demonstrated convergence to stationarity.

The dense implementation is retained as a diagnostic
reference for small reconstruction problems.

## 6. Observability Diagnostics

M27-C7 evaluated the undamped correlated-measurement
Gauss–Newton information matrix at the synthetic truth,
the PCG reconstruction, and the dense reconstruction.

At relative eigenvalue threshold 1e-10, the effective
rank was 148/163 at all three points.

The ten weakest eigenvectors were concentrated almost
entirely in initial-state coordinates.

Their local terminal-inventory sensitivities were small,
but this does not establish small total inventory uncertainty.

Earlier project milestones already investigated
observability and information spectra. No additional
spectral-analysis milestone is required for Phase 3 closure.

## 7. Conservation

The statistical campaign's maximum reported discrete
mass-conservation defect was below 5.87e-10 kg.

The extended optimizer experiments also preserved
mass conservation near floating-point roundoff.

## 8. Interpretation and Limitations

1. All statistical reconstructions began at the exact
   synthetic reference state in scaled coordinates.
   Operational initialization performance was not tested.
2. Statistical comparisons apply to 24-iteration estimators.
3. Fully converged estimator performance remains unknown.
4. Calibration parameters were weakly identifiable.
5. Measurement-objective improvement does not guarantee
   improved inventory accuracy.
6. The dense solver was tested as a reference method,
   not established as a scalable production solver.
7. The spectral diagnostics confirm weak observability
   but do not provide validated nonlinear uncertainty coverage.

## 9. Reproducibility

Principal artifacts:

- benchmarks/scripts/m27_c_statistical_validation.py
- benchmarks/results-m27/statistical-c/
- benchmarks/results-m27/convergence-c3/
- benchmarks/results-m27/convergence-c4/
- benchmarks/results-m27/convergence-c5/
- benchmarks/results-m27/convergence-c6/
- benchmarks/results-m27/spectrum-c7/

The preceding interim report is retained at:

- docs/M27_PHASE3_INTERIM_REPORT.md

The final validation run passed all 51 regression tests.

## 10. Phase 4 Handoff

Phase 4 will evaluate short-horizon operational predictions
at 600, 1200, and 1800 seconds.

The original 163-variable reconstruction will serve as
the reference estimator.

The calibration-aware estimator will remain a sensitivity
comparison until its convergence and calibration-recovery
limitations are resolved.

Primary engineering quantities:

- Terminal and time-dependent linepack.
- Inlet/outlet mass-flow imbalance.
- Linepack change rate.
- Minimum pressure and pressure-limit proximity.
- Discrete mass-conservation defect.

Reconstruction error, future boundary-condition uncertainty,
and forward-model numerical error will be reported separately.

## 11. Closure Decision

M27-C is closed as a completed finite-iteration statistical
validation campaign.

M27-D is closed as the Phase 3 documentation and handoff.

These closure decisions do not assert convergence of the
nonlinear estimators or validated operational forecasting
performance.

The next project milestone is Phase 4.
