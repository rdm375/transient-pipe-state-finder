# M27 — Phase 3 Nonlinear Reconstruction
## Interim Technical Report and Phase 4 Handoff

**Project:** Transient Linepack Estimation and Short-Horizon Operational Risk
**Repository:** transient-pipe-state-finder
**Branch:** feature/shared-trajectory-matrix-free
**Reference commit:** 409e7fd
**Date:** 2026-10-14
**Status:** M27-A and M27-B complete; M27-C statistical validation pending.

---

## 1. Executive Summary

Phase 3 develops and validates nonlinear reconstruction of transient
pipeline states and boundary histories from noisy measurements.

The implementation supports:

- Joint reconstruction of initial pipeline state and boundary histories.
- Temporally correlated measurement errors.
- Matrix-free tangent-linear and adjoint derivatives.
- Matrix-free Gauss-Newton optimization.
- Explicit pressure calibration uncertainty.
- Gaussian calibration priors with configurable correlation.
- Terminal linepack estimation and conservation diagnostics.

The original reconstruction contains 163 unknowns.

The calibration-aware reconstruction contains 165 unknowns, adding
inlet and outlet pressure calibration offsets.

The numerical implementations have passed all 51 regression tests.

For the corrected synthetic measurement realization, the original
24-iteration reconstruction produced a terminal inventory error of
approximately -222.20 kg.

The calibration-aware reconstruction with a 2 kPa independent
calibration prior produced an error of approximately -213.05 kg.

A calibration-prior sensitivity study demonstrated that the
reconstructed inventory depends on calibration assumptions, although
the response is not monotonic.

These results establish numerical functionality and sensitivity for
the tested scenarios. They do not yet establish statistical estimator
accuracy or coverage.

Independent-noise statistical validation remains M27-C.

---

## 2. Physical Reference Problem

The reference system is an isothermal transient natural-gas pipeline.

### 2.1 Geometry and operating conditions

| Quantity | Value |
|---|---:|
| Pipeline length | 50,000 m |
| Diameter | 0.6 m |
| Spatial cells | 20 |
| Time step | 10 s |
| Simulation steps | 60 |
| Observation interval | 0–600 s |
| Initial inlet pressure | 8 MPa |
| Initial mass flow | 100 kg/s |
| Maximum outlet demand | 110 kg/s |
| Compressibility factor | 0.9 |
| Time integration parameter theta | 0.65 |

The outlet mass-flow demand increases smoothly from 100 to
110 kg/s between 120 and 240 seconds.

The reference terminal inventory is:

    M_true = 812846.35761952039 kg

### 2.2 Conservation

The forward discretization satisfies the discrete inventory balance:

    (M[k+1] - M[k]) / dt
        = m_in_theta[k] - m_out_theta[k]

where the boundary mass flows are evaluated using the same
theta-weighting as the forward discretization.

The inventory balance is checked independently for reconstructed
trajectories.

---

## 3. Observation Model

Measurements are collected at 61 time points using four channels:

1. Inlet pressure.
2. Inlet mass flow.
3. Outlet pressure.
4. Outlet mass flow.

There are 244 scalar measurements.

The engineering normalization scales are:

    inlet pressure:   10000 Pa
    inlet mass flow:      1 kg/s
    outlet pressure:  10000 Pa
    outlet mass flow:     1 kg/s

The normalized measurement standard deviation is 0.5.

The temporal covariance uses an exponential correlation model with:

    tau = 120 s

The corrected synthetic noise realization is stored in:

    benchmarks/results-m27/
      correlated_measurement_realization_sigma0p5.csv

The reference realization has a truth-state measurement objective of:

    F_truth = 129.34864039289769

---

## 4. Joint Reconstruction

### 4.1 Unknown variables

The original inverse problem contains 163 variables:

- 41 initial-state variables.
- 61 inlet-pressure boundary-history variables.
- 61 outlet-flow boundary-history variables.

The initial inlet pressure is prescribed by the boundary history.

The calibrated problem adds two pressure calibration variables:

    b_in
    b_out

Both are expressed in kPa.

The augmented dimension is therefore:

    163 + 2 = 165

### 4.2 Nonlinear measurement objective

Let x denote the scaled physical reconstruction variables.

Let r(x) be the engineering-normalized measurement residual.

Let C be the temporal measurement covariance.

The original objective is:

    F(x) = 0.5 * r(x)^T * C^-1 * r(x)

### 4.3 Calibration-aware objective

Let b contain the inlet and outlet calibration offsets in kPa.

Let B map calibration offsets into the normalized pressure
measurement channels.

The augmented objective is:

    F(x,b)
      = 0.5 * [r(x) + B*b]^T * C^-1 * [r(x) + B*b]
      + 0.5 * b^T * C_b^-1 * b

The calibration covariance is:

    C_b = sigma_cal^2 * [[1, rho],
                         [rho, 1]]

where:

    sigma_cal > 0
    -1 < rho < 1

The zero-calibration case is represented by the original
163-variable problem, not by a singular calibration covariance.

### 4.4 Derivatives

The augmented gradient is:

    grad_x F = J^T * C^-1 * [r + B*b]

    grad_b F = B^T * C^-1 * [r + B*b]
               + C_b^-1 * b

The Gauss-Newton operator applies:

    H_GN * [v_x, v_b]

using the tangent-linear operator, covariance inverse,
calibration response, and adjoint.

The implementation does not require explicit assembly of the
full augmented Jacobian or normal matrix.

---

## 5. Numerical Verification

The completed M27-B2 tests verified:

- Augmented objective evaluation.
- Finite-difference gradient agreement.
- Matrix-free operator symmetry.
- Gauss-Newton curvature consistency.
- Calibration-prior correlation.
- Inclusion and exclusion of initial observations.
- Consistency with the original model at zero calibration offset.

Representative maximum observed errors:

| Check | Maximum relative error |
|---|---:|
| Gradient | 1.14e-7 |
| Operator symmetry | 4.99e-16 |
| Curvature consistency | 7.18e-16 |

The complete regression suite passed:

    51/51 tests

No forward-solver changes were required for M27-B.

---

## 6. M27-A — Original Nonlinear Reconstruction

The corrected measurement realization was reconstructed with
6, 12, and 24 Gauss-Newton iterations.

| Iterations | Objective | Inventory error (kg) |
|---:|---:|---:|
| 6 | 45.2712013853 | -215.4146671103 |
| 12 | 45.2332324210 | -217.0198868880 |
| 24 | 45.1593510670 | -222.1999823270 |

The measurement objective decreases with additional iterations.

For this particular realization, the absolute terminal inventory
error increases between 6 and 24 iterations.

This establishes that improved measurement fit does not
necessarily imply improved terminal inventory accuracy.

Earlier M27-A analysis also established that the inventory
response is sensitive to weakly identifiable reconstruction
directions.

No arbitrary singular-vector truncation was adopted.

---

## 7. M27-B3 — Calibrated Nonlinear Reconstruction

The calibrated reconstruction used:

    sigma_cal = 2 kPa
    rho = 0

The same corrected measurement realization and initial guess
were used as in M27-A.

### 7.1 Results

| Iterations | Total objective | Inlet bias (kPa) | Outlet bias (kPa) | Inventory error (kg) |
|---:|---:|---:|---:|---:|
| 6 | 45.2672016504 | -0.660668 | +0.425298 | -213.906910 |
| 12 | 45.2131009955 | -0.653000 | +0.438566 | -212.157977 |
| 24 | 45.1291532190 | -0.630464 | +0.446974 | -213.045505 |

At 24 iterations:

    measurement objective = 45.0544943646
    calibration penalty   =  0.0746588544
    total objective       = 45.1291532190

The calibrated reconstruction reduced the absolute inventory
error by approximately 9.15 kg relative to the original
24-iteration reconstruction.

This corresponds to an approximately 4.1% reduction in
absolute error for this realization.

The synthetic observations contain no deliberately injected
systematic calibration error.

The inferred offsets must therefore not be interpreted as
recovered physical calibration errors.

### 7.2 Conservation

The maximum reported mass-balance defects were:

| Iterations | Maximum defect (kg) |
|---:|---:|
| 6 | 3.56e-10 |
| 12 | 4.01e-10 |
| 24 | 4.13e-10 |

These results preserve the previously established discrete
conservation behavior.

### 7.3 Convergence qualification

All three runs terminated at their configured maximum
iteration count.

They did not satisfy the optimizer convergence criterion.

The 24-iteration results are therefore finite-iteration
reconstructions, not demonstrated converged optima.

---

## 8. M27-B4 — Calibration-Prior Sensitivity

Seven calibration-prior configurations were evaluated using
the same corrected measurement realization and a fixed
24-iteration budget.

| Sigma (kPa) | Rho | Objective | Inlet bias (kPa) | Outlet bias (kPa) | Inventory error (kg) |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.0 | 45.195524 | -0.045503 | +0.032237 | -218.296553 |
| 1.0 | 0.0 | 45.150456 | -0.173584 | +0.123858 | -218.276571 |
| 2.0 | 0.0 | 45.129153 | -0.630464 | +0.446974 | -213.045505 |
| 5.0 | 0.0 | 44.944274 | -1.320994 | +0.986582 | -216.097187 |
| 2.0 | +0.5 | 45.134811 | -0.433086 | +0.146378 | -206.900707 |
| 2.0 | +0.9 | 45.199216 | -0.257113 | -0.134432 | -199.299688 |
| 2.0 | -0.5 | 45.106411 | -0.727070 | +0.652233 | -220.106875 |

### 8.1 Calibration sensitivity

Increasing the independent calibration prior standard deviation
allows larger inferred calibration offsets.

The response is not monotonic in terminal inventory error.

### 8.2 Correlation sensitivity

At sigma_cal = 2 kPa, varying rho from -0.5 to +0.9 changes
the reconstructed terminal inventory error by approximately
20.8 kg.

Positive calibration correlation changes the preferred
combination of inlet and outlet calibration offsets.

### 8.3 Objective versus inventory accuracy

The sigma_cal = 5 kPa case has the lowest reported objective.

It does not have the smallest absolute terminal inventory error.

Consequently, measurement fit alone is insufficient for
selecting a linepack estimator.

### 8.4 Interpretation limits

The calibration-prior sweep uses one noise realization.

The optimization runs have fixed iteration budgets.

The results demonstrate sensitivity under the tested
conditions but do not establish statistical estimator
performance or fully converged parameter sensitivity.

---

## 9. Phase 2 Uncertainty Context

Previous M26 linearized uncertainty analysis established
the following terminal inventory standard deviations:

| Calibration assumption | Inventory sigma (kg) |
|---|---:|
| Fixed calibration | 247.118 |
| 2 kPa, rho = 0 | 290.758 |
| 2 kPa, rho = +0.5 | 310.262 |
| 2 kPa, rho = +0.9 | 325.017 |

Introducing calibration uncertainty increases the estimated
inventory uncertainty.

This is compatible with the observation that calibration
freedom may improve the fit for an individual noise realization.

An individual reconstruction error and an uncertainty
standard deviation measure different quantities.

---

## 10. Outstanding M27-C Statistical Validation

M27-C must evaluate the estimators over independent
measurement-noise realizations.

### 10.1 C1 — Zero true calibration bias

Use:

    true inlet calibration bias  = 0
    true outlet calibration bias = 0

Generate independent correlated measurement-noise realizations.

For each realization, run both:

- The original 163-variable reconstruction.
- The calibrated 165-variable reconstruction.

Use paired observations and matched solver budgets.

### 10.2 C2 — Injected calibration uncertainty

Generate true calibration offsets from the specified
calibration covariance.

Inject the offsets consistently with the observation-model
sign convention.

Run both reconstruction methods on the same observations.

Verify the sign convention before launching the campaign.

### 10.3 Statistical quantities

For terminal inventory error:

    e_i = M_estimated_i - M_true

Report:

    mean error
    RMSE
    median absolute error
    empirical error quantiles
    solver failure rate
    convergence status
    gradient norms
    maximum mass-conservation defects

Use paired comparisons between estimators.

For absolute-error differences:

    Delta_i = abs(e_calibrated_i) - abs(e_original_i)

Report the sample mean and its uncertainty.

Do not infer statistical superiority from the sign of
a single realization's error difference.

### 10.4 Initial campaign

Begin with 32 independent noise realizations for each
scenario.

Record random seeds and experimental configuration.

Treat this as an initial statistical campaign rather
than definitive uncertainty-coverage validation.

---

## 11. Phase 3 Closure Criteria

Phase 3 can be formally closed after:

1. M27-C1 and M27-C2 statistical experiments complete.
2. Random seeds and experimental configurations are retained.
3. Paired inventory-error statistics are reported.
4. Numerical failures and optimizer termination reasons are reported.
5. Mass-conservation diagnostics remain satisfactory.
6. Convergence limitations are explicitly documented.
7. The reconstruction configuration carried into Phase 4 is justified.
8. All regression tests pass.
9. Code, tests, and selected results are committed and pushed.

Until then, this report remains interim.

---

## 12. Phase 4 — Operational Prediction and Risk

The next project phase will use reconstructed states and
boundary histories as inputs to short-horizon transient
pipeline predictions.

### 12.1 Forecast horizons

Evaluate:

    600 s
    1200 s
    1800 s

### 12.2 Operational quantities

Primary diagnostics:

- Terminal and time-dependent linepack.
- Inlet and outlet mass-flow imbalance.
- Rate of linepack change.
- Minimum pipeline pressure.
- Pressure-limit proximity.
- Mass-conservation defect.
- Sensitivity to reconstructed initial conditions.
- Sensitivity to uncertain future boundary conditions.

### 12.3 Forecast validation

Use controlled synthetic scenarios with known reference
trajectories.

Separate:

- Reconstruction error.
- Future boundary-condition uncertainty.
- Forward numerical error.
- Calibration uncertainty.

Avoid combining these sources into a single unexplained
forecast-error statistic.

### 12.4 Initial forecast scenarios

Start with:

1. Constant inlet pressure and constant outlet demand.
2. Smooth outlet-demand increase.
3. Smooth outlet-demand decrease.
4. Band-limited outlet-demand perturbations.
5. Inlet-pressure perturbations.
6. Combined inlet-pressure and outlet-demand disturbances.

Maintain the existing isothermal single-pipe scope.

### 12.5 Operational risk

Risk metrics should be tied to explicitly defined
operational limits.

Examples include:

    minimum pressure below specified threshold
    linepack below specified threshold
    excessive inventory depletion rate

Thresholds must be declared as scenario assumptions.

They must not be presented as universal pipeline
operating limits.

---

## 13. Reproducibility

### 13.1 Reference implementation

Repository:

    https://github.com/rdm375/transient-pipe-state-finder

Reference commit:

    409e7fd7ce4e36cfca9b4279a38fe8e82576c099

Branch:

    feature/shared-trajectory-matrix-free

### 13.2 Calibration implementation

Key files:

    include/state_finder/pressure_calibration.hpp

    include/state_finder/calibrated_joint_gauss_newton.hpp

    apps/m27_calibrated_nonlinear_reconstruction.cpp

    tests/cpp/m27_pressure_calibration_test.cpp

    tests/cpp/m27_calibrated_joint_test.cpp

### 13.3 Numerical results

M27-B3:

    benchmarks/results-m27/calibrated-b3/

M27-B4:

    benchmarks/results-m27/calibrated-b4/

The committed CSV files contain reconstructed scaled
solutions and experiment summaries.

Some execution logs are not tracked.

### 13.4 Build and test

    cmake -S . -B build \
        -DPIPE_SIM_SOURCE_DIR="$HOME/Sync/isothermal-pipe-sim"

    cmake --build build -j 4

    ctest --test-dir build --output-on-failure

The last reported complete regression run passed 51/51 tests.

---

## 14. Conclusions

M27-A and M27-B establish a functioning nonlinear pipeline
state-reconstruction framework with correlated measurement
uncertainty and pressure calibration priors.

The principal findings are:

1. The nonlinear reconstruction is operational.
2. Matrix-free derivatives and Gauss-Newton operators are verified.
3. Discrete mass conservation remains accurate.
4. Pressure calibration uncertainty affects inferred linepack.
5. Calibration-prior correlation affects the reconstructed solution.
6. Improved measurement fit does not guarantee improved
   inventory accuracy.
7. Single-realization improvements cannot establish
   statistical estimator superiority.
8. The current fixed-iteration results are not demonstrated
   converged optima.

The next required milestone is M27-C statistical validation.

After statistical validation and Phase 3 closure, the project
will proceed to short-horizon operational prediction and
risk assessment.

---

## 15. Roadmap Status

| Milestone | Status |
|---|---|
| Phase 1 — Numerical foundation | Complete |
| Phase 2 — Measurement uncertainty | Complete |
| M27-A — Nonlinear reconstruction | Complete |
| M27-B1 — Calibration observation model | Complete |
| M27-B2 — Augmented derivatives and operator | Complete |
| M27-B3 — Calibrated nonlinear reconstruction | Complete |
| M27-B4 — Calibration-prior sensitivity | Complete |
| M27-C — Statistical validation | Pending |
| M27-D — Phase 3 closure | Pending |
| Phase 4 — Operational prediction | Next phase |
| Phase 5 — Engineering robustness | Pending |
| Phase 6 — Final technical report | Pending |
