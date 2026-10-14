# StateFinder Engineering Roadmap

## Project objective

Demonstrate that StateFinder can reconstruct the internal state of a
transient gas pipeline from imperfect boundary measurements, quantify
operational uncertainty, and predict whether the pipeline can safely
support future downstream demand.

The principal engineering application is:

**Transient Linepack Estimation and Short-Horizon Operational Risk**

This roadmap governs the transition from numerical-method validation
to a reproducible engineering case study.

## Reference configuration

The initial engineering benchmark uses:

- Pipeline length: 50 km
- Diameter: 0.6 m
- Nominal pressure: approximately 8 MPa
- Nominal mass flow: approximately 100 kg/s
- Spatial discretization: 20 cells
- Observation horizon: 600 s
- Observation interval: 10 s
- Four measurement channels:
  - Inlet pressure
  - Inlet mass flow
  - Outlet pressure
  - Outlet mass flow
- Prescribed inputs:
  - Inlet-pressure history
  - Outlet-mass-flow history
- Equation of state: constant compressibility factor, Z = 0.9

These parameters define the initial numerical benchmark, not a
fully calibrated operational pipeline model.

## Phase 1 — Numerical validation

Status: IN PROGRESS

### Completed

- Nonlinear transient forward simulation.
- Discrete adjoint validation against finite differences.
- Tangent-linear validation against discrete adjoints.
- Joint initial-state and boundary-history sensitivities.
- Four-channel measurement Jacobian.
- Observability and numerical-rank analysis.
- Measurement-weighting sensitivity.
- Linearized operational uncertainty optimization.
- Boundary-history uncertainty diagnostics.
- Fixed-boundary nonlinear uncertainty witnesses.

### Remaining: M2.5-C1

Nonlinear replay of joint uncertainty directions.

Required validation:

1. Reconstruct the perturbed initial state.
2. Reconstruct both prescribed boundary histories.
3. Verify initial/boundary compatibility.
4. Solve the nonlinear trajectory.
5. Evaluate all four nonlinear measurement residual channels.
6. Compare nonlinear and tangent-linear operational deviations.
7. Verify pressure positivity and physical admissibility.
8. Verify global discrete mass conservation.
9. Distinguish raw replay from strictly feasible nonlinear witnesses.

Completion criterion:

The nonlinear validity and limitations of the M25 uncertainty
results are established and documented.

## Phase 2 — Measurement uncertainty

Status: PENDING

Replace engineering normalization scales with explicitly documented
measurement uncertainty assumptions.

Investigate:

- Pressure measurement uncertainty.
- Mass-flow measurement uncertainty.
- Independent measurement errors.
- Temporally correlated errors.
- Systematic sensor bias.
- Uncertainty in measured boundary histories.

Distinguish deterministic uncertainty sets from statistical
confidence regions.

Completion criterion:

Every uncertainty constraint has a defensible physical
interpretation.

## Phase 3 — Engineering case study

Status: PENDING

Scenario:

A pipeline operator observes an increase in downstream demand
and must determine whether available linepack is sufficient
to maintain acceptable downstream pressure.

Compare:

A. Exactly known boundary histories.

B. Uncertain but measured boundary histories with independent
   measurement errors.

C. Uncertain but measured boundary histories with temporal
   correlation and sensor bias.

Engineering outputs:

- Estimated internal pressure distribution.
- Estimated internal mass-flow distribution.
- Estimated total linepack.
- Measurement-conditioned inventory uncertainty.
- Sensitivity to measurement quality.

Completion criterion:

Quantify how measurement quality and boundary uncertainty
affect estimated linepack.

## Phase 4 — Operational prediction

Status: PENDING

Separate observation and prediction:

- Observation horizon: 600 s.
- Initial prediction horizon: 1800 s.

Use measurements through 600 s to constrain the current state.

Propagate admissible states forward under a specified
future operating scenario.

Primary outputs:

- Future downstream pressure.
- Minimum downstream pressure over the prediction horizon.
- Future pipeline inventory.
- Available linepack.
- Time to an operating-pressure threshold.

Completion criterion:

Produce physically admissible uncertainty envelopes for
future operational quantities.

## Phase 5 — Engineering validation and reporting

Status: PENDING

Required work:

- Spatial refinement study.
- Temporal refinement study.
- Measurement uncertainty sensitivity.
- Boundary-history uncertainty sensitivity.
- Nonlinear validation of worst-case predictions.
- Conservation verification.
- Physical-admissibility verification.
- Reproducible benchmark artifacts.
- Comparison against a defined baseline method.
- Engineering interpretation and limitations.

Completion criterion:

A reproducible, technically reviewable engineering case study.

## Scope controls

1. Complete the current milestone before expanding scope.
2. Defer unrelated GPU, TPU, surrogate, and numerical-method work.
3. Preserve existing benchmark artifacts.
4. Require reproducible tests and documented assumptions.
5. Commit completed milestones to version control.
6. Do not interpret linearized bounds as certified nonlinear bounds.
7. Do not interpret engineering normalizers as calibrated sensor noise.
8. Keep operational decision quality as the ultimate objective.

## Current milestone

**M2.5-C1: Nonlinear Joint-Direction Replay**

Inputs:

- M24 four-channel measurement Jacobian.
- M25 joint operational sensitivity matrix.
- M25 optimized joint directions.

Initial cases:

- Outlet pressure, epsilon = 1, boundary radius = 1.
- Inlet flow, epsilon = 1, boundary radius = 1.
- Inventory, epsilon = 1, boundary radius = 1.

The next milestone is Phase 2: realistic measurement uncertainty.

## Definition of project success

StateFinder must demonstrate that imperfect boundary measurements
can be used to constrain the internal state of a transient gas
pipeline and produce useful, physically validated uncertainty
estimates for future operational decisions.

The final case study must distinguish:

- Measurement consistency.
- State-estimation uncertainty.
- Boundary-history uncertainty.
- Numerical approximation error.
- Model-form uncertainty.
- Future operational uncertainty.

The project is not complete merely because the inverse solver
converges or its derivatives are accurate.
