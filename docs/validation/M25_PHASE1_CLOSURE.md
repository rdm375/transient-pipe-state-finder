# M2.5 Phase 1 Numerical Validation Closure

## Status

PASS — October 14, 2026

The planned StateFinder numerical-method validation phase
is complete.

This does not constitute validation against field measurements
or certification of nonlinear global uncertainty bounds.

## Reference problem

- Single pipeline, 50 km
- Diameter 0.6 m
- Nominal inlet pressure 8 MPa
- Nominal mass flow 100 kg/s
- 20 spatial cells
- 60 timesteps of 10 s
- Constant compressibility factor Z = 0.9
- Four measurement channels
- 41 initial-state variables
- 122 boundary-history variables
- 163 total joint variables
- 244 normalized measurement components

## Nonlinear joint uncertainty witnesses

At normalized measurement tolerance epsilon = 1
and boundary radius = 1:

| Quantity | Linear | Nonlinear | Nonlinear residual norm |
|---|---:|---:|---:|
| Outlet pressure (Pa) | 2252.276227 | 2246.615033 | 0.997930 |
| Inlet flow (kg/s) | 0.991030 | 0.990920 | 0.999935 |
| Inventory (kg) | 116.641329 | 115.666568 | 0.992479 |

All three nonlinear witnesses satisfy the specified
measurement constraint.

Minimum pressures are approximately 6.85 MPa.

Maximum observed discrete mass-conservation defect
is approximately 5.08e-10 kg.

Initial inlet-pressure compatibility error is zero.

These are feasible nonlinear witnesses, not certified
global nonlinear optima.

## Independent boundary-history derivative validation

Eight directional tests:

1. Inlet pressure at 0 s.
2. Inlet pressure at 300 s.
3. Inlet pressure at 600 s.
4. Outlet flow at 0 s.
5. Outlet flow at 300 s.
6. Outlet flow at 600 s.
7. Mixed boundary-history perturbation.
8. Mixed joint initial-state/boundary perturbation.

Centered nonlinear finite differences were evaluated
at h = 1e-2, 1e-3, and 1e-4.

All 244 measurement components and three terminal
operational quantities were compared with tangent-linear
predictions.

Maximum of the eight best relative measurement errors:
approximately 3.18e-10.

Maximum of the eight best relative operational errors:
approximately 2.71e-9.

The step-size dependence is consistent with finite-
difference truncation and floating-point limitations.

## Interpretation

The tested joint tangent-linear sensitivities agree
with independently evaluated nonlinear finite differences.

The nonlinear replay confirms that the M25 linearized
uncertainty amplification persists in feasible nonlinear
trajectories.

The experiments support the numerical implementation,
but do not establish calibrated statistical uncertainty
or globally optimal nonlinear operational bounds.

## Reproducibility

Primary scripts:

- analysis/m25_joint_boundary_fd_validation.py
- analysis/m25_nonlinear_replay.py

Primary executable:

- state_finder_m25_joint_nonlinear_replay

Primary results:

- benchmarks/results-m25/joint_boundary_fd_validation.csv
- benchmarks/results-m25/nonlinear_joint_replay.csv

## Remaining engineering work

1. Establish physically defensible sensor uncertainty.
2. Construct the linepack estimation case study.
3. Extend uncertainty into future operational prediction.
4. Perform spatial and temporal refinement.
5. Assess model-form uncertainty.
6. Produce the final engineering report.
