# M1.6a — Scaled L-BFGS Initial-State Optimization

## Scope

This milestone introduces nonlinear optimization of the initial
state of the isothermal transient-pipe model.

The verified forward simulator and discrete adjoint are unchanged.

## Optimization method

- Limited-memory BFGS
- Armijo backtracking line search
- Objective-only line-search evaluations
- Dimensionless optimization coordinates
- Adjoint-derived gradients
- Rejection of failed numerical evaluations
- Curvature safeguards
- Iteration and evaluation diagnostics

## Initial-state parameterization

For 100 spatial cells, the optimization vector contains:

- 100 internal pressure variables
- 100 internal staggered mass-flow variables
- 1 inlet mass-flow variable

Total: 201 variables.

The prescribed inlet pressure is excluded.

The coordinate transformation is

    x_initial = x_baseline + D * xi

where pressure components are scaled by 1e5 Pa and
mass-flow components by 10 kg/s.

The transformed gradient is

    grad_xi J = D^T grad_x_initial J.

## Boundary measurements

The observation data model explicitly represents:

1. Inlet pressure
2. Inlet mass flow
3. Outlet pressure
4. Outlet mass flow

Current roles:

| Channel | Role |
|---|---|
| Inlet pressure | Prescribed input |
| Inlet mass flow | Objective residual |
| Outlet pressure | Objective residual |
| Outlet mass flow | Prescribed input |

The current forward model uses prescribed inlet pressure and
prescribed outlet mass flow.

Only the complementary response measurements contribute to
the present least-squares objective.

A compatibility conversion preserves the verified objective
and adjoint implementations.

## Synthetic reconstruction experiment

- Pipe length: 100 km
- Spatial cells: 100
- Time step: 15 s
- Time steps: 80
- Observation horizon: 20 minutes
- Inlet pressure: 8 MPa
- Outlet mass flow: 100 to 105 kg/s over 600 s
- Objective scales: 1 kg/s and 1e4 Pa

The initial perturbation reproduces the M1.5b experiment.

## Verified numerical results

Independent quadratic optimizer test:

- Exact minimizer: (2, -3)
- Final objective: approximately 2.47e-24
- Convergence: 8 iterations

Pipeline reconstruction:

| Metric | Initial | 10 iterations | 30 iterations |
|---|---:|---:|---:|
| Objective | 5.341593424 | 0.013841799 | 0.002638449 |
| Scaled gradient norm | 148.168308 | 2.518223 | 0.248103 |
| Pressure RMSE (Pa) | 711.833728 | 198.541087 | 164.303483 |
| Internal flow RMSE (kg/s) | 0.352249073 | 0.349992314 | 0.349018959 |
| Inlet flow error (kg/s) | 0.700000 | 0.059090 | -0.000292 |

The scaled directional derivative check gave a normalized
error of approximately 1.9455e-8.

Neither the 10-iteration nor 30-iteration experiment met
the configured convergence tolerances. Both stopped at
their iteration limits.

## Interpretation

The optimizer substantially reduces boundary-response
misfit and improves pressure reconstruction.

Internal mass-flow reconstruction improves much less.

This behavior may reflect limited observability, poor
conditioning, or insufficient optimization iterations.
These explanations have not yet been separated experimentally.

For 80 steps with both response channels sampled at t=0,
there are 162 scalar residual measurements for 201
unconstrained initial-state variables.

The corresponding local sensitivity matrix therefore
cannot have full column rank.

## Verification

The existing numerical regression suite remains intact.

The four-channel observation model has its own unit test.

The reconstruction executable is excluded from the
regular CTest suite and run separately.

No changes were made to the forward simulator or
discrete adjoint.

## Deferred work

- Longer optimization runs
- Initial-state compatibility analysis
- Sensitivity singular-value analysis
- Longer observation windows
- Boundary measurement uncertainty
- General measured boundary-input histories
- Regularization and reduced initial-state parameterizations
