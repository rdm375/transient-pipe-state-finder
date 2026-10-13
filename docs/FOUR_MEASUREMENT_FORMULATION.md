# Four-Measurement Transient Pipeline State Estimation

## Governing requirement

The primary engineering problem uses four measured boundary histories:

1. Inlet pressure
2. Inlet mass flow
3. Outlet pressure
4. Outlet mass flow

All four are uncertain field measurements.

All four must contribute to the inverse objective.

No field measurement is treated as exact.

## Forward model

The transient forward solver uses:

- Inlet pressure as a prescribed boundary condition.
- Outlet mass flow as a prescribed boundary condition.

These are mathematical boundary conditions, not declarations
that the corresponding field measurements are exact.

## Inverse model

The inverse solver estimates:

- The initial internal pressure distribution.
- The initial internal flow distribution.
- The initial inlet mass flow.
- The inlet pressure history.
- The outlet mass flow history.

The initial inlet pressure is constrained to equal the first
sample of the estimated inlet pressure history.

The remaining two boundary histories are computed by the
transient forward model.

## Measurement objective

At every included observation time, the objective contains
four uncertainty-normalized residuals, ordered:

    inlet_pressure
    inlet_flow
    outlet_pressure
    outlet_flow

The measurement objective is one half the sum of the squares
of these residuals.

Regularization may be added to stabilize the reconstruction,
but it must not replace any of the four measurement channels.

## Physical constraints

The reconstructed trajectory must satisfy the discrete
transient governing equations.

Numerical mass conservation must be verified independently
of the differences between measured and reconstructed flows.

Physical inlet/outlet flow imbalance is not the same as a
numerical conservation defect.

## Required numerical verification

- Four residuals per included observation time.
- Tangent-linear directional derivative verification.
- Adjoint transpose verification.
- Objective gradient verification.
- Matrix-free Gauss-Newton operator verification.
- Reconstruction from four noisy measurement histories.
- Initial-state and boundary-history reconstruction errors.
- Measurement residuals reported separately by channel.
- Discrete mass conservation diagnostics.

## Regression policy

The legacy two-residual formulation may remain for comparison
and regression testing.

It is not the primary engineering formulation.

Do not silently substitute exact measured boundary histories
for estimated boundary histories.
