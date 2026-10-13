# M1.5b — Long-Trajectory Discrete Adjoint Verification

## Status

**PASS — numerical verification completed**

The discrete adjoint was evaluated over 80, 800, and 8,000
time steps using the validated isothermal transient pipeline
solver.

The experiments used:

- 100 spatial cells
- 201 independent initial-state variables
- 15-second time steps
- Theta-method parameter 0.65
- Synthetic observations generated from a reference trajectory
- Perturbed initial conditions
- Independent central finite-difference directional derivatives

The forward simulator was not modified.

## 1. Long-trajectory results

| Quantity | 80 steps | 800 steps | 8,000 steps |
|---|---:|---:|---:|
| Objective | 5.341593424 | 5.423974479 | 5.423974479 |
| Scaled gradient norm | 148.168308 | 150.235260 | 150.235260 |
| Directional derivative | 277.543178 | 273.713946 | 273.713946 |
| Maximum linear backward error | 1.669e-16 | 1.907e-16 | 2.007e-16 |
| Maximum mass defect | 4.638e-10 | 5.583e-10 | 6.226e-8 |
| Forward runtime (s) | 0.007350 | 0.034175 | 0.102980 |
| Adjoint runtime (s) | 0.012098 | 0.068678 | 0.409235 |

Execution times are individual measurements, not statistically
characterized performance benchmarks.

The reported adjoint runtime includes its forward trajectory
evaluation.

## 2. Finite-difference verification

The adjoint directional derivative at 800 steps was:

    273.7139456209311

At 8,000 steps:

    273.7139456209308

The difference is approximately 3e-13.

Central finite differences exhibited the expected second-order
convergence over the truncation-dominated step-size range.

Selected normalized discrepancies:

| Step size | 800 steps | 8,000 steps |
|---|---:|---:|
| 1e-2 | 4.104e-3 | 4.104e-3 |
| 1e-3 | 4.104e-5 | 4.104e-5 |
| 1e-4 | 4.103e-7 | 4.103e-7 |
| 1e-5 | 4.248e-9 | 4.248e-9 |
| 3e-6 | 1.586e-9 | 1.586e-9 |
| 1e-6 | 1.586e-9 | 1.586e-9 |
| 3e-7 | 3.672e-8 | 3.672e-8 |
| 3e-8 | 1.973e-7 | 1.973e-7 |

The minimum observed normalized discrepancy was approximately
1.586e-9, occurring near step sizes 1e-6 to 3e-6.

Below this range, finite-difference error increased, consistent
with cancellation and numerical noise.

This minimum discrepancy is not a direct bound on the intrinsic
error of the analytical adjoint.

## 3. Observation-window decomposition

Both reference and candidate forward simulations completed all
8,000 requested time steps.

The objective was decomposed into nonoverlapping observation
windows:

| Window | Objective contribution |
|---|---:|
| Steps 0–80 | 5.3415934241208767 |
| Steps 81–800 | 0.082381054966959938 |
| Steps 801–8000 | 1.1947123543218275e-15 |

Total:

    5.423974479087843

Approximately 98.48% of the objective accumulated during the
first 80 steps.

The contribution after step 800 was negligible for this
particular candidate/reference pair.

### Observation differences

| Window | Maximum inlet-flow difference (kg/s) | Maximum outlet-pressure difference (Pa) |
|---|---:|---:|
| 0–80 | 0.8808600068 | 681.4229683 |
| 81–800 | 0.0840533386 | 146.4945760 |
| 801–8000 | 1.02356239e-8 | 1.81337819e-5 |

At the final time:

- Maximum pressure-state difference: 1.80676579e-7 Pa
- Maximum flow-state difference: 1.02701847e-10 kg/s
- Inlet-flow difference: 1.02701847e-10 kg/s

The reference and candidate trajectories converged toward the
same forced response to the resolution of this experiment.

These observations support decay of initial-condition
sensitivity for the tested perturbation.

They do not establish the rank, conditioning, or complete
information content of the observation Jacobian.

## 4. Linear-solve stability

Maximum normwise backward errors for the adjoint transpose
solves remained near binary64 machine precision:

- 80 steps: 1.66891052e-16
- 800 steps: 1.90662644e-16
- 8,000 steps: 2.00740424e-16

These measurements support backward stability of the linear
solves.

Backward stability alone does not establish corresponding
forward-error bounds without conditioning information.

## 5. Conservation diagnostic

At 8,000 steps:

    Reference maximum mass defect:
        6.2160353309081984e-8

    Candidate maximum mass defect:
        6.2263055156108749e-8

The reported maximum mass defect increased with trajectory
length.

Its units, normalization, and dependence on trajectory length
require further characterization.

Potential causes include floating-point cancellation,
diagnostic accumulation, and nonlinear solver residuals.

No specific cause has yet been established.

This remains an open numerical investigation.

The validated forward solver was not modified.

## 6. Regression testing

The existing CTest suite completed successfully:

    14/14 tests passed
    0 failures

The long-trajectory benchmarks are separate from the fast
CTest suite.

## 7. Reproducibility

Source files:

- tests/cpp/adjoint_long_trajectory.cpp
- tests/cpp/observation_windows.cpp

Results:

- benchmarks/results-m15b/adjoint-80.csv
- benchmarks/results-m15b/adjoint-800.csv
- benchmarks/results-m15b/adjoint-8000.csv
- benchmarks/results-m15b/adjoint-800-extended.csv
- benchmarks/results-m15b/adjoint-8000-extended.csv
- benchmarks/results-m15b/observation-windows.log

Corresponding execution logs are retained in the same
results directory.

## 8. Conclusions

The complete discrete adjoint passed long-trajectory numerical
verification through 8,000 time steps.

The gradient exhibited second-order agreement with independent
central finite differences until numerical noise dominated.

The transpose linear solves remained backward stable.

The objective-window decomposition confirmed that nearly all
measured misfit accumulated during the early transient for the
tested initial-state perturbation.

The experiments establish a verified numerical foundation for
subsequent inverse-state optimization.

## 9. Next milestone

M1.6 — Nonlinear inverse-state optimization.

Initial priorities:

1. Introduce a scaled optimization-variable representation.
2. Implement a deterministic optimization driver.
3. Start with synthetic observations and known initial states.
4. Verify objective reduction and gradient consistency.
5. Investigate identifiability and regularization.
6. Preserve conservation diagnostics independently of the
   optimization objective.
