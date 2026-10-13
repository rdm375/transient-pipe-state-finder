# M1.5a — Numerical Derivative Verification

## Status

PASS — 9/9 regression tests.

## High-precision physics verification

The Swamee–Jain friction source and its analytic derivatives
were compared against independent Boost.Multiprecision
50-decimal-digit reference calculations.

The tests cover positive and negative mass flows.

| Statistic | Relative error |
|---|---:|
| Comparisons | 24 |
| Minimum | 1.835157600807386e-17 |
| Mean | 1.262434501112989e-16 |
| Maximum | 2.697277835656381e-16 |

The maximum discrepancy is approximately 1.21 times
IEEE 754 binary64 machine epsilon.

These results demonstrate working-precision agreement for
the tested friction source and derivative evaluations.

They do not independently validate every assembled
residual Jacobian entry.

## Componentwise discrete-adjoint verification

Fourteen componentwise experiments were performed,
covering seven selected initial-state variables with
initial observations both included and excluded.

Each experiment used nine central finite-difference
step sizes.

| Statistic | Best-achieved discrepancy |
|---|---:|
| Minimum | 3.601703980047966e-14 |
| Mean | 1.458698662664508e-09 |
| Maximum | 4.923447176501755e-09 |

The reported discrepancy is normalized by
max(1, abs(scaled analytic directional derivative)).

The initial inlet-flow component achieved agreement
near 4e-14 in the tested cases.

## Previous complete-adjoint verification

The end-to-end directional-gradient test passed
six configurations, with best relative discrepancies
between approximately 6.95e-10 and 4.47e-09.

The maximum reported LAPACK transpose-solve
normwise relative backward error was approximately
1.512e-16.

## Interpretation

The evidence supports:

1. Working-precision agreement for tested analytic
   friction derivatives.
2. Near-machine-precision normwise backward errors
   for the tested transpose solves.
3. Correct discrete-adjoint sensitivity propagation
   within the resolution of independent finite differences.

The results do not prove universal machine-precision
forward accuracy of the complete objective gradient.

## Reproduction

Configure and build against the validated forward solver:

    cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DPIPE_SIM_SOURCE_DIR="$HOME/Sync/isothermal-pipe-sim"

    cmake --build build -j"$(nproc)"

    ctest --test-dir build \
        -R '^state_finder_' \
        --output-on-failure

Run the numerical diagnostics:

    ./build/state_finder_high_precision_physics_test
    ./build/state_finder_adjoint_components_test

The captured results are retained under:

    benchmarks/results-m15/

## Next milestone

M1.5b — Long-trajectory discrete-adjoint verification,
including forward residuals, linear-solve backward errors,
gradient accuracy, and computational cost.
