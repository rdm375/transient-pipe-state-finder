# M2.4-F2D-5: Nonlinear Operational Uncertainty Reproduction

**Status:** Computational reproduction successful
**Date:** 2026-10-14
**StateFinder revision:** `f7874f0`
**Forward solver revision:** `670ac493e4461e678ca66f2254dc3d402df3f312`

## Objective

Reproduce the M2.4-F2D-5 nonlinear operational uncertainty experiment
from committed source code using a freshly compiled nonlinear oracle.

The experiment tests whether physically admissible, measurement-compatible
initial-state perturbations can produce operational deviations exceeding
the corresponding linearized F2B predictions.

The reported perturbations are feasible witnesses, not certified global optima.

## Computational environment

- Operating system: Debian Linux
- Compiler: GCC 15.2.0
- Python: 3.14.4
- Build system: CMake
- Numerical libraries: system BLAS and LAPACK
- Forward solver: pinned revision `670ac493e4461e678ca66f2254dc3d402df3f312`
- Nonlinear oracle: `state_finder_m24_f2d_oracle`

The reproduction used an isolated detached Git worktree at revision
`f7874f0`.

The forward solver was freshly cloned at its pinned revision.
The nonlinear oracle was successfully rebuilt from source.

## Reproduction procedure

The following commands were executed in the isolated worktree:

    ./scripts/checkout-forward-solver.sh

    cmake -S . -B build-f2d5-replay -DCMAKE_BUILD_TYPE=Release

    cmake --build build-f2d5-replay --target state_finder_m24_f2d_oracle -j 4

    python3 analysis/m24_f2d5_closure.py \
        --oracle build-f2d5-replay/state_finder_m24_f2d_oracle

The analysis replayed saved candidate perturbations, evaluated them
with the freshly compiled nonlinear oracle, and performed strict-feasibility
corrections.

The experiment reported:

    pressure  PASS=True
    flow      PASS=True
    inventory PASS=True

    PASS: all three strictly feasible witnesses validated

## Reproduced numerical results

| Quantity | Linear F2B prediction | Nonlinear deviation | Ratio |
|---|---:|---:|---:|
| Terminal pressure | 868.060864 Pa | 870.398879 Pa | 1.002693377 |
| Terminal flow | 0.058217476 kg/s | 0.058441832 kg/s | 1.003853751 |
| Terminal inventory | 55.997501 kg | 56.152486 kg | 1.002767716 |

The nonlinear deviations exceed the corresponding linearized predictions by:

- Pressure: approximately 0.2693%
- Flow: approximately 0.3854%
- Inventory: approximately 0.2768%

These are exceedances of linearized predictions, not violations of
rigorous nonlinear upper bounds.

## Feasibility and conservation

All three reproduced witnesses passed the implemented strict-feasibility
checks, including:

- Scaled initial-state perturbation constraints.
- Nonlinear measurement-residual constraints.
- Positive pressure.
- Conservation-defect tolerance of `1e-6 kg`.
- Strict squared-constraint margin of `1e-8`.
- Maximum relative objective reduction of `1e-5`.

The reference closure results reported conservation defects:

| Quantity | Conservation defect |
|---|---:|
| Pressure | 6.61e-10 kg |
| Flow | 4.89e-10 kg |
| Inventory | 4.93e-10 kg |

All are substantially below the specified tolerance.

## Reference artifact comparison

The reproduced files were compared against the committed reference files:

- `benchmarks/results-m24/f2d5_closure_results.csv`
- `benchmarks/results-m24/f2d5_strict_directions.csv`

The comparison required identical column names and row counts,
exact agreement for nonnumeric columns, and numeric agreement
using NumPy `allclose` with `rtol=1e-8` and `atol=1e-10`.

Both comparisons passed.

This establishes numerical agreement within the stated tolerances,
not bitwise reproducibility.

Strict feasibility was checked separately by the replayed nonlinear
oracle and closure procedure.

## Artifact integrity

Before reproduction, SHA-256 checks passed for:

- `analysis/m24_f2d5_closure.py`
- `benchmarks/results-m24/f2d5_closure_results.csv`
- `benchmarks/results-m24/f2d5_strict_directions.csv`
- `benchmarks/results-m24/f2d4_replay_results.csv`
- `benchmarks/results-m24/f2d4_validated_directions.csv`

All five matched the hashes recorded in the original F2D-5 manifest.

## Scientific interpretation

For the tested synthetic transient pipeline scenario, feasible
initial-state perturbations exist that satisfy the prescribed
nonlinear measurement constraints while producing terminal
operational deviations larger than the corresponding linearized
F2B predictions.

This demonstrates that finite-amplitude nonlinear effects can make
linearized uncertainty predictions nonconservative if those
predictions are interpreted as bounds on nonlinear responses.

## Limitations

1. The reproduction uses the same physical model and numerical
   implementation as the original experiment.
2. It is not independent physical validation against experimental
   measurements or a separate simulator.
3. The optimization problem is nonconvex; global optimality
   has not been certified.
4. The witnesses establish feasible deviations, not globally
   maximal deviations.
5. The measurement normalization and admissibility constraints
   are specific to the synthetic experiment.
6. The CSV comparison tolerances do not independently certify
   the approximately `1e-8` strict-feasibility margins. Those
   margins were checked by the replayed nonlinear oracle.

## Conclusion

**Computational reproduction: PASS**

The nonlinear oracle was rebuilt from committed StateFinder source
using the pinned forward-solver revision.

All three operational uncertainty witnesses passed the replayed
strict-feasibility checks.

Both generated numerical datasets agreed with their committed
reference artifacts within the specified numerical tolerances.

Independent physical validation and global optimization
certification remain outside the demonstrated scope.
