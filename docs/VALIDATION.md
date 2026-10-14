# StateFinder: Numerical Verification and Validation Status

**Evidence date:** 2026-10-14
**StateFinder revision:** `8cf54de3bdea0ce5f4d145f592d31dd2fade3fbe`
**Forward-solver revision:** `670ac493e4461e678ca66f2254dc3d402df3f312`

## 1. Scope and terminology

This report documents **numerical verification** of the StateFinder implementation using the tests and synthetic experiments available at the revisions above. It does **not** claim physical validation against independently measured pipeline data, uniqueness of the inverse solution, or accuracy across all operating regimes.

The numerical evidence comes from `docs/validation/ctest-inventory.md` and `docs/validation/derivative-reconstruction-results.txt`. The latter is the verbose CTest output for three selected tests. Values below are transcribed from the reported run; timing values are hardware- and workload-dependent.

## 2. Reproducibility and regression status

The recorded local CTest inventory reports **45/45 tests passed**, with **0 failures** and total elapsed test time **9.63 s**. The selected verbose run reports **3/3 tests passed** in **1.34 s**:

- `state_finder_adjoint_gradient` (test 7)
- `state_finder_tangent_linear` (test 13)
- `state_finder_joint_nonlinear_reconstruction` (test 34)

Reproduce the complete suite from a configured build directory:

```bash
ctest --test-dir build-reproducible --output-on-failure
```

Reproduce the selected diagnostic output:

```bash
ctest --test-dir build-reproducible -V \
  -R '^(state_finder_adjoint_gradient|state_finder_tangent_linear|state_finder_joint_nonlinear_reconstruction)$'
```

The source revisions alone do not capture the compiler, build flags, dependency versions, or CPU configuration; consult the repository's reproducible-build documentation and CI workflow for build instructions. The recorded measurements are local results, not cross-platform error bounds.

## 3. Discrete-adjoint directional derivatives

The test compares the adjoint directional derivative of the observation objective against a central finite-difference approximation across nine step sizes for each of six cases. Its reported discrepancy is

\[
\varepsilon_{\mathrm{FD}}(h)=
\frac{|D_{\mathrm{FD}}(h)-D_{\mathrm{adj}}|}
{\max(1,|D_{\mathrm{adj}}|)}.
\]

The acceptance criterion is the **minimum** discrepancy across the tested step sizes, not the discrepancy at every step size: `best_relative_error <= 1e-4`. The test separately requires a maximum adjoint linear-solve backward error of at most `1e-12`.

| Include initial observations | Seed | Minimum reported discrepancy |
|---|---:|---:|
| No | 1 | 6.951926479983722e-10 |
| No | 2 | 4.429325520983658e-09 |
| No | 3 | 2.114384973340427e-09 |
| Yes | 1 | 6.954067782478126e-10 |
| Yes | 2 | 4.470353938673456e-09 |
| Yes | 3 | 2.149509193995178e-09 |

**Largest minimum discrepancy across six cases:** `4.470353938673456e-09`.
**Largest reported adjoint linear-solve backward error:** `1.511743331621529e-16`.

The step-size sweep shows decreasing truncation error followed by a roundoff/solver-noise floor. This is strong evidence for the tested discrete adjoint directional derivatives, but it is not a uniform bound over arbitrary states or scenarios.

## 4. Tangent-linear / adjoint consistency

The test covers `n ∈ {2,10,100}`, time-step counts `{0,1,12,80}`, and inclusion/exclusion of initial observations: **24 configurations**. It compares the objective directional derivative calculated using tangent-linear observation perturbations with the scaled adjoint-gradient directional derivative. The reported normalized discrepancy is

\[
\varepsilon_{\mathrm{TL/adj}}=
\frac{|D_{\mathrm{TL}}-D_{\mathrm{adj}}|}
{\max(1,|D_{\mathrm{TL}}|,|D_{\mathrm{adj}}|)}.
\]

**Maximum observed discrepancy:** `1.4665870594839092e-15`, against a threshold of `1e-9`.
**Maximum tangent-linear solve backward error:** `7.7095284844124635e-19`, against a threshold of `1e-12`.

The agreement is near floating-point precision in these cases. Because the tangent-linear and adjoint implementations may share modeling assumptions, this agreement is an internal consistency check, complementary to the finite-difference experiment, not an independent physical-model validation.

## 5. Joint nonlinear inverse reconstruction

Test 34 constructs synthetic observations from a known joint initial-state/boundary-history trajectory, perturbs a baseline estimate, and compares Gauss–Newton (GN), preconditioned GN (PCGN), and L-BFGS under regularized and unregularized objectives. `truth_error` is the Euclidean distance between recovered and true **packed/scaled joint variables**, not an error in raw physical units.

**Initial scaled-variable distance to truth:** `0.235714`.

| Objective | Solver | Final objective | Scaled truth error | Iterations | Converged |
|---|---|---:|---:|---:|---|
| Regularized | GN | 7.34941e-4 | 0.0208341 | 31 | Yes |
| Regularized | PCGN | 7.34941e-4 | 0.0208327 | 30 | Yes |
| Regularized | L-BFGS | 7.37549e-4 | 0.0218361 | 150 | No (iteration limit) |
| Unregularized | GN | 4.74585e-10 | 0.0180444 | 12 | Yes |
| Unregularized | PCGN | 4.74585e-10 | 0.0180444 | 12 | Yes |
| Unregularized | L-BFGS | 1.10533e-6 | 0.0208223 | 150 | No (iteration limit) |

The unregularized GN scaled truth error fell by approximately **92.34%** relative to the initial error. The small unregularized observation objective coexists with a nonzero state error. This motivates an observability analysis; **these results alone do not establish nonuniqueness or weak observability**, since stopping tolerances and other numerical effects may also contribute.

Cumulative inner PCG iterations were `1748` (GN) versus `1905` (PCGN) for the regularized case, and `525` versus `536` for the unregularized case. The tested preconditioner did **not** reduce cumulative PCG iterations in these examples. L-BFGS did not satisfy its convergence criterion within 150 iterations in either case. These comparisons are specific to the test problem and settings.

Importantly, the test asserts objective nonincrease, finite reconstruction error, and dimension consistency; it does **not** assert an upper bound on truth error. Reported recovery accuracy is an observed outcome, not a test-enforced guarantee.

## 6. Forward model and conservation: evidence still to extract

The passing suite includes forward smoke/parity tests, high-precision physics, transient tests, spatial convergence, and numerical-reference tests. The inventory alone does not provide numerical error magnitudes, convergence orders, or mass-conservation defects for these tests. These quantities must be extracted from test output or dedicated numerical experiments before quantitative claims are made.

For conservation, distinguish **physical mass imbalance** (inlet minus outlet mass flow) from the **numerical conservation defect** (the mismatch between that imbalance and computed inventory change rate).

## 7. Observability and uncertainty: open validation questions

The current evidence does not establish the singular spectrum or effective rank of the observation sensitivity operator, uniqueness of initial-state reconstruction, or uncertainty bounds under measurement noise. Suggested follow-up studies include sensitivity singular values under explicitly stated variable/observation scalings; perturbations along weakly observed directions; recovery under noise; and stability of predicted linepack and flow histories.

## 8. Limitations and claims boundary

1. The tests primarily exercise synthetic scenarios and known numerical references; independent field-data validation has not been demonstrated here.
2. Derivative checks cover finite sets of states, grids, time horizons, directions, and finite-difference steps.
3. Near-machine-precision tangent/adjoint agreement demonstrates consistency, not absence of shared implementation errors.
4. The inverse-recovery experiment is not a proof of uniqueness, global convergence, or general reconstruction accuracy.
5. The preconditioner has not shown a benefit in the measured small reconstruction examples.
6. Other passing tests have not yet been mined for quantitative error metrics; their status is regression-pass evidence only.

## 9. Evidence files

- `docs/validation/ctest-inventory.md`: revisions, registered tests, and complete local CTest pass/fail output.
- `docs/validation/derivative-reconstruction-results.txt`: verbose diagnostics for tests 7, 13, and 34.
- `docs/validation/numerical-assertions.txt`: initial grep-based assertion index (110 matching lines in `tests/cpp`; not an exhaustive assertion count).

**Status:** Numerical verification is well supported for the tested adjoint and tangent-linear calculations and for synthetic optimization behavior. Physical validation and broader inverse-problem identifiability/uncertainty claims remain open.
