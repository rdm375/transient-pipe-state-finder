# StateFinder v0.1.0 — Release Candidate Checklist

Status: **planning / not release-ready**
Scope: **single isothermal transient gas pipe**
Baseline: M2.2, commit `b88e2d1` on `feature/shared-trajectory-matrix-free`

## Release contract

Given a known single-pipe model, initial-state baseline, prescribed
inlet-pressure and outlet-mass-flow histories, and measured
inlet-mass-flow and outlet-pressure histories, estimate the initial
internal pressure and flow distribution and inlet flow.

Provide quality and observability diagnostics. Do not claim unique
recovery where measurements do not support it.

## 1. Physics and numerical correctness

- [ ] Document model assumptions, units, spatial staggering,
      boundary half-cells, and theta-method conventions.
- [ ] Preserve global discrete mass conservation.
- [ ] Test inventory change against theta-weighted boundary
      flux imbalance.
- [ ] Regression-test the forward model.
- [ ] Regression-test tangent-linear sensitivities.
- [ ] Regression-test discrete adjoint sensitivities.
- [ ] Verify gradients against finite differences.
- [ ] Verify matrix-free normal-operator symmetry.
- [ ] Test failed nonlinear solves.
- [ ] Reject invalid states and nonfinite inputs.
- [ ] Verify reproducibility with deterministic test seeds.

## 2. Input and observations

- [ ] Define a versioned configuration schema.
- [ ] Specify pipe geometry and physical properties.
- [ ] Specify gas properties and equation-of-state assumptions.
- [ ] Specify time grid and solver tolerances.
- [ ] Define explicit measurement roles:

      Inlet pressure: prescribed boundary input.
      Outlet mass flow: prescribed boundary input.
      Inlet mass flow: observation residual.
      Outlet pressure: observation residual.

- [ ] Validate units and timestamps.
- [ ] Detect missing and duplicate measurements.
- [ ] Document interpolation and resampling policies.
- [ ] Support per-channel measurement uncertainty scales.
- [ ] Distinguish noise standard deviations from arbitrary
      objective weights.
- [ ] Reject invalid data with actionable diagnostics.

## 3. Reconstruction

- [ ] Support arbitrary pipe discretization sizes.
- [ ] Document the initial-state coordinate layout.
- [ ] Provide configurable initial guesses.
- [ ] Support unregularized reconstruction.
- [ ] Support Tikhonov regularization.
- [ ] Support spatial-curvature regularization.
- [ ] Document regularization reference states.
- [ ] Report objective-function components separately.
- [ ] Report iteration counts and termination reasons.
- [ ] Report forward-solver failures.
- [ ] Distinguish successful termination from reaching
      the maximum iteration count.
- [ ] Prevent failed reconstructions from being silently
      reported as successful estimates.

## 4. Conservation and physical diagnostics

- [ ] Report reconstructed initial linepack.
- [ ] Report linepack changes over the observation window.
- [ ] Report physical inlet/outlet mass-flow imbalance.
- [ ] Report numerical mass-conservation defect separately.
- [ ] Report pressure and flow ranges.
- [ ] Flag physically invalid reconstructed states.
- [ ] Test conservation under transient demand changes.

## 5. Observability

- [x] Construct the M2.1 observation sensitivity matrix.
- [x] Compute the M2.2 information spectrum.
- [x] Establish the dimension-counting nullspace bound.
- [x] Evaluate reconstruction errors in information modes.
- [x] Compare regularized and unregularized reconstructions.
- [ ] Compute singular values directly using SVD.
- [ ] Report effective rank using documented thresholds.
- [ ] Include state and observation scaling in diagnostics.
- [ ] Identify weakly constrained state combinations.
- [ ] Quantify sensitivity to the observation-window length.
- [ ] Distinguish numerical rank from structural observability.
- [ ] Provide production-facing observability warnings.

## 6. M2.3 measurement uncertainty

- [ ] Test independent random measurement noise.
- [ ] Use deterministic random seeds.
- [ ] Preserve noise realizations for reproducibility.
- [ ] Test constant inlet-flow measurement bias.
- [ ] Test constant outlet-pressure measurement bias.
- [ ] Test combined measurement biases.
- [ ] Evaluate state-error distributions across trials.
- [ ] Evaluate pressure reconstruction errors.
- [ ] Evaluate flow reconstruction errors.
- [ ] Evaluate initial linepack errors.
- [ ] Compare regularization strategies.
- [ ] Quantify regularization-induced estimation bias.
- [ ] Validate the linearized analysis against nonlinear
      reconstruction experiments.
- [ ] Investigate missing samples.
- [ ] Investigate timestamp errors.
- [ ] Investigate outliers.
- [ ] Establish documented operating limits.

## 7. Production API

- [ ] Define a public C++ StateFinder interface.
- [ ] Separate model configuration from observations.
- [ ] Separate optimizer configuration from physical data.
- [ ] Define stable reconstruction-result types.
- [ ] Define structured solver-status types.
- [ ] Include numerical diagnostics in results.
- [ ] Avoid exposing research-only implementation details.
- [ ] Provide API examples.

## 8. Command-line application

- [ ] Implement a StateFinder CLI.
- [ ] Support configuration files.
- [ ] Support versioned measurement-file formats.
- [ ] Produce machine-readable reconstruction results.
- [ ] Produce machine-readable diagnostics.
- [ ] Return nonzero exit codes on failure.
- [ ] Provide useful error messages.
- [ ] Support reproducible batch execution.

## 9. Testing and validation

- [ ] Clean-checkout build passes.
- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] Forward-model regression tests pass.
- [ ] Gradient and adjoint tests pass.
- [ ] Conservation tests pass.
- [ ] Synthetic reconstruction tests pass.
- [ ] Noise and bias tests pass.
- [ ] CLI end-to-end tests pass.
- [ ] Invalid-input tests pass.
- [ ] Failure-path tests pass.
- [ ] Tests run in CI.

## 10. Packaging and documentation

- [ ] Provide CMake install targets.
- [ ] Export CMake library targets.
- [ ] Test installation outside the source tree.
- [ ] Provide a README quick start.
- [ ] Document supported platforms.
- [ ] Document numerical assumptions and limitations.
- [ ] Document observability limitations.
- [ ] Document regularization choices.
- [ ] Document uncertainty interpretation.
- [ ] Provide example input datasets.
- [ ] Provide example output datasets.
- [ ] Provide troubleshooting documentation.
- [ ] Confirm third-party license compatibility.
- [ ] Provide a project license.
- [ ] Provide a changelog.
- [ ] Provide release notes.
- [ ] Record build and reference-data provenance.

## 11. Release acceptance gates

The v0.1.0 release may proceed only when:

- [ ] All required tests pass from a clean checkout.
- [ ] No unexplained conservation regressions remain.
- [ ] No unexplained sensitivity or gradient regressions remain.
- [ ] Failed solves produce explicit failure statuses.
- [ ] Measurement-fit diagnostics are available.
- [ ] Observability limitations are reported.
- [ ] Regularization choices are documented.
- [ ] Noise and bias experiments establish defensible
      operating limits.
- [ ] The public API and CLI are documented.
- [ ] Installation works on supported Linux platforms.
- [ ] The release has a documented version and license.

Only then create the `v0.1.0` tag.

## 12. Deferred features

The following are explicitly outside the first release:

- Pipeline networks.
- Compressor stations.
- Nonisothermal flow.
- GPU and TPU acceleration.
- Neural-network surrogates.
- Online telemetry integration.
- Automatic instrument calibration.

These may be investigated after the single-pipe
statefinder has reached a dependable release.

## Guiding principle

A small measurement residual does not establish an accurate
internal-state reconstruction.

The production StateFinder must report measurement fit,
numerical convergence, conservation, observability, and
regularization dependence as distinct properties.
