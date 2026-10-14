# M26 Phase 2 Closure: Measurement Uncertainty and Calibration

## Status

Phase 2 numerical investigation complete.

Validation status:
- M26-C4 calibration threshold verification: PASS.
- M26-C4 threshold monotonicity verification: PASS.
- CTest regression suite: 45/45 PASS.
- M26-C3 augmented-prior/marginal-covariance cross-check: PASS.

All uncertainty conclusions remain conditional on the documented
linearized statistical model and synthetic pipeline scenario.

This document records the uncertainty assumptions, numerical results,
identifiability limitations, and engineering implications of M26.

All results concern the specified synthetic pipeline and its linearized
joint initial-state/boundary-history inverse problem.

## 1. Reference problem

The reference system is a 50 km, 0.6 m diameter isothermal gas pipeline
with nominal inlet pressure 8 MPa and nominal flow 100 kg/s.

The outlet mass flow ramps from 100 to 110 kg/s over 120–240 s.

The observation interval is 600 s with 10 s sampling.

The discretization uses 20 spatial cells and theta-method time
weighting of 0.65.

The joint inverse problem contains:

- 41 independent initial-state variables.
- 122 boundary-history variables.
- 163 physical unknowns in total.
- 244 normalized measurements across four channels.

The four measurement channels are inlet pressure, inlet flow,
outlet pressure, and outlet flow.

The physical forward solver remains separately maintained and pinned.

## 2. Measurement assumptions

The hypothetical marginal random measurement standard deviations are:

| Channel | Standard deviation |
|---|---:|
| Inlet pressure | 5,000 Pa |
| Inlet flow | 0.5 kg/s |
| Outlet pressure | 5,000 Pa |
| Outlet flow | 0.5 kg/s |

These are assumed noise levels, not instrument specifications.

Measurement normalization scales are separate engineering quantities
and must not be interpreted as noise standard deviations.

The temporal correlation model is exponential:

    corr(e_i, e_j) = exp(-|t_i - t_j| / tau)

Channels are independent under the random-noise model.

The primary correlated-noise reference uses tau = 120 s.

## 3. Independent-noise baseline (M26-A)

The full-rank unregularized linearized generalized least-squares
terminal inventory standard deviation is approximately 65.5458 kg.

A 10,000-sample Monte Carlo experiment supported the predicted
linearized uncertainty.

The inverse problem is severely ill-conditioned despite full
numerical column rank at the stated rank threshold.

## 4. Inventory variance decomposition (M26-A2)

For the independent-noise model, the terminal inventory variance
decomposition was approximately:

| Contribution | Variance (kg^2) |
|---|---:|
| Initial-state term | 2,899.1522 |
| Boundary-history term | 2,383.5771 |
| Cross-covariance term | -986.4804 |
| Total | 4,296.2488 |

The fixed-boundary conditional inventory standard deviation was
33.7207 kg, compared with 65.5458 kg for joint estimation.

The decomposition is an estimator covariance identity, not a unique
causal attribution of physical uncertainty.

## 5. Temporal correlation (M26-B)

At tau = 120 s:

| Estimator | Terminal inventory standard deviation |
|---|---:|
| Correlation-aware GLS | 247.1183 kg |
| Correlation-blind OLS, actual | 261.7327 kg |
| Correlation-blind OLS, reported | 65.5458 kg |

The correlated-noise Monte Carlo experiment validated the predicted
standard deviations to within the observed sampling uncertainty.

The central issue is uncertainty calibration: ignoring temporal
correlation substantially understates uncertainty even when the
point estimator remains reasonably efficient.

## 6. Systematic sensor errors (M26-C1)

For an isolated constant +5 kPa pressure offset:

| Sensor | GLS terminal inventory error |
|---|---:|
| Inlet pressure | +275.9869 kg |
| Outlet pressure | +265.8224 kg |

Approximately 4.48 kPa inlet-pressure bias or 4.65 kPa
outlet-pressure bias produces an inventory error equal in magnitude
to the 247.1183 kg GLS random-noise standard deviation.

Constant bias, linear drift, and late-onset drift were evaluated.

These are first-order systematic-error sensitivities, not nonlinear
worst-case bounds.

## 7. Combined pressure bias and identifiability (M26-C2)

For simultaneous constant pressure offsets of magnitude 5 kPa:

| Bias combination | GLS terminal inventory error |
|---|---:|
| Both sensors +5 kPa | +541.8093 kg |
| Inlet +5 kPa, outlet -5 kPa | +10.1645 kg |

The projected calibration-bias singular values were:

    0.82324554
    0.00155167

The corresponding bias directions were approximately:

    Strong: (+0.758227, -0.651990)
    Weak:   (+0.651990, +0.758227)

The weakly identifiable direction is predominantly common mode.

Augmenting the inverse problem with two unconstrained constant
pressure calibration offsets increased the formal inventory
standard deviation from 247.1183 kg to 245,861.3 kg.

This is a formal linearized covariance result, not a physically
validated 246-tonne inventory confidence interval.

## 8. Calibration priors (M26-C3)

Calibration offsets were assigned Gaussian priors with marginal
standard deviation sigma_cal and correlation rho.

Marginalizing calibration offsets yields a low-rank covariance
contribution to the measurement model.

The augmented-prior and marginalized-covariance calculations agreed
to relative difference 2.441e-9 in the reported cross-check.

At sigma_cal = 1 kPa:

| Correlation model | Inventory sigma |
|---|---:|
| Independent, rho = 0 | 258.7263 kg |
| Positive, rho = +0.5 | 264.3348 kg |
| Strong common mode, rho = +0.9 | 268.7369 kg |
| Strong differential mode, rho = -0.9 | 248.3107 kg |

The calibration prior is independent information and should not be
confused with measurement information.

## 9. Calibration accuracy thresholds (M26-C4)

The following table gives the marginal calibration standard deviation
at which terminal inventory uncertainty reaches the indicated
amplification above the known-calibration baseline.

| Calibration model | Correlation | +5% | +10% | +25% |
|---|---:|---:|---:|---:|
| independent | +0.00 | 1.0325 kPa | 1.4780 kPa | 2.4199 kPa |
| positive_0p5 | +0.50 | 0.8431 kPa | 1.2069 kPa | 1.9758 kPa |
| common_0p9 | +0.90 | 0.7491 kPa | 1.0724 kPa | 1.7556 kPa |
| differential_m0p9 | -0.90 | 3.2599 kPa | 4.6701 kPa | 7.6516 kPa |

The thresholds are model-dependent probabilistic requirements.
They are not deterministic maximum calibration errors.

## 10. Engineering conclusions

1. Temporal measurement correlation materially increases terminal
   inventory uncertainty and must be included in uncertainty reporting.

2. Persistent pressure calibration errors can dominate random
   measurement uncertainty.

3. Common-mode pressure calibration uncertainty is particularly
   consequential because it is both weakly identifiable and strongly
   coupled to absolute pipeline inventory.

4. Unconstrained joint estimation of pressure offsets and pipeline
   state is not practically informative for terminal inventory under
   the present linearized measurement configuration.

5. Independent calibration information can restore useful uncertainty
   levels, with requirements depending strongly on calibration-error
   correlation.

## 11. Limitations

- All uncertainty results are based on a linearization about the
  reference trajectory.
- Gaussian noise and calibration priors are modeling assumptions.
- No nonlinear confidence-interval coverage is established.
- No empirical instrument specifications are asserted.
- The Jacobian is severely ill-conditioned.
- Full numerical rank does not imply practical identifiability.
- The synthetic scenario is not a field-data validation.
- Calibration threshold requirements are scenario-dependent.
- The formal unregularized augmented uncertainty should not be
  interpreted as a physically admissible inventory range.

## 12. Phase 3 handoff

Phase 3 will use the existing forward solver to estimate joint
initial-state and boundary-history variables from synthetic noisy
measurements.

Initial validation should include:

- Recovery against known synthetic truth.
- Explicit measurement noise and calibration assumptions.
- Nonlinear least-squares convergence diagnostics.
- Terminal inventory estimation error.
- Repeated-trial uncertainty coverage.
- Physical admissibility and mass-conservation checks.

The existing forward solver must remain unchanged.

## 13. Reproducibility

Primary analysis scripts:

- analysis/m26_independent_noise.py
- analysis/m26_inventory_variance_decomposition.py
- analysis/m26_temporal_correlation.py
- analysis/m26_correlated_noise_monte_carlo.py
- analysis/m26_sensor_bias_drift.py
- analysis/m26_pressure_bias_identifiability.py
- analysis/m26_calibration_prior_sensitivity.py
- analysis/m26_calibration_thresholds.py

Primary numerical results are preserved under:

    benchmarks/results-m26/

The reference forward solver is pinned separately.
