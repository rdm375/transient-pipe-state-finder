# Engineering Case Study
## Transient Linepack Estimation, Forecast Reliability, and Operational Time Scales

**Project:** transient-pipe-state-finder  
**Status:** Engineering demonstration; synthetic validation  
**Primary objective:** Determine when reconstructed pipeline states become sufficiently
accurate for engineering decisions and whether that accuracy is achieved before
an operational constraint is violated.

---

## 1. Executive Summary

The central engineering question is:

> Given uncertain measurements and an imperfectly reconstructed pipeline state,
> when do predictions become accurate enough to support an operational decision,
> and is there still time to act?

This case study integrates existing state reconstruction, forecast continuation,
measurement calibration, and uncertainty experiments.

The principal engineering findings concern four characteristic time scales:

1. Initial-state memory time.
2. Observation-window requirements.
3. Forecast-error settling time.
4. Time remaining before an operational constraint is violated.

These quantities are related but distinct.

The engineering value of state estimation depends on their relationship,
not simply on the numerical accuracy of the reconstructed initial state.

### Principal result

For the synthetic 6.800 MPa downstream-pressure threshold:

- Reference crossing time: 515.722 seconds.
- Calibrated C1 outlet-pressure RMSE settles below 2 kPa at 410 seconds.
- Calibrated C2 outlet-pressure RMSE settles below 2 kPa at 630 seconds.
- Corresponding retrospective warning times: +105.722 and -114.278 seconds.

Thus, the same estimator and engineering tolerance can produce substantially
different operational outcomes under different measurement conditions.

These are empirical, retrospective findings, not certified operational guarantees.

---

## 2. Engineering Problem

Consider a transient gas pipeline with:

- Length: 50 km.
- Diameter: 0.6 m.
- Nominal inlet pressure: 8 MPa.
- Nominal mass flow: approximately 100 kg/s.
- Spatial discretization: 20 cells.
- Reconstruction observation window: 600 seconds.
- Forecast horizon: 1,800 seconds.

The operational scenario involves increased downstream demand.

The engineering objective is to estimate the pipeline state from imperfect
boundary measurements and predict downstream pressure and gas inventory.

A useful estimator must provide sufficient accuracy before an operational
decision deadline.

---

## 3. Governing Engineering Principle

The central question is not merely:

> How accurately can the initial state be reconstructed?

It is:

> When does the reconstructed state become sufficiently reliable for a
> particular engineering quantity, and does that occur early enough to act?

A forecast may be accurate for downstream pressure but not for total linepack.

Likewise, a forecast may eventually become accurate but only after an
operational limit has already been violated.

Consequently, reliability must be assessed separately for each quantity
and engineering tolerance.

---

## 4. Characteristic Time Scales

### 4.1 Initial-State Memory Time

Let the transient flow solution operator be

    x(t) = Phi_t(x0).

For a perturbation in the initial state,

    delta_x(t) approximately equals D Phi_t(x0) delta_x0.

The initial-state memory time characterizes how rapidly the dynamics
attenuate initial-state perturbations.

Earlier sensitivity investigations reported approximately:

| Elapsed time | Initial-state sensitivity norm |
|---|---:|
| 5 minutes | 113.3 |
| 20 minutes | 28.2 |
| 50 minutes | 1.95 |
| 200 minutes | 3.48e-6 |

These historical values require verification against their original
experiment definitions, normalization, and saved evidence before being
treated as independently reproduced results.

The decay rate is dependent on the boundary conditions, perturbation
direction, chosen norm, and physical operating regime.

A single universal memory time must not be inferred.

### 4.2 Observation-Window Requirement

The observation time is the measurement duration required to constrain
the desired reconstructed state.

This is not equivalent to initial-state memory time.

As the system forgets its initial state, measurements taken later may
contain less information about that original state.

Earlier observation-window investigations indicated that the first
20 minutes contributed approximately 98.48% of the accumulated
measurement objective in one decomposition.

This historical figure must be recovered and verified from the
original experiment before use as a general observability conclusion.

The engineering report must distinguish:

- Information about the original initial state.
- Information about the current state.
- Information sufficient to predict future operational quantities.

### 4.3 Forecast-Error Settling Time

For an engineering quantity q and N reconstructed forecasts, define

    E_q(t) = sqrt(
        (1/N) sum_j [q_estimated,j(t) - q_reference(t)]^2
    ).

Given an engineering tolerance epsilon_q, define the empirical
settling time as the first sampled time after which the selected
error statistic remains below tolerance through the saved forecast.

    t_settle,q = inf {
        t : E_q(s) <= epsilon_q for all sampled s in [t,T]
    }.

We also calculate settling times using the maximum observed
absolute error across the ensemble.

These two criteria must be reported separately.

Neither establishes a certified uncertainty bound.

### 4.4 Operational Event Time

For a downstream-pressure threshold p_limit,

    t_limit = inf {
        t : p_out(t) <= p_limit
    }.

For the selected synthetic threshold of 6.800 MPa,

    t_limit = 515.722 seconds.

The threshold was selected using the reference pressure trajectory
alone, before examining reconstructed crossing-time errors.

It is a demonstration threshold, not an independently established
operating limit.

### 4.5 Retrospective Warning Time

Define

    t_warning = t_limit - t_settle.

A positive result indicates that the specified empirical error
criterion was achieved before the reference event.

A negative result indicates that the criterion was achieved afterward.

This is a retrospective diagnostic.

An operator cannot directly observe the true forecast error.
Prospective use requires a validated uncertainty estimator or
another operationally available accuracy indicator.

---

## 5. Integrated Computational Workflow

The demonstration uses the existing numerical infrastructure:

    M27 reconstructed states
        |
        v
    M28 forward continuation
        |
        v
    Time-dependent pressure and inventory forecasts
        |
        v
    Ensemble error analysis
        |
        v
    Settling-time and threshold-crossing assessment

The original forward solver is reused directly.

The D3-T analysis does not introduce a new numerical method and
does not require additional forward simulations.

---

## 6. Forecast Error Attenuation

The saved M28 histories contain 32 realizations for each combination
of measurement scenario and estimator.

### 6.1 Outlet-Pressure Error

| Scenario | Estimator | Initial RMSE (kPa) | Final RMSE (kPa) |
|---|---|---:|---:|
| C1 | Original | 3.4925 | 0.54055 |
| C1 | Calibrated | 3.4366 | 0.51054 |
| C2 | Original | 4.0022 | 0.67754 |
| C2 | Calibrated | 3.8869 | 0.63728 |

All four cases show substantial attenuation over 1,800 seconds.

The final outlet-pressure RMSE is approximately 15-17% of
its initial value.

### 6.2 Total Inventory Error

| Scenario | Estimator | Initial RMSE (kg) | Final RMSE (kg) |
|---|---|---:|---:|
| C1 | Original | 241.21 | 34.828 |
| C1 | Calibrated | 225.47 | 32.894 |
| C2 | Original | 306.31 | 43.654 |
| C2 | Calibrated | 285.85 | 41.060 |

Inventory errors also attenuate substantially.

However, inventory and pressure have different error scales
and different settling times.

This reinforces the need for output-specific reliability criteria.

---

## 7. Pressure Settling Times

The following settling times are measured relative to the start
of the 1,800-second forecast.

### 7.1 Ensemble RMSE Criterion: 2 kPa

| Scenario | Estimator | Settling time (s) | Warning time (s) |
|---|---|---:|---:|
| C1 | Original | 470 | +45.722 |
| C1 | Calibrated | 410 | +105.722 |
| C2 | Original | 700 | -184.278 |
| C2 | Calibrated | 630 | -114.278 |

For C1, the pressure forecast satisfies the selected RMSE
criterion before the pressure-limit crossing.

For C2, the same criterion is not achieved until after the event.

Calibration improves settling time by:

- 60 seconds in C1.
- 70 seconds in C2.

These are results for the selected 2 kPa exploratory tolerance.

### 7.2 Maximum Observed Error Criterion: 5 kPa

| Scenario | Estimator | Settling time (s) | Warning time (s) |
|---|---|---:|---:|
| C1 | Original | 480 | +35.722 |
| C1 | Calibrated | 450 | +65.722 |
| C2 | Original | 460 | +55.722 |
| C2 | Calibrated | 430 | +85.722 |

All four cases meet this less stringent absolute-error criterion
before the reference event.

The 2 kPa ensemble-RMSE and 5 kPa maximum-error criteria
are not equivalent confidence requirements.

Their settling times must not be compared as though they were.

---

## 8. Inventory Settling Times

### 8.1 Ensemble RMSE Criterion: 100 kg

| Scenario | Original (s) | Calibrated (s) |
|---|---:|---:|
| C1 | 720 | 670 |
| C2 | 950 | 890 |

None of these cases reaches the 100 kg RMSE tolerance before
the 515.722-second pressure-limit crossing.

### 8.2 Ensemble RMSE Criterion: 200 kg

| Scenario | Original (s) | Calibrated (s) |
|---|---:|---:|
| C1 | 80 | 40 |
| C2 | 270 | 210 |

All four cases satisfy this looser tolerance before the event.

The conclusion about whether the forecast is useful therefore
depends on the required inventory accuracy.

The full tolerance-versus-settling-time maps are more informative
than a single selected tolerance.

---

## 9. Pressure-Threshold Crossing Accuracy

The reference outlet pressure declines from approximately
6.851287 MPa to 6.745091 MPa during the forecast.

For the synthetic 6.800 MPa threshold:

    Reference crossing time = 515.722 seconds.

Crossing times are estimated by interpolation between saved
forecast samples.

| Scenario | Estimator | MAE (s) | RMSE (s) | Maximum error (s) |
|---|---|---:|---:|---:|
| C1 | Original | 20.848 | 25.204 | 65.955 |
| C1 | Calibrated | 19.423 | 23.818 | 63.813 |
| C2 | Original | 27.511 | 31.406 | 64.509 |
| C2 | Calibrated | 25.815 | 29.588 | 62.620 |

Calibration improves crossing-time RMSE by approximately:

- 5.5% for C1.
- 5.8% for C2.

The maximum observed errors are approximately 63-66 seconds.

These errors must not be interpreted as certified operational
uncertainty bounds.

The original 7.000 MPa threshold assessment is retained as a
separate already-violated-limit case: the reference pressure is
below 7.000 MPa at the beginning of the forecast.

Its zero-second crossing-time error is not a meaningful measure
of impending-event prediction.

---

## 10. Engineering Interpretation

The central engineering result is the relationship between:

    Observation time
    Initial-state memory time
    Forecast settling time
    Operational event time

The time scales are related, but they are not necessarily
sequential physical processes.

The principal findings are:

1. Reconstruction errors attenuate during forward propagation.
2. The required settling time depends on the engineering quantity.
3. Settling time depends strongly on the chosen accuracy tolerance.
4. Measurement conditions influence settling and warning times.
5. Calibration improves the observed settling times.
6. Some forecasts reach the selected accuracy before the
   operational deadline; others do not.
7. Event-time accuracy and state-error settling are complementary
   diagnostics, not interchangeable measures.

A small initial-state reconstruction error is not, by itself,
a sufficient criterion for operational usefulness.

Likewise, eventual error attenuation is not useful for a decision
if the necessary accuracy is achieved only after the deadline.

---

## 11. Limitations

### 11.1 Retrospective Assessment

Settling times were calculated using known reference solutions
and complete saved error histories.

An operator would not have access to the true forecast errors.

Prospective trust-time prediction remains an unresolved requirement.

### 11.2 Synthetic Experiments

The demonstration uses synthetic measurements and a known
reference trajectory.

The existing reconstruction campaign used reference-informed
initialization, and optimizer convergence limitations remain.

Consequently, the present results do not establish field readiness.

### 11.3 Future Boundary Conditions

Forecast continuation assumes specified future boundary histories.

Uncertainty in future controls and demands has not been fully
incorporated into the operational trust-time assessment.

### 11.4 Limited Forecast Window

The empirical settling-time definitions are evaluated on the
saved 0-1,800-second time grid.

No conclusion is established beyond that horizon.

### 11.5 Exploratory Tolerances

The pressure and inventory tolerances used in this report are
exploratory.

They are not independently justified operating requirements.

### 11.6 Available Linepack

The current analysis quantifies total gas inventory and
inventory forecast error.

Available linepack requires an explicitly defined minimum
physically admissible operating state.

It is not established by the present results.

---

## 12. Reproducibility

Primary existing artifacts:

    benchmarks/results-m27/statistical-c/
    benchmarks/results-m28/forecast-a4/
    benchmarks/results-engineering-demo/

Integrated forecast records:

    benchmarks/results-engineering-demo/engineering_summary.csv

Pressure-threshold crossing analysis:

    benchmarks/results-engineering-demo/threshold_crossing_6p8mpa.csv

Time-scale analysis script:

    benchmarks/scripts/engineering_timescales.py

Time-resolved error curves:

    benchmarks/results-engineering-demo/timescales/forecast_error_curves.csv

Tolerance-versus-settling-time results:

    benchmarks/results-engineering-demo/timescales/settling_time_map.csv

The saved time-scale analysis produced:

- 2,172 time-resolved error records.
- 144 settling-time assessments.

The analysis reads existing forecast histories without invoking
the forward solver.

The final reproducibility package must identify the exact
source revision, solver dependency revision, build configuration,
and provenance of the original sensitivity experiments.

---

## 13. Mandatory Research Priorities

The following priorities govern subsequent engineering development.

### Priority 1: Recover Physical Memory-Time Evidence

Locate the original sensitivity-decay experiments.

Verify:

- Perturbation definitions.
- Sensitivity norms and normalization.
- Boundary conditions.
- Spatial and temporal discretization.
- Numerical decay rates.
- Applicable operating regimes.

Do not substitute new simulations for existing evidence.

### Priority 2: Recover Observation-Window Evidence

Locate the original observation-window and observability studies.

Determine:

- Which states or outputs were observable.
- How information changed with observation duration.
- Whether the original 98.48% result was objective accumulation
  or a genuine observability measure.
- How observation time interacts with initial-state forgetting.

### Priority 3: Preserve the Full Settling-Time Maps

Retain both ensemble-RMSE and maximum-observed-error criteria.

Do not replace the tolerance maps with a single arbitrarily
chosen trust time.

### Priority 4: Develop Prospective Reliability Assessment

Investigate how an operator could estimate forecast uncertainty
without knowing the reference solution.

This is necessary before retrospective settling times can
become operational trust-time estimates.

### Priority 5: Validate Operational Decision Timing

Evaluate whether the forecast accuracy required for a particular
decision can be established before its operational deadline.

Distinguish accuracy achieved before the event from accuracy
achieved after it.

---

## 14. Report Completion Criteria

This engineering report is incomplete unless it quantitatively
addresses all four characteristic time scales:

1. Initial-state memory.
2. Observation-window requirement.
3. Forecast-error settling.
4. Operational event timing.

It must also explain how these time scales interact.

The report must identify the applicable engineering quantity,
accuracy tolerance, uncertainty criterion, and operating scenario.

A collection of reconstruction RMSEs, solver benchmarks,
or calibration statistics is not a substitute for this analysis.

The governing engineering principle is:

> A state estimator becomes useful when we can determine when
> its predictions are sufficiently accurate, for which operational
> quantities, and whether that accuracy is available in time
> to support a decision.

This principle governs subsequent development and reporting.

## Research Evidence and Reproducibility

The mathematical definitions, original experiments, characteristic time-scale results, computational procedures, and limitations are documented in [Characteristic Time Scales, Observability, and Operational Prediction](CHARACTERISTIC_TIMESCALES.md).

The machine-readable source inventory is [CHARACTERISTIC_TIMESCALES_PROVENANCE.csv](CHARACTERISTIC_TIMESCALES_PROVENANCE.csv).

The governing research principle is:

> The purpose of our research is not simply to reconstruct a pipeline state. It is to determine how much measurement information is needed, how rapidly uncertainty loses its influence on future behavior, and whether the resulting predictions become sufficiently reliable in time to support an engineering decision.
