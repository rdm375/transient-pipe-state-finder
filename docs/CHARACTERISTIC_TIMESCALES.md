# Characteristic Time Scales, Observability, and Operational Prediction

**Project:** transient-pipe-state-finder

**Status:** Research evidence and reproducibility record

**Scope:** Initial-state memory, observation requirements,
predictive observability, forecast-error settling, and
operational decision timing.

## 1. Governing Research Principle

> The purpose of our research is not simply to reconstruct a pipeline state.
> It is to determine how much measurement information is needed, how rapidly
> uncertainty loses its influence on future behavior, and whether the
> resulting predictions become sufficiently reliable in time to support
> an engineering decision.

This principle governs the research program.

A numerically accurate reconstruction is not, by itself,
the ultimate engineering objective.

The important questions are:

1. What can the available measurements tell us?
2. Which initial-state uncertainties remain unresolved?
3. How do those uncertainties influence future quantities?
4. How rapidly does that influence diminish?
5. When is forecast uncertainty sufficiently small?
6. Does that happen before an operational decision deadline?

A useful analysis must connect these questions quantitatively.

---

## 2. Distinct Characteristic Time Scales

The following characteristic times must not be conflated.

### 2.1 Physical propagation time

The characteristic time for disturbances to travel through
the pipeline.

For nominal acoustic speed c and pipe length L,

    t_propagation = L / c.

A round-trip propagation time is approximately

    t_round_trip = 2 L / c.

These are physical scales, not proofs of observability.

### 2.2 Initial-state memory time

The time over which perturbations in the initial state
continue to influence the evolving solution.

For the discrete solution operator,

    x(t) = Phi_t(x0),

the local sensitivity is

    S(t) = D Phi_t(x0).

For an initial perturbation delta_x0,

    delta_x(t) approximately equals S(t) delta_x0.

The decay depends on the perturbation direction,
norm, boundary forcing, and operating regime.

There is no universal memory time without specifying
an attenuation criterion.

### 2.3 Structural observability horizon

The earliest observation duration for which the
linearized observation map has full column rank
on the admissible initial-state manifold.

This is a local, model-dependent property.

It is not equivalent to practical recoverability.

### 2.4 Practical reconstruction horizon

The observation duration required to constrain
the desired state quantities to specified accuracy
under measurement noise, model discrepancy,
regularization, and numerical limitations.

A formally full-rank observation matrix may still
be too ill-conditioned for practical reconstruction.

### 2.5 Information-saturation horizon

The duration beyond which additional observations
provide negligible improvement according to a
specified information metric.

The metric must be identified.

Accumulated residual energy is not the same as
Fisher information or posterior uncertainty reduction.

### 2.6 Predictive observability horizon

The observation duration needed to constrain
specified future outputs to engineering tolerances.

This can be much shorter than the observation duration
required to reconstruct all initial-state variables.

### 2.7 Forecast-error settling time

The first sampled forecast time after which a
specified error statistic remains below a
specified tolerance through the saved horizon.

This is output-specific and tolerance-specific.

### 2.8 Operational event time

The time at which a specified operational limit
is reached or violated.

### 2.9 Operational warning interval

The difference between the operational event time
and a specified forecast-reliability time.

A positive retrospective warning interval does
not establish a prospective operational guarantee.

---

## 3. Mathematical Framework

### 3.1 State evolution

Let the discrete state satisfy

    x[k+1] = F_k(x[k], u[k]),

with observations

    y[k] = h_k(x[k]).

Here u[k] denotes prescribed boundary inputs.

The initial state is x[0].

### 3.2 Observation sensitivity

For observations collected through step N,

    Y_N(x0) = [y[0], ..., y[N]].

Define

    H_N = dY_N / dx0.

For a linearization around a reference trajectory,

    A_k = dF_k / dx[k]
    C_k = dh_k / dx[k].

The state-transition sensitivity is

    Phi[k,0] = A[k-1] ... A[0].

Consequently,

    H_N =
        [ C_0
          C_1 Phi[1,0]
          ...
          C_N Phi[N,0] ].

The observation Jacobian describes sensitivity
to the initial state.

A single least-squares gradient is not a substitute
for the full observation Jacobian.

### 3.3 Scaling and whitening

Let D_x contain physical scales for initial-state
coordinates and let R be the measurement covariance.

The dimensionless whitened Jacobian is

    B_N = R^(-1/2) H_N D_x.

Compute its singular-value decomposition:

    B_N = U Sigma V^T.

Small singular values identify weakly observed
initial-state directions.

All reported ranks and condition numbers must
identify their scaling and numerical threshold.

### 3.4 Information matrix

For locally Gaussian measurement errors,

    W_N = B_N^T B_N.

A prior with covariance Gamma_0 gives the
local posterior precision approximation

    Gamma_post^(-1) =
        W_N + Gamma_0^(-1).

The prior can constrain directions that measurements
do not observe.

That constraint must not be attributed to the data.

Numerical rank should be assessed using the
Jacobian or its singular values, rather than
normal equations that square the condition number.

### 3.5 Predictive uncertainty

Let q(t) be an engineering output at future time t.

Its initial-state sensitivity is

    g_q(t) = d q(t) / d xi,

where xi denotes dimensionless initial-state coordinates.

For a local Gaussian approximation,

    Var[q(t)] approximately equals
        g_q(t)^T Gamma_post g_q(t).

The uncertainty of a future output depends on
the overlap between its sensitivity and the
remaining uncertain state directions.

Therefore, weak initial-state observability does
not necessarily imply weak predictive observability.

Conversely, accurate prediction of one output
does not imply accurate prediction of all outputs.

### 3.6 Ensemble forecast error

For N realizations and output q, define

    e_j(t) = q_estimated,j(t) - q_reference(t).

The ensemble root-mean-square error is

    RMSE_q(t) =
        sqrt((1/N) sum_j e_j(t)^2).

The maximum observed absolute error is

    MAX_q(t) = max_j abs(e_j(t)).

These are different statistics.

Neither is automatically a certified uncertainty bound.

### 3.7 Empirical settling time

For an error statistic E_q(t), tolerance epsilon,
and saved forecast times t_0,...,t_K, define

    t_settle =
        min {t_i :
             E_q(t_j) <= epsilon for every j >= i}.

If the criterion is not satisfied by the final
saved time, record the result as not settled.

This definition is retrospective.

It requires the complete saved error history.

### 3.8 Operational event and warning time

For a downstream pressure threshold p_limit,

    t_limit =
        first time p_out(t) <= p_limit.

Define

    t_warning = t_limit - t_settle.

A negative warning time indicates that the
chosen empirical accuracy criterion is reached
after the reference operational event.

---

## 4. M15B: Long-Trajectory Attenuation

### 4.1 Configuration

The original experiment used:

- Pipeline length: 100 km.
- Spatial discretization: 100 cells.
- Initial-state dimension: 201.
- Time step: 15 seconds.
- Theta-method time weight: 0.65.
- Prescribed inlet pressure and outlet mass flow.
- Measured outputs: inlet mass flow and outlet pressure.
- Reference and perturbed candidate initial states.
- Common prescribed boundary forcing.

The experiment compared forward trajectories
and decomposed their observation misfit.

### 4.2 Numerical procedure

The procedure was:

1. Generate the reference trajectory.
2. Perturb the initial state.
3. Propagate the candidate under identical
   prescribed boundary conditions.
4. Compute weighted boundary-observation misfit.
5. Partition the objective into nonoverlapping
   observation windows.
6. Compare boundary and state differences.
7. Verify adjoint directional derivatives
   against independent central finite differences.

The objective used normalization values:

- Inlet flow: 1 kg/s.
- Outlet pressure: 10,000 Pa.

These are objective weights, not independently
validated sensor-noise standard deviations.

### 4.3 Verified results

| Observation window | Objective contribution |
|---|---:|
| Steps 0-80 | 5.3415934241208767 |
| Steps 81-800 | 0.082381054966959938 |
| Steps 801-8000 | 1.1947123543218275e-15 |

Total objective:

    5.423974479087843

The first 80 steps correspond to 20 minutes.

Approximately 98.48% of the tested objective
accumulated during those first 20 minutes.

Maximum observed differences:

| Window | Inlet flow (kg/s) | Outlet pressure (Pa) |
|---|---:|---:|
| 0-80 | 0.880860006826 | 681.422968264 |
| 81-800 | 0.084053338577 | 146.494575984 |
| 801-8000 | 1.0235623904e-8 | 1.8133781850e-5 |

At the final time:

- Maximum pressure-state difference:
  1.8067657947540283e-7 Pa.
- Maximum flow-state difference:
  1.0270184702676488e-10 kg/s.

The trajectories converged toward the same
forced response to the resolution of the experiment.

### 4.4 Interpretation

The experiment demonstrates strong attenuation
of the tested initial-state perturbation.

It does not establish:

- Full initial-state observability.
- A universal 20-minute observation requirement.
- A smallest singular value.
- A practical reconstruction horizon.
- A universal initial-state forgetting time.

The 98.48% result is a decomposition of one
weighted misfit objective.

### 4.5 Source and evidence

Documentation:

    docs/M15B_LONG_TRAJECTORY_VERIFICATION.md

Source:

    tests/cpp/adjoint_long_trajectory.cpp
    tests/cpp/observation_windows.cpp

Original results:

    benchmarks/results-m15b/observation-windows.log
    benchmarks/results-m15b/adjoint-80.csv
    benchmarks/results-m15b/adjoint-800.csv
    benchmarks/results-m15b/adjoint-8000.csv

The original log is the primary evidence for
the objective-window decomposition.

---

## 5. M16B: Initial-State Observability

### 5.1 Research question

How much observation history is needed before
the initial-state observation Jacobian becomes
full rank, and how well conditioned is it?

### 5.2 Configuration

The experiment used the 201-variable,
100-cell initial-state representation.

Both boundary outputs were sampled at
15-second intervals.

With observations at steps 0 through N,
the number of scalar observations is

    2(N+1).

For 201 unconstrained initial-state variables,
a necessary count condition for full column rank is

    2(N+1) >= 201.

Therefore,

    N >= 100.

This corresponds to 25 minutes.

It is a necessary count condition,
not proof of full rank or practical recovery.

### 5.3 Numerical procedure

The research constructed an initial-state
observation-sensitivity matrix and examined
its singular-value spectrum across multiple
observation horizons.

The retained summary reports:

- Matrix dimensions.
- Numerical rank and nullity.
- Largest and smallest singular values.
- Condition number.
- Weak-mode coordinate composition.

The original implementation and exact
rank-threshold convention should be inspected
before extending these results.

### 5.4 Original numerical results

| Duration | Matrix size | Reported rank | Smallest singular value | Condition |
|---|---|---:|---:|---:|
| 20 min | 162 x 201 | 162 | 0 | infinite |
| 25 min | 202 x 201 | 201 | 3.707119e-10 | 2.829471e11 |
| 50 min | 402 x 201 | 201 | 5.917542e-8 | 1.772558e9 |
| 100 min | 802 x 201 | 201 | 2.329772e-7 | 4.502238e8 |
| 200 min | 1602 x 201 | 201 | 3.751299e-7 | 2.796148e8 |

At 20 minutes, the observation matrix
has at least 39 first-order null directions
by the rank-nullity theorem.

At 25 minutes, the reported matrix is
full column rank but severely ill-conditioned.

Increasing observation duration improves
the weakest singular values.

Nevertheless, the matrix remains
severely ill-conditioned at 200 minutes.

### 5.5 Interpretation

The experiment distinguishes:

- Observation-count sufficiency.
- Numerical full rank.
- Practical identifiability.

Full numerical rank does not imply
acceptable reconstruction uncertainty.

The practical observation horizon depends
on noise levels, state scaling, prior
assumptions, and engineering tolerances.

### 5.6 Evidence

    benchmarks/results-m16b/observability_summary.csv

Related analysis:

    docs/INITIAL_STATE_OBSERVABILITY.md

The original source implementation should
be located and its exact computational
settings added to the provenance inventory.

---

## 6. M17A: Predictive Observability

### 6.1 Research question

Must the entire initial state be accurately
reconstructed to predict future boundary outputs?

The central distinction is between:

    uncertainty in x0

and

    uncertainty in q(t).

A weakly observed state direction may have
little influence on a future output.

### 6.2 Computational framework

The predictive-observability analysis combines:

1. An initial-state prior.
2. Measurement sensitivity.
3. Observation-conditioned uncertainty.
4. Future-output sensitivity.
5. Propagation of uncertainty into
   selected forecast quantities.

In a local Gaussian approximation,

    Gamma_post^(-1) =
        B^T B + Gamma_0^(-1).

For output q,

    sigma_q^2 =
        g_q^T Gamma_post g_q.

The precise implementation should be
checked against the retained source scripts
before asserting additional algorithmic details.

### 6.3 Verified saved results

For a 30-minute forecast of outlet pressure:

| Assimilation duration | Prior std (Pa) | Posterior std (Pa) |
|---|---:|---:|
| 5 min | 100.052931872 | 0.276233812 |
| 10 min | 100.052931872 | 0.182902603 |
| 20 min | 100.052931872 | 0.160634327 |

The corresponding reduction fractions are
approximately:

- 99.724% at 5 minutes.
- 99.817% at 10 minutes.
- 99.839% at 20 minutes.

These results depend on the specified
prior and linearized model.

They are not validated empirical coverage
probabilities or operational guarantees.

### 6.4 Prior-sensitivity evidence

The retained M17A prior-observability
sweep includes multiple pressure and
flow prior scales.

For the 10,000 Pa / 1 kg/s prior case:

| Observation duration | Effective dimension |
|---|---:|
| 20 min | 32.097198 |
| 25 min | 32.098645 |
| 50 min | 32.099594 |
| 100 min | 32.099605 |
| 200 min | 32.099605 |

This indicates saturation of that
specific effective-dimension metric.

It is not a universal observation horizon.

### 6.5 Engineering interpretation

The initial state can contain weakly
observed directions even when a particular
future output is well constrained.

This is one of the central engineering
motivations for predictive observability.

We should optimize observation duration
and sensor placement for the operational
outputs of interest, not merely for
full-state reconstruction.

### 6.6 Evidence and implementation

Saved results:

    benchmarks/results-m17a/predictive_observability.csv
    benchmarks/results-m17a/prior_observability_summary.csv

Analysis scripts:

    analysis/predictive_observability.py
    analysis/prior_observability.py

The scripts should be retained with the
original result files.

---

## 7. M24-M25: Additional Observability Evidence

Subsequent milestones investigated
compensated observability, boundary-input
effects, and alternative measurement channels.

Relevant source and artifacts include:

    apps/m24_observability.cpp
    analysis/m24_observability_svd.py
    benchmarks/results-m24/initial_observability_modes.csv
    benchmarks/results-m24/compensated_observability_modes.csv

    analysis/m25_four_channel_observability.py
    benchmarks/results-m25/four_channel_observability_summary.csv
    benchmarks/results-m25/operational_sensitivity.csv

These studies must be retained in the
research provenance.

The M25 summary distinguishes joint,
boundary, fixed-initial-state, and
profiled-initial-state ranks.

Those ranks depend on the parameterization,
measurement-channel scaling, and numerical
threshold conventions.

They should not be combined numerically
with the 201-variable M16B experiment
without documenting the differences.

---

## 8. M27: Reconstruction Information Spectrum

### 8.1 Configuration

The M27 reconstruction model used:

- 163 variables in the original estimator.
- 165 variables in the calibration-aware estimator.
- 32 independent noise realizations per scenario.
- Two measurement scenarios, C1 and C2.
- A fixed budget of 24 Gauss-Newton iterations.

### 8.2 Observability procedure

M27-C7 examined the undamped
correlated-measurement Gauss-Newton
information matrix at:

1. The synthetic reference state.
2. A matrix-free PCG reconstruction.
3. A dense-solver reconstruction.

At relative eigenvalue threshold 1e-10,
the effective rank was:

    148 / 163

at all three evaluation points.

The ten weakest eigenvectors were
concentrated primarily in initial-state
coordinates.

Their local terminal-inventory
sensitivities were small.

This does not establish small total
inventory uncertainty.

### 8.3 Important limitations

All 128 statistical reconstructions
terminated at the 24-iteration limit
without satisfying the gradient tolerance.

The reconstructions were initialized
at the synthetic reference state
in scaled coordinates.

The statistical comparisons therefore
characterize fixed-budget procedures,
not operational initialization or
fully converged estimators.

Calibration offsets were not
reliably recovered.

### 8.4 Evidence

    docs/M27_PHASE3_CLOSURE.md
    benchmarks/results-m27/spectrum-c7/
    benchmarks/results-m27/statistical-c/

These results reinforce the distinction
between state identifiability and
operational-output accuracy.

---

## 9. M28 and D3-T: Forecast Settling

### 9.1 Engineering configuration

The current engineering demonstration uses:

- Pipeline length: 50 km.
- Diameter: 0.6 m.
- Spatial discretization: 20 cells.
- Nominal inlet pressure: 8 MPa.
- Nominal flow: approximately 100 kg/s.
- Observation window: 600 seconds.
- Forecast horizon: 1,800 seconds.
- Future downstream demand: 110 kg/s.

This configuration is different from
the 100-km M15B-M17A research model.

Characteristic times cannot be transferred
between them without qualification.

### 9.2 Numerical procedure

The D3-T analysis reads existing
M28 forecast histories.

For each scenario and estimator:

1. Read 32 saved forecast histories.
2. Align records by forecast time.
3. Calculate the ensemble RMSE.
4. Calculate maximum observed absolute error.
5. Repeat for outlet pressure,
   total inventory, and spatial pressure.
6. Apply exploratory tolerance grids.
7. Find the first sampled time after
   which the selected error statistic
   remains within tolerance through
   the saved 1,800-second horizon.
8. Compare settling time with the
   reference operational event time.

No new forward simulations are required.

### 9.3 Exploratory tolerances

Inventory RMSE and maximum error:

    10, 25, 50, 100, 200, 400 kg

Outlet-pressure RMSE and maximum error:

    0.5, 1, 2, 5, 10, 20 kPa

Spatial-pressure RMSE and maximum error:

    1, 2, 5, 10, 20, 50 kPa

These are analysis tolerances,
not established operating limits.

### 9.4 Outlet-pressure attenuation

| Scenario | Estimator | RMSE at 0 s (kPa) | RMSE at 1800 s (kPa) |
|---|---|---:|---:|
| C1 | Original | 3.4925 | 0.54055 |
| C1 | Calibrated | 3.4366 | 0.51054 |
| C2 | Original | 4.0022 | 0.67754 |
| C2 | Calibrated | 3.8869 | 0.63728 |

### 9.5 Inventory attenuation

| Scenario | Estimator | RMSE at 0 s (kg) | RMSE at 1800 s (kg) |
|---|---|---:|---:|
| C1 | Original | 241.21 | 34.828 |
| C1 | Calibrated | 225.47 | 32.894 |
| C2 | Original | 306.31 | 43.654 |
| C2 | Calibrated | 285.85 | 41.060 |

### 9.6 Outlet-pressure RMSE settling: 2 kPa

| Scenario | Estimator | Settling time (s) |
|---|---|---:|
| C1 | Original | 470 |
| C1 | Calibrated | 410 |
| C2 | Original | 700 |
| C2 | Calibrated | 630 |

Calibration improves the observed
settling time by 60 seconds in C1
and 70 seconds in C2.

### 9.7 Maximum observed outlet-pressure error: 5 kPa

| Scenario | Estimator | Settling time (s) |
|---|---|---:|
| C1 | Original | 480 |
| C1 | Calibrated | 450 |
| C2 | Original | 460 |
| C2 | Calibrated | 430 |

The RMSE and maximum-error criteria
have different tolerances and
must not be treated as equivalent.

### 9.8 Inventory RMSE settling

| Scenario | Estimator | 100 kg (s) | 200 kg (s) |
|---|---|---:|---:|
| C1 | Original | 720 | 80 |
| C1 | Calibrated | 670 | 40 |
| C2 | Original | 950 | 270 |
| C2 | Calibrated | 890 | 210 |

The engineering conclusion depends
strongly on the chosen tolerance.

### 9.9 Original data and code

Analysis script:

    benchmarks/scripts/engineering_timescales.py

Time-resolved ensemble errors:

    benchmarks/results-engineering-demo/timescales/forecast_error_curves.csv

Settling-time map:

    benchmarks/results-engineering-demo/timescales/settling_time_map.csv

Input histories:

    benchmarks/results-engineering-demo/
    benchmarks/results-m28/forecast-a4/

The retained D3-T analysis produced:

- 2,172 time-resolved error records.
- 144 settling-time assessments.

These saved results, not rounded
report tables, are the primary
numerical evidence.

---

## 10. Operational Threshold and Warning Time

### 10.1 Synthetic pressure threshold

The reference downstream pressure
declines during the forecast.

A synthetic demonstration threshold
of 6.800 MPa was selected using
the reference trajectory.

The interpolated reference crossing is

    t_limit = 515.722 seconds.

This is not an independently
validated operational pressure limit.

### 10.2 Warning time for 2 kPa pressure RMSE

| Scenario | Estimator | Settling (s) | Warning (s) |
|---|---|---:|---:|
| C1 | Original | 470 | +45.722 |
| C1 | Calibrated | 410 | +105.722 |
| C2 | Original | 700 | -184.278 |
| C2 | Calibrated | 630 | -114.278 |

For this particular criterion,
C1 reaches the selected empirical
accuracy before the event.

C2 does not.

### 10.3 Warning time for 5 kPa maximum error

| Scenario | Estimator | Settling (s) | Warning (s) |
|---|---|---:|---:|
| C1 | Original | 480 | +35.722 |
| C1 | Calibrated | 450 | +65.722 |
| C2 | Original | 460 | +55.722 |
| C2 | Calibrated | 430 | +85.722 |

This is a different and less
stringent error criterion.

### 10.4 Event-time prediction errors

| Scenario | Estimator | MAE (s) | RMSE (s) |
|---|---|---:|---:|
| C1 | Original | 20.848 | 25.204 |
| C1 | Calibrated | 19.423 | 23.818 |
| C2 | Original | 27.511 | 31.406 |
| C2 | Calibrated | 25.815 | 29.588 |

Maximum observed crossing-time
errors were approximately 63-66 seconds.

These are empirical errors,
not certified bounds.

### 10.5 Reproducibility

    benchmarks/results-engineering-demo/threshold_crossing_6p8mpa.csv
    benchmarks/results-engineering-demo/engineering_summary.csv

The original 7 MPa threshold
was already violated at forecast start
and does not provide a meaningful
impending-event prediction test.

---

## 11. Important Unresolved Historical Values

An earlier report recorded approximate
initial-state sensitivity norms:

| Time | Historical value |
|---|---:|
| 5 min | 113.3 |
| 20 min | 28.2 |
| 50 min | 1.95 |
| 200 min | 3.48e-6 |

The original experiment and exact
definitions for these four numbers
have not yet been recovered.

They must remain explicitly
classified as unverified historical
values until their provenance is found.

Do not use them to fit a physical
forgetting-time constant.

Do not silently substitute M16B
singular values for these numbers.

They measure different quantities.

---

## 12. Engineering Interpretation

The research establishes several
important distinctions.

### 12.1 Forgetting is not observability

A particular perturbation can decay
rapidly while other initial-state
directions remain poorly observed.

### 12.2 Full rank is not practical recovery

The observation Jacobian can become
formally full rank while remaining
too ill-conditioned for accurate
reconstruction under noise.

### 12.3 State recovery is not predictive recovery

Weakly observed initial-state directions
may have little influence on a
particular future engineering output.

### 12.4 Forecast settling is output-specific

Pressure and inventory can have
different settling times.

Neither can be represented by
a universal trust time.

### 12.5 Reliability is tolerance-specific

A forecast may be adequate for
one engineering tolerance and
inadequate for another.

### 12.6 Timing determines usefulness

Forecast accuracy achieved after
an operational event may be
insufficient for the required decision.

### 12.7 Retrospective settling is not prospective trust

An operational estimator cannot
directly observe its true future error.

Prospective reliability requires
validated uncertainty estimation
or an operationally available
accuracy indicator.

---

## 13. Required Provenance for Future Experiments

Every future characteristic-time
investigation must preserve:

1. Research question.
2. Mathematical definition.
3. Physical configuration.
4. State and observation definitions.
5. Boundary conditions.
6. Numerical discretization.
7. Solver and analysis source files.
8. Source revision and dependencies.
9. Inputs and random seeds.
10. Original result files.
11. Analysis procedure.
12. Numerical tolerances.
13. Reported uncertainty or error metric.
14. Interpretation.
15. Limitations and applicability.

Original results must remain available
even when later investigations improve
upon them.

A rounded table in a report is not
a substitute for the saved data.

---

## 14. Research Priorities

### Priority 1: Preserve and verify existing evidence

Recover source implementations and
exact settings for M16B and M17A.

Resolve the provenance of the
four historical sensitivity norms.

Do not rerun experiments unnecessarily.

### Priority 2: Quantify information versus observation duration

Identify the practical observation
requirements for specific engineering
outputs and noise assumptions.

### Priority 3: Quantify uncertainty attenuation

Determine how uncertainty in the
reconstructed state influences
future pressure and inventory.

### Priority 4: Establish prospective reliability

Develop and validate methods for
predicting forecast accuracy
without access to the true trajectory.

### Priority 5: Connect reliability to decisions

Compare output-specific reliability
times with operational deadlines.

### Priority 6: Preserve physical meaning

Maintain separate diagnostics for:

- Total linepack.
- Available linepack.
- Inlet/outlet mass-flow imbalance.
- Linepack change rate.
- Minimum pressure.
- Pressure-limit proximity.
- Numerical mass-conservation defect.

Physical imbalance must not be
confused with numerical conservation error.

---

## 15. Final Research Principle

> The purpose of our research is not simply to reconstruct a pipeline state.
> It is to determine how much measurement information is needed, how rapidly
> uncertainty loses its influence on future behavior, and whether the
> resulting predictions become sufficiently reliable in time to support
> an engineering decision.

This principle is the standard
against which subsequent research
and engineering demonstrations
should be evaluated.
