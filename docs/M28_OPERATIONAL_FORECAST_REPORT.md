# M28 — Operational Forecast Validation

Date: 2026-10-14

Status: Research milestone completed; operational validation remains open.

## Objective

Evaluate the effect of nonlinear state-reconstruction errors on
subsequent pipeline inventory and pressure forecasts, reusing the
existing isothermal-pipe-sim forward solver.

The observation window ends at t0 = 600 s. Forecast horizons are
600, 1200, and 1800 s beyond t0.

Future boundary conditions are prescribed and known in this experiment.

## Implemented milestones

### M28-A1: Forecast continuation

A continuation interface converts the terminal joint-trajectory
state into the initial state of a subsequent forward simulation.

The split-trajectory regression test passed.

### M28-A2: Single-realization forecast

The C1 original reconstruction r017 was continued for 1800 s.

Its inventory error decreased from +178.429 kg initially to
+19.544 kg at the terminal horizon.

This was an exploratory result, not an ensemble conclusion.

### M28-A3: Original-estimator ensemble

32 realizations were evaluated in each of two scenarios.

Original-estimator inventory RMSE:

| Scenario | Initial | 600 s | 1200 s | 1800 s |
|---|---:|---:|---:|---:|
| C1 | 241.213 | 112.491 | 62.311 | 34.828 |
| C2 | 306.310 | 141.037 | 78.108 | 43.654 |

Units: kg.

The 1800 s RMSE reductions relative to initial reconstruction
error are approximately 85.6% and 85.8%, respectively.

### M28-A4: Paired calibration comparison

Each scenario contains 32 matched measurement-noise realizations,
with original and calibration-aware reconstructions.

The calibrated estimator adds two pressure-sensor offset coordinates
to the 163 physical reconstruction coordinates. The forecast
driver excludes those calibration coordinates when reconstructing
the physical state.

| Scenario | Horizon (s) | Original RMSE (kg) | Calibrated RMSE (kg) | Reduction | Calibrated lower absolute error |
|---|---:|---:|---:|---:|---:|
| C1 | 600 | 112.491 | 106.240 | 5.56% | 26/32 |
| C1 | 1200 | 62.311 | 58.852 | 5.55% | 26/32 |
| C1 | 1800 | 34.828 | 32.894 | 5.55% | 26/32 |
| C2 | 600 | 141.037 | 132.646 | 5.95% | 26/32 |
| C2 | 1200 | 78.108 | 73.465 | 5.94% | 26/32 |
| C2 | 1800 | 43.654 | 41.060 | 5.94% | 26/32 |

There are 128 reconstruction/estimator forecast runs and 384
horizon-specific records. The paired comparison contains 192 rows.

The calibrated estimator reduces forecast inventory RMSE by
approximately 5.5–6.0% across these scenarios and horizons.

The nearly constant relative improvement is consistent with,
but does not prove, a shared dominant forecast-error attenuation
mechanism.

## Conservation

The maximum absolute reported per-step mass-conservation defect
over the M28-A4 histories is 6.534435215e-10 kg.

Physical inlet–outlet flow imbalance is distinct from numerical
conservation defect and must not be interpreted as numerical error.

## Mathematical interpretation

Let Phi_T denote the nonlinear forward flow map and M the
terminal inventory functional. A first-order forecast perturbation
satisfies

    delta_M(T) approximately equals
        grad(M(x(T)))^T D(Phi_T)(x0) delta_x0.

This motivates a future forecast-oriented sensitivity and
observability analysis.

The present ensemble statistics do not establish a single
exponential decay mode, an eigenvalue, or a globally contractive
flow map.

## Limitations

1. The underlying M27 nonlinear reconstruction solves reached
   their iteration limits rather than satisfying convergence
   criteria.
2. Synthetic truth was used as the reconstruction baseline.
3. Future inlet pressure and outlet demand are assumed known.
4. The realizations share the same physical configuration and
   reference operating trajectory.
5. Sensor calibration offsets were not reliably identified.
6. Field-data accuracy, uncertain future controls, and realistic
   operational initialization remain untested.
7. Error attenuation is demonstrated for these scenarios, not
   proved for arbitrary pipeline operating conditions.

## Reproducibility

Primary source files:

- include/state_finder/forecast_continuation.hpp
- tests/cpp/forecast_continuation_test.cpp
- apps/m28_forecast_reconstruction.cpp
- benchmarks/scripts/m28_a3_forecast_campaign.py
- benchmarks/scripts/m28_a4_forecast_comparison.py

Data products:

- benchmarks/results-m28/forecast-a3/forecast_summary.csv
- benchmarks/results-m28/forecast-a4/forecast_summary.csv
- benchmarks/results-m28/forecast-a4/paired_comparison.csv
- benchmarks/results-m28/forecast-a4/aggregate_summary.csv

M28 depends on the saved M27 reconstruction solution CSV files.
The M27 optimizer does not need to be rerun to reproduce the
forecast campaign, provided those inputs are available.

The project uses the existing isothermal-pipe-sim C++ solver.
The solver dependency must remain pinned and documented.

## Conclusion

M28 demonstrates substantial attenuation of reconstruction
inventory error under known future boundary histories.

Calibration-aware reconstruction gives a modest, consistent
additional forecast improvement.

The next priority is forecast-oriented numerical analysis and
validation under realistic initialization and boundary uncertainty.
