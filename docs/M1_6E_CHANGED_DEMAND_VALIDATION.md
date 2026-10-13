# M1.6e: Changed-Demand Predictive Validation

## Objective

Evaluate future pipeline predictions under alternative outlet-demand schedules, using reconstructed initial states.

## Configuration

- Assimilation: 80 steps × 15 seconds (1200 s).
- Prediction: 80 additional steps (1200 s).
- Demand transition starts at t = 1200 s.
- Transition duration: 15 seconds.
- Baseline demand: 105 kg/s.
- Increased demand: 110 kg/s.
- Decreased demand: 100 kg/s.
- Reconstruction methods: unregularized, Tikhonov, curvature, hybrid.
- Optimization budget: 30 L-BFGS iterations.
- Identical physical model for truth and reconstruction.

## Predictive results

| Method | Scenario | Inlet-flow RMSE (kg/s) | Outlet-pressure RMSE (Pa) | Relative linepack RMSE |
|---|---|---:|---:|---:|
| unregularized | baseline | 0.00100437507 | 1.7596872 | 1.40083354e-07 |
| unregularized | increased | 0.000998470267 | 1.77078401 | 1.4066771e-07 |
| unregularized | decreased | 0.00101061521 | 1.7483533 | 1.39483513e-07 |
| tikhonov | baseline | 0.00100437677 | 1.75969018 | 1.40083591e-07 |
| tikhonov | increased | 0.00099847196 | 1.77078702 | 1.40667948e-07 |
| tikhonov | decreased | 0.00101061693 | 1.74835626 | 1.3948375e-07 |
| curvature | baseline | 0.00100437724 | 1.75969101 | 1.40083657e-07 |
| curvature | increased | 0.000998472429 | 1.77078785 | 1.40668014e-07 |
| curvature | decreased | 0.0010106174 | 1.74835709 | 1.39483815e-07 |
| hybrid | baseline | 0.00100437894 | 1.75969399 | 1.40083894e-07 |
| hybrid | increased | 0.00099847412 | 1.77079085 | 1.40668253e-07 |
| hybrid | decreased | 0.00101061911 | 1.74836005 | 1.39484051e-07 |

## Numerical verification

- All 12 experiments completed successfully.
- Baseline restart parity verified at every time sample.
- Conservation defects checked for truth and reconstruction.
- Maximum truth mass defect: `4.27928581459e-10`.
- Maximum reconstructed mass defect: `4.53680204515e-10`.
- Conservation acceptance threshold: `1e-8`.

## Interpretation

The tested reconstruction methods yield nearly identical future predictions. Moderate increases and decreases in outlet demand do not substantially amplify the residual initial-state errors.

These results support distinguishing initial-state identifiability from the predictability of future engineering quantities.

## Limitations

- Synthetic observations and identical-model truth.
- No physical-parameter uncertainty.
- No additional measurement noise in this experiment.
- Single pipeline geometry and operating regime.
- Future boundary inputs known exactly.
- Demand changes are one-step ramps, not discontinuities.
- Conservation tolerance is 1e-8 kg/s; the largest measured defect is approximately 4.54e-10 kg/s.
