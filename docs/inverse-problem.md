# Inverse problem specification

Let z denote the discretized initial pressure and staggered mass-flow fields. Given prescribed inlet pressure and outlet mass flow, define the observation operator h(z) as predicted inlet mass flow and outlet pressure sampled at measurement times.

Minimize J(z) = 1/2 || W^(1/2) (h(z) - y_obs) ||_2^2, subject to the discrete transient pipe equations and physically admissible states.

Initial milestones: differentiable forward model, synthetic data generation, discrete adjoint gradient verification, L-BFGS-B recovery, and sensitivity singular-value analysis.

The first experiment must distinguish low observation misfit from accurate interior-state recovery. Preserve theta for time weighting, p for pressure, and bold xi for scenario parameters.
