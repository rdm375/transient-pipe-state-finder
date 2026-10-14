# Initial-State Observability in a Transient Gas Pipeline

**Status:** Analytical research note and proposed numerical investigation; full observability has **not** yet been demonstrated.
**Repository:** `rdm375/transient-pipe-state-finder`
**Baseline:** M1.5a (`m1.5a-verified`) and M1.5b (`m1.5b-verified`).

## Abstract

How much boundary-measurement history is needed to reconstruct the internal initial pressure and mass-flow state of a gas pipeline? This report separates finite disturbance-propagation time, structural/local observability, practical identifiability under noise, and saturation of a particular least-squares objective. It derives necessary dimension bounds and a scaled discrete observability Jacobian and Gramian, gives numerical examples using the verified 100-cell transient model, and proposes a reproducible numerical campaign. The existing 80/800/8000-step tests demonstrate decay of **one tested perturbation's output misfit**, not full observability of all 201 initial-state variables.

## 1. Model, unknowns, and observations

The validated forward model uses an isothermal one-dimensional gas-pipeline discretization with 100 spatial cells, length $L=100{,}000\,\mathrm m$, time step $\Delta t=15\,\mathrm s$, and $\theta$-method weight $\theta=0.65$. Prescribed inlet pressure and outlet mass-flow demand are boundary inputs; the measured outputs are inlet mass flow and outlet pressure.

The current inverse-state representation contains 201 scalar initial-state variables: 100 pressure values, 100 flow values, and one separately represented inlet-flow value. Their precise spatial indexing and any boundary-induced constraints should be checked against the actual forward-state layout before declaring all 201 variables physically independent.

Writing the discrete state as $\mathbf x_k\in\mathbb R^{201}$, the dynamics and measurement map are

$$
\mathbf x_{k+1}=F_k(\mathbf x_k;\mathbf u_k),\qquad
\mathbf y_k=h_k(\mathbf x_k)=
\begin{bmatrix}\dot m_{\mathrm{in}}(t_k)\\p_{\mathrm{out}}(t_k)\end{bmatrix}.
$$

For observations at $t_0,\ldots,t_N$, define

$$
\mathcal Y_N(\mathbf x_0)=
[\mathbf y_0^T,\ldots,\mathbf y_N^T]^T.
$$

## 2. Four distinct time horizons

1. **Propagation horizon:** When disturbances can first reach the measured boundaries.
2. **Structural/local observability horizon:** When the linearized observation map first has full column rank on the admissible initial-state manifold.
3. **Practical reconstruction horizon:** When estimated initial-state uncertainty meets specified accuracy tolerances given sensor noise and model discrepancy.
4. **Information-saturation horizon:** When additional measurements cease materially improving a specified information metric.

These horizons need not coincide. A decaying residual does not establish observability.

## 3. Necessary observation-count bound

If both boundary outputs are recorded at each of $N+1$ time levels, the number of scalar observations is $2(N+1)$. The initial-state sensitivity matrix is

$$
H_N=\frac{\partial\mathcal Y_N}{\partial\mathbf x_0}
\in\mathbb R^{2(N+1)\times201},\qquad
\mathrm{rank}(H_N)\le\min\{2(N+1),201\}.
$$

A **necessary, not sufficient** condition for full local column rank is

$$
2(N+1)\ge201\quad\Longrightarrow\quad N\ge100.
$$

At 15 seconds per step, the earliest count-compatible span is **25 minutes**, assuming both outputs are measured at $t_0$. If initial-time observations are omitted, the bound becomes $2N\ge201$, or $N\ge101$ (25 minutes 15 seconds). Boundary constraints or known initial values may reduce the effective unknown dimension; redundant measurements may reduce the independent information count.

| Steps $N$ | Duration | Scalar observations, including $t_0$ | Maximum rank | Full rank of 201 possible by count? |
|---:|---:|---:|---:|---|
| 20 | 5 min | 42 | 42 | No |
| 40 | 10 min | 82 | 82 | No |
| 80 | 20 min | 162 | 162 | No |
| 100 | 25 min | 202 | 201 | Not excluded |
| 160 | 40 min | 322 | 201 | Not excluded |
| 240 | 60 min | 482 | 201 | Not excluded |
| 400 | 100 min | 802 | 201 | Not excluded |
| 800 | 200 min | 1602 | 201 | Not excluded |

At 80 steps, rank-nullity implies at least $201-162=39$ infinitesimal directions with zero **first-order** change in the sampled outputs. This does not imply exact nonlinear indistinguishability at finite perturbation amplitude.

## 4. Characteristic propagation time

For the baseline constant-property isothermal gas model, the nominal acoustic speed is

$$
c=\sqrt{ZRT}=\sqrt{(0.9)(500)(288.15)}\approx360.09\,\mathrm{m/s}.
$$

For a 100-km pipe,

$$
t_{\mathrm{one\ way}}=\frac Lc\approx277.7\,\mathrm s\approx4.63\,\mathrm{min},
\qquad t_{\mathrm{round\ trip}}\approx9.26\,\mathrm{min}.
$$

Thus 20 minutes spans about 4.32 nominal one-way transit times, while 25 minutes spans about 5.40. These are useful physical scales, **not** proofs of observability: frictional attenuation, boundary conditions, sample interval, numerical dissipation, and modal cancellation also matter.

## 5. Discrete local observability

Linearizing the time-varying discrete system around a reference trajectory gives

$$
\delta\mathbf x_{k+1}=A_k\delta\mathbf x_k,
\qquad \delta\mathbf y_k=C_k\delta\mathbf x_k,
$$

where $A_k=\partial F_k/\partial\mathbf x_k$ and $C_k=\partial h_k/\partial\mathbf x_k$. With $\Phi_{0,0}=I$ and $\Phi_{k,0}=A_{k-1}\cdots A_0$ for $k\ge1$,

$$
H_N=\begin{bmatrix}
C_0\\C_1\Phi_{1,0}\\\vdots\\C_N\Phi_{N,0}
\end{bmatrix}.
$$

For measurement covariance $R=\mathrm{blockdiag}(R_0,\ldots,R_N)$, the discrete information Gramian is

$$
W_N=H_N^TR^{-1}H_N
=\sum_{k=0}^N\Phi_{k,0}^TC_k^TR_k^{-1}C_k\Phi_{k,0}.
$$

With positive-definite $R$, positive definiteness of $W_N$ is equivalent to full column rank of the whitened Jacobian. **Do not use a normal-equations Gramian to estimate numerical rank:** it squares the condition number.

### 5.1 Scaling and whitening

Use dimensionless optimization coordinates

$$
\boldsymbol\xi=D_x^{-1}(\mathbf x_0-\mathbf x_{\mathrm{ref}})
$$

with diagonal physical scales $D_x$ (illustratively $10^5$ Pa for pressure and $10$ kg/s for flow). The whitened, dimensionless observation sensitivity is

$$
B_N=R^{-1/2}H_ND_x.
$$

Compute the SVD $B_N=U\Sigma V^T$. Small right-singular directions are combinations of initial-state variables poorly represented in the measurements. Report singular values and rank thresholds for explicitly stated scales and noise assumptions.

The M1.5b objective used normalization values $\sigma_{\dot m}=1$ kg/s and $\sigma_p=10^4$ Pa. These are **objective weights**, not experimentally validated sensor-noise standard deviations.

For a Gaussian prior on $\boldsymbol\xi$ with covariance $\Gamma_0$, a local Gaussian approximation gives posterior precision

$$
\Gamma_{\mathrm{post}}^{-1}\approx B_N^TB_N+\Gamma_0^{-1}.
$$

A prior can regularize an unobservable direction, but that information then comes from the prior rather than from the boundary measurements.

## 6. Verified numerical evidence: M1.5b

The benchmark compared a reference trajectory and one perturbed initial state. Both completed 8,000 steps.

| Window | Observation-time span | Contribution to objective |
|---|---|---:|
| 0–80 | 0–20 min | 5.3415934241208767 |
| 81–800 | 20 min–3 h 20 min | 0.082381054966959938 |
| 801–8000 | 3 h 20 min–33 h 20 min | $1.1947123543218275\times10^{-15}$ |

Total objective: **5.423974479087843**. Approximately **98.48%** of the tested misfit accumulated in the first 80 steps.

| Window | Maximum inlet-flow difference (kg/s) | Maximum outlet-pressure difference (Pa) |
|---|---:|---:|
| 0–80 | 0.880860006826 | 681.422968264 |
| 81–800 | 0.084053338577 | 146.494575984 |
| 801–8000 | $1.0235623904\times10^{-8}$ | $1.8133781850\times10^{-5}$ |

At the final time, maximum pressure-state and flow-state differences were approximately $1.807\times10^{-7}$ Pa and $1.027\times10^{-10}$ kg/s, respectively.

**Supported interpretation:** The boundary response to this particular initial perturbation decays strongly, and the observation-window objective decomposition is consistent.

**Not established:** Full column rank, smallest singular value, practical noise-limited reconstruction horizon, or a universal 20-minute observability threshold.

See [M1.5b numerical verification](M15B_LONG_TRAJECTORY_VERIFICATION.md) and the raw data in [`benchmarks/results-m15b`](../benchmarks/results-m15b/).

### 6.1 Separate conservation question

The reported maximum mass defect grew from about $5.58\times10^{-10}$ at 800 steps to $6.23\times10^{-8}$ at 8,000 steps. Its units, normalization, and numerical cause remain to be characterized. This is distinct from observability.

## 7. Illustrative conditioning example (not pipeline output)

Suppose two dimensionless state combinations have whitened singular values $10$ and $10^{-3}$. With unit-variance whitened measurement noise, formal least-squares standard deviations along those directions are about $0.1$ and $1000$, respectively. The Jacobian is full rank, yet one combination is practically unrecoverable without a strong prior.

If extending the horizon increases the smaller singular value from $10^{-3}$ to $10^{-2}$, its uncertainty falls tenfold to about $100$, still poor. Adding observations that do not sense the weak direction cannot fix that direction merely by increasing the sample count.

These singular values are **hypothetical numerical examples**, not measured results from this pipeline.

## 8. Proposed numerical campaign

### 8.1 Baseline and horizons

Preserve the validated forward simulator and discrete adjoint. Study $N=20,40,80,100,101,120,160,240,400,800$; explicitly document whether $t_0$ is observed, the admissible initial-state dimension, and any boundary constraints. Use multiple reference trajectories and perturbation directions.

### 8.2 Sensitivity construction

A single scalar-objective adjoint gradient is **not** the entire observation Jacobian. Construct $B_N$ by either (a) tangent-linear propagation of 201 independent initial-state seeds, or (b) output-specific adjoint seeds for Jacobian rows. Verify one implementation against the other or against selected centered finite differences. At 800 steps, a dense $1602\times201$ double-precision matrix uses about **2.6 MB**, so a direct SVD is feasible once the derivatives are correct.

### 8.3 Verification requirements

- Verify $\langle Bv,w\rangle\approx\langle v,B^Tw\rangle$ for independent test vectors.
- Compare selected Jacobian columns against centered finite-difference step-size sweeps.
- Confirm observation indexing, especially inclusion or exclusion of $t_0$.
- Verify all forward time steps completed and sensitivities use the same reference trajectory.
- Compare multiple physically justified state scalings and report explicit singular-value thresholds.
- Retain conservation diagnostics separately from the inverse objective.

### 8.4 Required results

For each horizon, report the Jacobian shape, numerical rank at several thresholds, singular-value spectrum, condition estimates, least-observable spatial modes, and noise-propagated uncertainty. Test zero-noise and noisy synthetic reconstructions, alternate demand excitation, sensor sampling intervals, and sensor locations. Report pressure and flow reconstruction errors separately. State accuracy tolerances **before** selecting a practical observation horizon.

## 9. Open questions

1. Are all 201 parameterized initial-state values physically independent under the boundary equations?
2. Do the selected boundary outputs observe all linearized modes?
3. Which spatial pressure/flow modes are least observable, and how does friction attenuate them?
4. How do boundary excitation, sampling, and sensor placement change the practical horizon?
5. Would an additional interior sensor be more valuable than a longer boundary observation record?
6. How do regularization and physically constrained initial-state representations change the result?

## 10. Provisional conclusions

- **20 minutes (80 steps):** Captures approximately 98.48% of the tested output misfit, but supplies at most 162 independent scalar observations for 201 unconstrained unknowns; it cannot establish full local column rank.
- **25 minutes (100 steps):** First count-compatible horizon if both outputs are measured at $t_0$. It is **not** an established reconstruction threshold.
- **About 4.63 minutes:** Nominal acoustic one-way transit time; physically informative but not decisive for observability.
- **Minimum practical reconstruction horizon:** **Unknown.** It must be determined from scaled Jacobian singular values, noise assumptions, and controlled reconstruction experiments.

This report provides the mathematical basis and experimental plan for determining that horizon without conflating sensitivity decay with observability.
