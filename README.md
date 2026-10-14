# Transient Pipe State Finder

**Physics-constrained initial-state reconstruction, observability analysis, and operational uncertainty quantification for a single isothermal gas pipeline.**

**Status:** research software; not a validated production estimator. The CMake project version is `0.1.0`, but that is not a declaration of release readiness. **License:** AGPL-3.0-only.

## 1. The problem: infer the interior from the boundary

Suppose a gas pipeline has been operating for some time, but its internal pressure and mass-flow profiles are not known accurately. We can prescribe the inlet pressure and outlet mass-flow histories and measure the resulting inlet mass flow and outlet pressure. Can those boundary measurements tell us what the entire pipe looked like initially?

This is an **inverse initial-value problem**. Unlike a forward simulation, in which the initial state determines the subsequent trajectory, the unknown initial state must be inferred from its effects on a comparatively small set of measurements. The problem is often ill-conditioned: two very different internal states may produce nearly indistinguishable boundary histories.

Our objective is therefore broader than finding a good least-squares fit. We want to determine **which parts of the state are observable**, how uncertainty propagates into operational predictions, and whether numerical results respect the conservation laws of the forward model.

The present scope is one isothermal compressible-gas pipe with prescribed inlet pressure and outlet mass flow. A pipe network, independently established field-data accuracy, and global certification of nonlinear uncertainty bounds are outside the present claims.

## 2. Forward dynamics and conservation

Let $x$ denote distance along a pipe of length $L$, $t$ time, $p(x,t)$ pressure, $\rho(x,t)$ density, and $\dot m(x,t)$ mass flow. With cross-sectional area $A$, the one-dimensional mass balance is

$$
\frac{\partial (A\rho)}{\partial t}
+\frac{\partial \dot m}{\partial x}=0.
$$

The momentum balance combines pressure forces, transport of momentum, and wall friction. Its precise discrete form, along with the gas-property assumptions, is supplied by the companion [`isothermal-pipe-sim`](../isothermal-pipe-sim) solver. The State Finder reuses that C++ forward model rather than independently implementing a second, potentially inconsistent set of physics.

### Staggered spatial variables

Pressure is stored at $n+1$ nodes, and mass flow at $n$ interior faces. The initial inlet mass flow is an additional unknown. The initial inlet pressure is prescribed and is **not** an optimization coordinate. Consequently, a natural initial-state vector contains $2n+1$ independent quantities:

$$
\boldsymbol{x}_0 =
\left(p_1,\ldots,p_n,\dot m_0,\ldots,\dot m_{n-1},\dot m_{\mathrm{in},0}\right)^T.
$$

The implementation uses explicit coordinate layouts and scaling to keep the physical units of pressure and flow from distorting numerical optimization.

### Time discretization

The forward integrator uses a $\theta$-method, with $\theta$ reserved throughout this project for the time-weighting parameter. If $\boldsymbol{u}_k$ denotes the full discrete state at time $t_k$, a step can be written abstractly as

$$
\boldsymbol{R}_k(\boldsymbol{u}_k,\boldsymbol{u}_{k-1})=\boldsymbol{0}.
$$

This is an *implicit* equation: the new state is found through a nonlinear solve. The residual formulation is also the starting point for the tangent-linear and adjoint calculations below.

### Linepack and mass conservation

Linepack $M(t)$ is the total mass stored in the pipe. Summing the discrete continuity equations, including the boundary half-cells, gives the inventory balance

$$
\frac{M_{k}-M_{k-1}}{\Delta t}
=\dot m_{\mathrm{in},k}^{\theta}
-\dot m_{\mathrm{out},k}^{\theta}
+\varepsilon_{M,k},
$$

where the superscript indicates $\theta$-weighted boundary flow and $\varepsilon_{M,k}$ is the **numerical conservation defect**. The physical inlet–outlet flow imbalance is not itself an error: it causes the pipe's inventory to change. The defect measures failure of the *numerical* balance and is reported separately.

## 3. Formulating the inverse problem

Let $\boldsymbol{\xi}\in\mathbb{R}^{2n+1}$ be a dimensionless representation of the unknown initial state, and let $\mathcal{F}(\boldsymbol{\xi})$ denote the trajectory produced by the nonlinear forward solver. An observation operator $\mathcal{H}$ extracts the simulated inlet mass flow and outlet pressure at measurement times:

$$
\boldsymbol{y}_{\mathrm{pred}}(\boldsymbol{\xi})
=\mathcal{H}(\mathcal{F}(\boldsymbol{\xi})).
$$

The inlet pressure and outlet mass flow are **prescribed boundary inputs**; they are not confused with the two measured output channels. Given measurements $\boldsymbol{y}$ and a positive diagonal uncertainty matrix $\Sigma$, define the normalized residual

$$
\boldsymbol{r}(\boldsymbol{\xi})
=\Sigma^{-1}\left(\boldsymbol{y}_{\mathrm{pred}}(\boldsymbol{\xi})-\boldsymbol{y}\right).
$$

A representative reconstruction objective is

$$
J(\boldsymbol{\xi})
=\frac12\|\boldsymbol{r}(\boldsymbol{\xi})\|_2^2
+\frac{\lambda}{2}\|L(\boldsymbol{\xi}-\boldsymbol{\xi}_{\mathrm{ref}})\|_2^2.
$$

Here $L$ may encode a simple Tikhonov penalty or a spatial smoothness preference; $\lambda\geq0$ controls regularization. If the scales in $\Sigma$ are measured standard deviations, the data-fit term has a statistical interpretation under a Gaussian noise model. If they are merely tuning weights, it does **not** automatically acquire that interpretation.

A small value of $J$ does not imply that the initial state is uniquely or accurately recovered. That requires an observability analysis.

## 4. Differentiating an implicit simulator

A naive approach would perturb every initial coordinate, rerun the entire transient simulation, and estimate each gradient component by finite differences. This is useful as a check, but it is expensive and susceptible to truncation and roundoff error.

Instead, the C++ implementation includes **tangent-linear** and **discrete-adjoint** sensitivity calculations derived from the implicit time-step residual.

Set

$$
A_k=\frac{\partial\boldsymbol{R}_k}{\partial\boldsymbol{u}_k},
\qquad
B_k=\frac{\partial\boldsymbol{R}_k}{\partial\boldsymbol{u}_{k-1}}.
$$

Differentiating $\boldsymbol{R}_k=\boldsymbol{0}$ in a perturbation direction gives

$$
A_k\,\delta\boldsymbol{u}_k
=-B_k\,\delta\boldsymbol{u}_{k-1}.
$$

This **tangent-linear equation** propagates a directional derivative forward through time without differencing two complete nonlinear trajectories.

For a scalar objective accumulated along the trajectory, a discrete adjoint solves transposed linear systems backward through time. With one consistent sign convention, the adjoint recursion has the form

$$
A_k^T\boldsymbol{\lambda}_k
=\boldsymbol{g}_k-B_{k+1}^T\boldsymbol{\lambda}_{k+1},
$$

where $\boldsymbol{g}_k$ is the direct derivative of the objective with respect to the state at step $k$. The resulting initial-state gradient can be obtained at a cost that does not grow linearly with the number of optimization coordinates in the way coordinate-by-coordinate finite differences do.

### What derivative tools are actually used?

The **production-oriented C++ derivative path is not simply “JAX differentiation.”** The discrete adjoint reuses the forward solver's **analytic banded Jacobian** (`assemble_jacobian_banded`), performs transposed banded solves through **LAPACK**, and applies a previous-state vector–Jacobian product (`old_state_vjp`). The tangent-linear implementation propagates directions through the same implicit dynamics. This combination is closely related to the mathematics of automatic differentiation, but the verified implementation should be described by the specific analytic and matrix-free operations it performs.

The Python environment includes **NumPy, SciPy, and JAX**. Python research drivers use numerical optimization and analysis tools, including SciPy's constrained optimizers; their presence does not mean every C++ derivative is generated by JAX. Derivative verification and the exact roles of any additional automatic-differentiation tools should be documented against the corresponding source and tests, rather than inferred from dependencies.

## 5. Observability: what can boundary data determine?

Near a reference initial state $\boldsymbol{\xi}_*$, linearize the normalized observation map:

$$
\boldsymbol{r}(\boldsymbol{\xi}_*+\delta\boldsymbol{\xi})
\approx\boldsymbol{r}(\boldsymbol{\xi}_*)+S\,\delta\boldsymbol{\xi},
\qquad
S=\frac{\partial\boldsymbol{r}}{\partial\boldsymbol{\xi}}.
$$

The singular value decomposition $S=U\Lambda V^T$ separates initial-state directions by how strongly they affect the observations. Large singular values correspond to strongly measured directions; small singular values correspond to weakly constrained combinations of pressure and flow. A zero singular value identifies a direction invisible to the **linearized** measurements.

The normal or information matrix is

$$
G=S^TS.
$$

Its eigenvectors are the right singular vectors of $S$, and its eigenvalues are the squared singular values. When there are fewer independent scalar measurements than initial-state unknowns, elementary rank counting already proves that the linearized map has a nontrivial nullspace.

For quadratic regularization, a Gauss–Newton step solves a system of the schematic form

$$
\left(S^TS+\lambda L^TL\right)\delta\boldsymbol{\xi}
=-S^T\boldsymbol{r}
-\lambda L^TL(\boldsymbol{\xi}-\boldsymbol{\xi}_{\mathrm{ref}}).
$$

The project studies information spectra, regularized reconstruction, predictive observability, and **matrix-free** operations. Matrix-free methods compute products such as $S\boldsymbol{v}$ and $S^T\boldsymbol{w}$ without requiring the full sensitivity matrix to be stored for every large-scale calculation. Small explicit matrices remain useful for verification.

Importantly, **regularization chooses among weakly distinguishable states; it does not create information that was absent from the measurements.**

## 6. From state uncertainty to operational uncertainty

State estimation is not the final goal. Operators may care more about downstream pressure, mass flow, or inventory at a specified future time than about every initial coordinate individually.

Let $q(\boldsymbol{\xi})$ denote one such terminal quantity. In a local linear approximation,

$$
q(\boldsymbol{\xi}_*+\delta\boldsymbol{\xi})
\approx q(\boldsymbol{\xi}_*)+
\nabla q(\boldsymbol{\xi}_*)^T\delta\boldsymbol{\xi}.
$$

The M2.4 experiments extend this idea to constrained perturbations. In one studied configuration, the initial perturbation has **41 scaled coordinates**, the prescribed boundary histories are fixed, and admissible perturbations satisfy

$$
\|\boldsymbol{v}\|_2\leq1,
\qquad
\|\boldsymbol{r}_{\mathrm{NL}}(\boldsymbol{v})\|_2\leq1.
$$

Here $\boldsymbol{r}_{\mathrm{NL}}$ is evaluated using the nonlinear forward model. The research code explores perturbations that are consistent with the chosen initial-state and measurement constraints yet produce large terminal changes in **pressure, flow, or inventory**. Python drivers communicate with a persistent C++ nonlinear oracle and use SciPy optimization; candidate solutions are replayed and checked for numerical feasibility and conservation.

These experiments produce **feasible adversarial witnesses**, not certified global maximizers. Their results are conditional on the model, chosen scales, measurement window, and boundary histories. The experiments do not establish accuracy against independent field data.

## 7. What is implemented?

The research branch contains, among other components:

- A C++20 forward-model wrapper and shared nonlinear trajectory recording, including Newton iteration and mass-defect diagnostics.
- Tangent-linear propagation and a discrete adjoint with analytic Jacobians, vector–Jacobian products, and LAPACK banded solves.
- Uncertainty-scaled observation residuals, gradient-based optimization, and regularization experiments.
- Initial-state observability and information-spectrum analysis, including predictive and matrix-free studies.
- Reconstruction experiments involving measurement uncertainty, regularization, and synthetic exact-state recovery.
- Operational sensitivity and nonlinear constrained adversarial studies for terminal pressure, flow, and inventory.
- Python analysis and reproducibility artifacts: CSV tables, selected figures, experiment drivers, and validation records.

These are **research capabilities**, not a promise of a stable end-user command-line or library API. See `docs/RELEASE_CANDIDATE_CHECKLIST.md` for the outstanding release contract.

## 8. Build and test

The C++ project uses **CMake**, **C++20**, and **LAPACK**, and reuses the sibling `isothermal-pipe-sim` source tree. With both repositories checked out under the same parent directory:

```bash
cmake -S transient-pipe-state-finder -B transient-pipe-state-finder/build
cmake --build transient-pipe-state-finder/build -j
ctest --test-dir transient-pipe-state-finder/build --output-on-failure
```

Alternatively, supply the forward solver explicitly with `-DPIPE_SIM_SOURCE_DIR=/path/to/isothermal-pipe-sim`. The compatible forward-solver commit should be pinned and recorded for reproducible releases; a sibling directory by itself is **not** a dependency lock.

Python analysis scripts use dependencies including **NumPy, SciPy, pandas, and Matplotlib**; `pyproject.toml` additionally declares **JAX**. The two dependency declarations should be reconciled and locked as part of the reproducibility work. Many research scripts require previously generated benchmark files and compiled C++ executables and are not intended to run as independent entry points from a fresh checkout.

## 9. Scientific limits and development priorities

- **Identifiability:** an excellent boundary fit can coexist with substantial uncertainty in weakly observed internal modes.
- **Model dependence:** inverse and predictive results inherit the assumptions and numerical errors of the forward solver.
- **Nonlinear optimization:** feasible adversarial solutions do not certify global worst cases.
- **Scope:** the demonstrated model is a single isothermal pipe; network-scale inference and field validation remain separate tasks.
- **Release status:** configuration validation, public interfaces, clean-checkout reproducibility, CI, and user-facing diagnostics remain work in progress.

The development roadmap prioritizes reproducibility, an independently readable mathematical report, regression automation, stable interfaces, and a release process that keeps **research milestones** distinct from **production acceptance**.

## License

GNU Affero General Public License v3.0 **only** (`AGPL-3.0-only`). No additional linking exception.
