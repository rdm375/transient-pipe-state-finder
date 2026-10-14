# Mathematical foundations of transient pipeline state estimation

**Audience:** senior undergraduate mathematics students familiar with multivariable calculus, ordinary differential equations, linear algebra, and elementary numerical analysis.

**Scope:** conceptual formulation of the research problem. The exact equation of state, momentum closure, spatial discretization, and scaling constants must be checked against the source code before this document is treated as a complete model specification.

## 1. State, inputs, and observations

Consider a pipe partitioned into spatial cells. The unknown initial state is represented by a finite-dimensional vector

\[
 x_0 \in \mathbb R^n,
\]

whose entries encode initial nodal pressure or density and staggered mass flow. The coordinate layout and units are properties of the selected discretization. The model is driven by prescribed inlet-pressure and outlet-mass-flow histories, collectively denoted by \(u(t)\).

Let \(\mathcal F\) denote the discrete forward evolution operator. Given \(x_0\) and \(u\), it produces a trajectory \(x^0,\ldots,x^N\). An observation map \(\mathcal H\) extracts the inlet mass flow and outlet pressure at measurement times. The predicted measurement vector is

\[
 F(x_0)=\mathcal H(\mathcal F(x_0,u)) \in \mathbb R^m.
\]

The actual measurements satisfy an idealized error model

\[
 y=F(x_0^\star)+\varepsilon.
\]

Here \(x_0^\star\) is the unknown true initial state and \(\varepsilon\) represents measurement errors. A prescribed boundary input is **not** an independently fitted observation residual.

## 2. Weighted nonlinear least squares

Choose a reference initial state \(x_{\mathrm{ref}}\), an invertible state scaling matrix \(S_x\), and dimensionless coordinates \(z\) by

\[
 x_0=x_{\mathrm{ref}}+S_x z.
\]

If the measurement covariance is represented by a positive definite matrix \(\Sigma\), a conventional reconstruction objective is

\[
 \min_z\; \frac12\|\Sigma^{-1/2}[F(x_{\mathrm{ref}}+S_x z)-y]\|_2^2
       +\frac{\lambda}{2}\|L(z-z_{\mathrm{prior}})\|_2^2.
\]

The first term penalizes mismatch with observations; the second penalizes deviations from a prior or spatially rough states. The matrix \(L\) may be the identity or a discrete derivative. This equation describes a general formulation, not a claim that every experiment uses the same weighting and regularizer.

## 3. Linearization and identifiability

Near a candidate \(\bar z\), differentiability gives

\[
 F(x_{\mathrm{ref}}+S_x(\bar z+\delta z))
 \approx F(x_{\mathrm{ref}}+S_x\bar z)+J\delta z,
\]

where \(J\in\mathbb R^{m\times n}\) is the Jacobian with respect to scaled state coordinates. Define the whitened Jacobian \(A=\Sigma^{-1/2}J\). Its singular value decomposition is

\[
 A=U\mathrm{diag}(\sigma_1,\ldots,\sigma_r)V^T,
 \qquad \sigma_1\ge\cdots\ge\sigma_r>0.
\]

A right singular vector associated with a small singular value is a state direction that produces little change in the observations. If \(A v=0\), then that direction is invisible to the *linearized* measurements. A small nonlinear residual therefore does not establish unique recovery of the state.

For the unregularized linearized problem, \(A^TA\) is singular whenever \(\mathrm{rank}(A)<n\). With quadratic regularization, the local normal equations take the form

\[
 (A^TA+\lambda L^TL)\delta z
 =-A^T r-\lambda L^TL(\bar z-z_{\mathrm{prior}}),
\]

where \(r=\Sigma^{-1/2}[F(x_{\mathrm{ref}}+S_x\bar z)-y]\). Regularization can stabilize a reconstruction, but it also introduces dependence on the prior and \(L\).

## 4. Matrix-free sensitivities

An explicit Jacobian is not necessary for many iterative methods. A tangent-linear calculation evaluates \(Jv\): the first-order change in observations caused by a state perturbation \(v\). An adjoint calculation evaluates \(J^Tw\): the gradient of a weighted observation functional with respect to initial-state coordinates.

These operations permit matrix-free products

\[
 (J^TWJ+\lambda L^TL)v
 =J^T[W(Jv)]+\lambda L^T(Lv).
\]

An important numerical check is the adjoint identity

\[
 \langle Jv,w\rangle=\langle v,J^Tw\rangle,
\]

with the inner products and scaling conventions explicitly specified. Passing this check supports implementation consistency; it does not, by itself, prove the physical model correct.

## 5. Mass conservation and linepack

Let \(M^k\) be total discrete pipe inventory (linepack) at time index \(k\), and let \(\dot m_{\mathrm{in}}\) and \(\dot m_{\mathrm{out}}\) be boundary mass-flow rates. For a conservative theta-weighted step, the expected discrete identity is

\[
 \frac{M^{k+1}-M^k}{\Delta t}
 =\dot m_{\mathrm{in}}^{\theta}-\dot m_{\mathrm{out}}^{\theta},
\]

where \(\dot m^{\theta}=(1-\theta)\dot m^k+\theta\dot m^{k+1}\). Here \(\theta\) is reserved for the time-integration weight.

The **physical mass imbalance** is the right-hand side. The **numerical conservation defect** is the difference between the two sides. A nonzero physical imbalance may be entirely correct; a large numerical defect signals a discretization or solver problem.

## 6. Operational uncertainty and adversarial directions

After a reconstruction, one may ask how much a terminal quantity \(Q(z)\)—such as pressure, mass flow, or linepack—could change while respecting uncertainty and measurement-consistency constraints.

For a scaled perturbation \(v\in\mathbb R^n\), consider

\[
 \max_v\; s[Q(\bar z+v)-Q(\bar z)],\qquad s\in\{-1,+1\},
\]

subject to an initial-state uncertainty ball \(\|v\|_2\le1\), a normalized nonlinear observation-residual constraint \(\|r_{\mathrm{NL}}(v)\|_2\le1\), and physical/numerical admissibility checks.

The linearized approximation replaces the nonlinear response by its derivative. A nonlinear candidate satisfying the constraints is a **feasible witness** and therefore a lower bound on the corresponding maximization problem's true supremum. It is not automatically a global maximizer. The M2.4-F2D-5 closure checks strict *numerical* feasibility of selected witnesses using the same nonlinear oracle; it is not an independent-physics or interval-arithmetic proof.

## 7. Limits of inference

The mathematical distinctions to retain throughout the project are: (i) a good fit versus an identifiable state; (ii) physical imbalance versus conservation defect; (iii) linearized uncertainty versus nonlinear feasible witnesses; and (iv) verified implementation identities versus validation against independent physical evidence.
