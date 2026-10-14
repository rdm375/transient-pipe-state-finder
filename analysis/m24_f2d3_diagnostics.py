#!/usr/bin/env python3
"""M2.4-F2D-3: gradient convergence, constraint geometry, KKT and warm starts.

Fixed terminal target index per quantity. Local diagnostics, no global certificate.
Runs from the repository root; requires existing F2D/F2D-2 CSVs and oracle.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m24_f2d_nonlinear_optimize import Oracle, ROOT, LABELS, N, read_directions


def is_true(v):
    return str(v).strip().lower() in ('true', '1', 'yes')


def candidates():
    sources = [
        ('f2d1', 'f2d_nonlinear_local_optima.csv',
         'f2d_nonlinear_local_optima_directions.csv'),
        ('f2d2', 'f2d2_trust_constr_optima.csv',
         'f2d2_trust_constr_directions.csv'),
    ]
    for name, summary, coords in sources:
        a, b = ROOT / summary, ROOT / coords
        if not (a.is_file() and b.is_file()):
            print(f'SKIP {name}: missing CSV pair', flush=True)
            continue

        rows, vectors = pd.read_csv(a), pd.read_csv(b)

        for _, row in rows.iterrows():
            if not is_true(row.get('feasible', False)):
                continue

            quantity, sign = str(row.quantity), int(row.sign)
            subset = vectors[
                (vectors.quantity == quantity) &
                (vectors.sign == sign)
            ].sort_values('coordinate')

            if len(subset) != N or not np.array_equal(
                subset.coordinate.to_numpy(), np.arange(N)
            ):
                raise ValueError(
                    f'Incomplete vector {name} {quantity} {sign}'
                )

            yield (
                name, quantity, sign,
                subset.value.to_numpy(dtype=float)
            )


def evaluation(oracle, x, q, sign, bound):
    values, r2, pmin, defect, grads = oracle.evaluate(x)
    score = float(sign * values[q] / bound)
    return score, float(r2), float(pmin), float(defect), grads


def diagnostic(oracle, x, q, sign, bound, active_tol):
    score, r2, pmin, defect, grads = evaluation(
        oracle, x, q, sign, bound
    )

    c = np.array([float(x @ x), r2])
    jac = np.vstack((2 * x, grads[3]))

    singular = np.linalg.svd(jac, compute_uv=False)

    n0 = np.linalg.norm(jac[0])
    n1 = np.linalg.norm(jac[1])

    cosine = (
        float(jac[0] @ jac[1] / (n0 * n1))
        if n0 * n1 else np.nan
    )

    # Minimize f = -sign*target/bound with c_i <= 1.
    # Stationarity: grad(f) + J_active.T @ lambda = 0.
    grad_f = -sign * grads[q] / bound

    active = np.flatnonzero(c >= 1 - active_tol)
    multipliers = np.zeros(2)

    if len(active):
        lam, _ = nnls(jac[active].T, -grad_f)
        multipliers[active] = lam

    stationarity = grad_f + jac.T @ multipliers

    return dict(
        score=score,
        initial_norm=float(np.linalg.norm(x)),
        measurement_norm=float(np.sqrt(max(r2, 0))),
        initial_violation=max(0., c[0] - 1.),
        measurement_violation=max(0., c[1] - 1.),
        min_pressure_Pa=pmin,
        conservation_defect_kg=defect,
        active_initial=bool(0 in active),
        active_measurement=bool(1 in active),
        constraint_sigma_max=float(singular[0]),
        constraint_sigma_min=float(singular[-1]),
        constraint_condition=(
            float(singular[0] / singular[-1])
            if singular[-1] > 0 else np.inf
        ),
        constraint_cosine=cosine,
        multiplier_initial=float(multipliers[0]),
        multiplier_measurement=float(multipliers[1]),
        kkt_stationarity_inf=float(
            np.linalg.norm(stationarity, ord=np.inf)
        ),
        kkt_stationarity_2=float(np.linalg.norm(stationarity)),
        kkt_complementarity_inf=float(
            np.max(np.abs(multipliers * (c - 1)))
        ),
        grad_objective_norm=float(np.linalg.norm(grad_f)),
    )


def audit(oracle, x, steps):
    exact = oracle.evaluate(x)[4]
    rows = []

    for h in steps:
        numeric = np.zeros_like(exact)

        for j in range(N):
            delta = np.zeros(N)
            delta[j] = h

            vp, rp, *_ = oracle.evaluate(x + delta)
            vm, rm, *_ = oracle.evaluate(x - delta)

            numeric[:3, j] = (vp - vm) / (2 * h)
            numeric[3, j] = (rp - rm) / (2 * h)

        for i, label in enumerate(
            (*LABELS, 'measurement_squared')
        ):
            difference = exact[i] - numeric[i]
            j = int(np.argmax(np.abs(difference)))

            rows.append(dict(
                step=h,
                quantity=label,
                worst_coordinate=j,
                analytic=float(exact[i, j]),
                finite_difference=float(numeric[i, j]),
                max_absolute_error=float(
                    np.max(np.abs(difference))
                ),
                relative_l2_error=float(
                    np.linalg.norm(difference) /
                    max(np.linalg.norm(exact[i]), 1e-30)
                ),
                rms_error=float(
                    np.linalg.norm(difference) / np.sqrt(N)
                ),
            ))

        print(
            f'gradient step={h:.1e} ' +
            ' '.join(
                f'{label}={rows[-4+i]["relative_l2_error"]:.3e}'
                for i, label in enumerate(
                    (*LABELS, 'measurement_squared')
                )
            ),
            flush=True,
        )

    return rows


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--oracle', type=Path,
        default=Path('build/state_finder_m24_f2d_oracle')
    )
    p.add_argument('--active-tol', type=float, default=1e-3)
    p.add_argument(
        '--gradient-steps', type=float, nargs='+',
        default=[1e-2, 1e-3, 1e-4, 1e-5]
    )
    p.add_argument(
        '--warmstart', action='store_true',
        help='Run scaled SLSQP from best feasible prior point'
    )
    p.add_argument('--maxiter', type=int, default=300)
    p.add_argument('--ftol', type=float, default=1e-10)

    args = p.parse_args()

    if any(h <= 0 for h in args.gradient_steps):
        p.error('gradient steps must be positive')

    if args.active_tol <= 0:
        p.error('active-tol must be positive')

    bounds_df = pd.read_csv(
        ROOT / 'exact_operational_bounds.csv'
    )

    linear = {}

    for quantity in LABELS:
        sub = bounds_df[
            np.isclose(bounds_df.measurement_bound, 1.) &
            np.isclose(bounds_df.time_s, 600.) &
            (bounds_df.quantity == quantity)
        ]

        if len(sub) != 1:
            raise ValueError(
                f'Expected one F2B bound for {quantity}'
            )

        linear[quantity] = float(sub.iloc[0].exact_bound)

    prior = list(candidates())

    if not prior:
        raise RuntimeError(
            'No feasible F2D-1 or F2D-2 CSV candidates found'
        )

    oracle = Oracle(args.oracle)
    reports = []
    warmrows = []
    warmcoords = []

    try:
        # Interior point for finite-difference evaluation.
        xcheck = 0.1 * read_directions()['flow']

        audit_rows = audit(
            oracle, xcheck, args.gradient_steps
        )

        for name, quantity, sign, x in prior:
            q = LABELS.index(quantity)

            d = diagnostic(
                oracle, x, q, sign,
                linear[quantity], args.active_tol
            )

            reports.append(dict(
                source=name,
                quantity=quantity,
                sign=sign,
                **d
            ))

            print(
                f'{name:4s} {quantity:9s} '
                f'sign={sign:+d} '
                f'ratio={d["score"]:.9f} '
                f'|v|={d["initial_norm"]:.9g} '
                f'|r|={d["measurement_norm"]:.9g} '
                f'cond={d["constraint_condition"]:.3e} '
                f'KKTinf={d["kkt_stationarity_inf"]:.3e}',
                flush=True
            )

        ROOT.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(audit_rows).to_csv(
            ROOT / 'f2d3_gradient_step_sweep.csv',
            index=False
        )

        pd.DataFrame(reports).to_csv(
            ROOT / 'f2d3_kkt_diagnostics.csv',
            index=False
        )

        if args.warmstart:
            for quantity in LABELS:
                q = LABELS.index(quantity)

                for sign in (1, -1):
                    group = [
                        (name, x)
                        for name, qty, s, x in prior
                        if qty == quantity and s == sign
                    ]

                    if not group:
                        continue

                    name, x0 = max(
                        group,
                        key=lambda pair: evaluation(
                            oracle, pair[1], q, sign,
                            linear[quantity]
                        )[0]
                    )

                    x0 = x0.copy()

                    # Move the warm start strictly inside
                    # both uncertainty constraints.
                    for _ in range(100):
                        _, r2, pmin, defect, _ = evaluation(
                            oracle, x0, q, sign,
                            linear[quantity]
                        )

                        if (
                            x0 @ x0 < 1 - 1e-9
                            and r2 < 1 - 1e-9
                            and pmin > 0
                            and defect <= 1e-6
                        ):
                            break

                        x0 *= 0.995

                    else:
                        raise RuntimeError(
                            'Cannot make feasible warm start: '
                            f'{quantity} {sign}'
                        )

                    best = x0.copy()
                    bestscore = evaluation(
                        oracle, best, q, sign,
                        linear[quantity]
                    )[0]

                    def fun(x):
                        return (
                            -sign * oracle.evaluate(x)[0][q] /
                            linear[quantity]
                        )

                    def jac(x):
                        return (
                            -sign * oracle.evaluate(x)[4][q] /
                            linear[quantity]
                        )

                    constraints = [
                        dict(
                            type='ineq',
                            fun=lambda x: 1 - float(x @ x),
                            jac=lambda x: -2 * np.asarray(x),
                        ),
                        dict(
                            type='ineq',
                            fun=lambda x: 1 - oracle.evaluate(x)[1],
                            jac=lambda x: -oracle.evaluate(x)[4][3],
                        ),
                    ]

                    def callback(x):
                        nonlocal best, bestscore

                        try:
                            val, r2, pmin, defect, _ = evaluation(
                                oracle, x, q, sign,
                                linear[quantity]
                            )

                            if (
                                x @ x <= 1 + 1e-10
                                and r2 <= 1 + 1e-10
                                and pmin > 0
                                and defect <= 1e-6
                                and val > bestscore
                            ):
                                best = x.copy()
                                bestscore = val

                        except RuntimeError:
                            pass

                    result = minimize(
                        fun,
                        x0,
                        jac=jac,
                        method='SLSQP',
                        bounds=[(-1., 1.)] * N,
                        constraints=constraints,
                        callback=callback,
                        options=dict(
                            maxiter=args.maxiter,
                            ftol=args.ftol,
                            disp=False
                        )
                    )

                    callback(result.x)

                    d = diagnostic(
                        oracle, best, q, sign,
                        linear[quantity], args.active_tol
                    )

                    warmrows.append(dict(
                        quantity=quantity,
                        sign=sign,
                        seed_source=name,
                        solver_success=bool(result.success),
                        solver_status=int(result.status),
                        iterations=int(result.nit),
                        **d
                    ))

                    warmcoords.extend(
                        dict(
                            quantity=quantity,
                            sign=sign,
                            coordinate=j,
                            value=float(z)
                        )
                        for j, z in enumerate(best)
                    )

                    print(
                        f'warmstart {quantity:9s} '
                        f'sign={sign:+d} '
                        f'ratio={d["score"]:.9f} '
                        f'success={result.success} '
                        f'KKTinf={d["kkt_stationarity_inf"]:.3e}',
                        flush=True
                    )

            pd.DataFrame(warmrows).to_csv(
                ROOT / 'f2d3_warmstart_optima.csv',
                index=False
            )

            pd.DataFrame(warmcoords).to_csv(
                ROOT / 'f2d3_warmstart_directions.csv',
                index=False
            )

        print(
            f'Oracle evaluations: {oracle.calls}',
            flush=True
        )

    finally:
        oracle.close()


if __name__ == '__main__':
    main()
