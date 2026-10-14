#!/usr/bin/env python3
"""M2.4-F2D-2: gradient-audited, locally constrained nonlinear adversaries.

Nonconvex optimization: feasible objectives are lower bounds on nonlinear worst cases,
not certified global maxima. Preserves F2D outputs in separate F2D-2 CSVs.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, NonlinearConstraint, minimize

# Reuse the previously validated oracle protocol and F2B/F2C-2 input parsing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m24_f2d_nonlinear_optimize import Oracle, ROOT, LABELS, N, read_directions


def feasible_record(oracle, x, tolerance=1e-8):
    v, r2, pmin, defect, _ = oracle.evaluate(x)
    norm = float(np.linalg.norm(x))
    rnorm = float(np.sqrt(max(r2, 0.0)))
    good = (norm <= 1.0 + tolerance and r2 <= 1.0 + tolerance
            and pmin > 0.0 and defect <= 1e-6)
    return good, norm, rnorm, v, pmin, defect


def gradient_audit(oracle, x, step, rtol, atol):
    """Check all 41 coordinates for 3 outputs and squared measurement norm."""
    analytic = oracle.evaluate(x)[4]
    numeric = np.empty((4, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = step
        vp, rp, *_ = oracle.evaluate(x + e)
        vm, rm, *_ = oracle.evaluate(x - e)
        numeric[:3, j] = (vp - vm) / (2 * step)
        numeric[3, j] = (rp - rm) / (2 * step)
    scale = atol + rtol * np.maximum(np.abs(analytic), np.abs(numeric))
    normalized = np.abs(analytic - numeric) / scale
    for i, label in enumerate((*LABELS, 'measurement_squared')):
        j = int(np.argmax(normalized[i]))
        print(f'gradient {label:20s} worst_coordinate={j:2d} '
              f'analytic={analytic[i,j]:.12g} numeric={numeric[i,j]:.12g} '
              f'normalized_error={normalized[i,j]:.5g}', flush=True)
    if not np.isfinite(normalized).all() or np.max(normalized) > 1.0:
        raise RuntimeError('Analytical gradient audit FAILED; optimization not started')
    print('Gradient audit PASS', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', type=Path, default=Path('build/state_finder_m24_f2d_oracle'))
    parser.add_argument('--maxiter', type=int, default=400)
    parser.add_argument('--gtol', type=float, default=1e-8)
    parser.add_argument('--xtol', type=float, default=1e-11)
    parser.add_argument('--barrier-tol', type=float, default=1e-10)
    parser.add_argument('--gradient-step', type=float, default=1e-5)
    parser.add_argument('--gradient-rtol', type=float, default=1e-3)
    parser.add_argument('--gradient-atol', type=float, default=1e-5)
    parser.add_argument('--skip-gradient-check', action='store_true')
    args = parser.parse_args()

    bounds_df = pd.read_csv(ROOT / 'exact_operational_bounds.csv')
    feasible_df = pd.read_csv(ROOT / 'adversarial_feasible_amplitudes.csv')
    directions = read_directions()
    oracle = Oracle(args.oracle)
    rows, coords = [], []
    try:
        if not args.skip_gradient_check:
            # A small interior perturbation avoids checking exactly on active boundaries.
            xcheck = 0.1 * directions['flow']
            gradient_audit(oracle, xcheck, args.gradient_step,
                           args.gradient_rtol, args.gradient_atol)

        # Bounds and both nonlinear constraints are marked keep_feasible.
        # The norm constraint is squared, matching the exact mathematical set.
        coordinate_bounds = Bounds(-np.ones(N), np.ones(N), keep_feasible=True)
        initial_ball = NonlinearConstraint(
            lambda x: np.dot(x, x), -np.inf, 1.0,
            jac=lambda x: 2.0 * np.asarray(x), keep_feasible=True)
        measurement_ball = NonlinearConstraint(
            lambda x: oracle.evaluate(x)[1], -np.inf, 1.0,
            jac=lambda x: oracle.evaluate(x)[4][3], keep_feasible=True)

        for q, quantity in enumerate(LABELS):
            target = bounds_df[(np.isclose(bounds_df.measurement_bound, 1.0)) &
                               (np.isclose(bounds_df.time_s, 600.0)) &
                               (bounds_df.quantity == quantity)]
            if len(target) != 1:
                raise ValueError(f'Missing F2B bound for {quantity}')
            index = int(target.iloc[0].worst_index)
            linear_bound = float(target.iloc[0].exact_bound)
            f = feasible_df[feasible_df.quantity == quantity]
            if len(f) != 1 or int(f.iloc[0].worst_index) != index:
                raise ValueError(f'Missing F2C-2 seed for {quantity}')
            seed = directions[quantity] * float(f.iloc[0].alpha_feasible)
            for sign in (1., -1.):
                x0 = sign * seed * 0.999
                # Some F2C-2 rays only established feasibility in the + direction.
                # Backtrack the negative ray independently before trust-constr.
                for _ in range(60):
                    try:
                        good, norm, rnorm, *_ = feasible_record(oracle, x0, tolerance=0.0)
                    except RuntimeError:
                        good = False
                    if good and norm < 1.0 and rnorm < 1.0:
                        break
                    x0 *= 0.8
                else:
                    raise RuntimeError(f'Unable to find feasible starting point: {quantity} {sign:+g}')

                def fun(x):
                    return -sign * oracle.evaluate(x)[0][q]

                def jac(x):
                    return -sign * oracle.evaluate(x)[4][q]

                # Preserve the known feasible starting point as a fallback.
                best_x = x0.copy()
                best_score = -fun(best_x)

                def callback(xk, state):
                    nonlocal best_x, best_score
                    good, *_ = feasible_record(oracle, xk)
                    score = -fun(xk)
                    if good and score > best_score:
                        best_x, best_score = np.array(xk, copy=True), score
                    return False

                start_calls = oracle.calls
                res = minimize(
                    fun, x0, jac=jac, method='trust-constr',
                    bounds=coordinate_bounds,
                    constraints=[initial_ball, measurement_ball],
                    callback=callback,
                    options=dict(maxiter=args.maxiter, gtol=args.gtol,
                                 xtol=args.xtol, barrier_tol=args.barrier_tol,
                                 verbose=0))
                good_final, *_ = feasible_record(oracle, res.x)
                if good_final and -fun(res.x) > best_score:
                    best_x, best_score = np.array(res.x, copy=True), -fun(res.x)
                good, norm, rnorm, values, pmin, defect = feasible_record(oracle, best_x)
                row = dict(quantity=quantity, worst_index=index, sign=int(sign),
                           solver_success=bool(res.success), solver_status=int(res.status),
                           solver_message=str(res.message), iterations=int(res.nit),
                           oracle_calls=oracle.calls - start_calls,
                           accepted_final=bool(good_final),
                           accepted_fallback=not (good_final and np.array_equal(best_x, res.x)),
                           initial_norm=norm, measurement_norm=rnorm,
                           min_pressure_Pa=float(pmin),
                           max_conservation_defect_kg=float(defect),
                           terminal_signed=float(values[q]),
                           terminal_abs=abs(float(values[q])),
                           signed_objective=float(sign * values[q]),
                           F2B_linear_bound=linear_bound,
                           nonlinear_to_F2B_ratio=abs(float(values[q])) / linear_bound,
                           feasible=bool(good))
                rows.append(row)
                coords.extend(dict(quantity=quantity, sign=int(sign), coordinate=j,
                                   value=float(value)) for j, value in enumerate(best_x))
                print(f'{quantity:9s} sign={sign:+.0f} feasible={good} '
                      f'converged={res.success} iters={res.nit} '
                      f'|v|={norm:.10g} |r|={rnorm:.10g} '
                      f'|target|={abs(values[q]):.12g} '
                      f'ratio={row["nonlinear_to_F2B_ratio"]:.9g}', flush=True)
        ROOT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(ROOT / 'f2d2_trust_constr_optima.csv', index=False)
        pd.DataFrame(coords).to_csv(ROOT / 'f2d2_trust_constr_directions.csv', index=False)
        print('\nBest feasible solutions by quantity (local; not globally certified):')
        df = pd.DataFrame(rows)
        for quantity in LABELS:
            sub = df[(df.quantity == quantity) & (df.feasible)]
            if sub.empty:
                print(f'{quantity}: NO FEASIBLE SOLUTION')
            else:
                best = sub.loc[sub.terminal_abs.idxmax()]
                print(f'{quantity}: {best.terminal_abs:.12g} '
                      f'({best.nonlinear_to_F2B_ratio:.9g} x F2B), '
                      f'sign={best.sign:+d}, converged={best.solver_success}')
        print(f'Oracle evaluations: {oracle.calls}')
    finally:
        oracle.close()


if __name__ == '__main__':
    main()
