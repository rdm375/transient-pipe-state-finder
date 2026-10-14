#!/usr/bin/env python3
"""M2.4-F2D: multi-start, locally optimized nonlinear operational adversaries.

Optimization is over 41 scaled initial-state coordinates; boundary histories fixed.
Constraints: ||v||_2 <= 1 and ||r_nonlinear(v)||_2 <= 1.
This is a nonconvex problem: results are feasible lower bounds, NOT global maxima.
Run from repository root after generating M2.4-F2B and F2C-2 CSVs.
"""
import argparse
import csv
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path('benchmarks/results-m24')
N = 41
LABELS = ('pressure', 'flow', 'inventory')

class Oracle:
    def __init__(self, executable):
        self.process = subprocess.Popen([str(executable.resolve())], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=None, text=True, bufsize=1)
        ready = self.process.stdout.readline().strip()
        if ready != 'READY 41 3':
            raise RuntimeError(f'Unexpected oracle handshake: {ready!r}')
        self.last_x = None
        self.last_result = None
        self.calls = 0

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (N,) or not np.isfinite(x).all():
            raise ValueError('Invalid initial state')
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return self.last_result
        self.process.stdin.write(','.join(format(float(z), '.17g') for z in x) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline().strip()
        if not line.startswith('OK,'):
            print(
                "F2D ORACLE FAILURE",
                file=__import__('sys').stderr,
                flush=True,
            )
            print(
                "Trial point:",
                np.array2string(
                    np.asarray(x),
                    precision=17,
                    separator=", ",
                    max_line_width=160,
                ),
                file=__import__('sys').stderr,
                flush=True,
            )
            print(
                "Initial norm:",
                np.linalg.norm(x),
                file=__import__('sys').stderr,
                flush=True,
            )
            raise RuntimeError(f'Nonlinear oracle failed: {line}')
        values = np.fromstring(line[3:], sep=',')
        if values.size != 6 + 4*N or not np.isfinite(values).all():
            raise RuntimeError(f'Invalid oracle response ({values.size} values)')
        self.last_x = x.copy()
        self.last_result = (values[:3], values[3], values[4], values[5],
                            values[6:].reshape(4, N))
        self.calls += 1
        return self.last_result

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('QUIT\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)


def read_directions():
    df = pd.read_csv(ROOT/'adversarial_initial_states.csv')
    result = {}
    for quantity in LABELS:
        sub = df[(np.isclose(df.measurement_bound, 1.0)) &
                 (np.isclose(df.time_s, 600.0)) & (df.quantity == quantity)]
        sub = sub.sort_values('initial_coordinate')
        if len(sub) != N or not np.array_equal(sub.initial_coordinate.to_numpy(), np.arange(N)):
            raise ValueError(f'Incomplete adversarial direction: {quantity}')
        result[quantity] = sub.scaled_perturbation.to_numpy(dtype=float)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', type=Path, default=Path('build/state_finder_m24_f2d_oracle'))
    parser.add_argument('--maxiter', type=int, default=150)
    parser.add_argument('--ftol', type=float, default=1e-11)
    args = parser.parse_args()
    bounds = pd.read_csv(ROOT/'exact_operational_bounds.csv')
    directions = read_directions()
    feasible = pd.read_csv(ROOT/'adversarial_feasible_amplitudes.csv')
    oracle = Oracle(args.oracle)
    rows, coords = [], []
    try:
        for q, quantity in enumerate(LABELS):
            target = bounds[(np.isclose(bounds.measurement_bound, 1)) &
                            (np.isclose(bounds.time_s, 600)) & (bounds.quantity == quantity)]
            if len(target) != 1:
                raise ValueError(f'Missing F2B bound: {quantity}')
            index = int(target.iloc[0].worst_index)
            linear_bound = float(target.iloc[0].exact_bound)
            f = feasible[feasible.quantity == quantity]
            if len(f) != 1 or int(f.iloc[0].worst_index) != index:
                raise ValueError(f'Missing F2C-2 starting point: {quantity}')
            seed = directions[quantity] * float(f.iloc[0].alpha_feasible)
            # Both objective signs are optimized; include the opposite ray.
            for sign in (1., -1.):
                x0 = sign * seed
                # A margin protects SLSQP from starting on a numerically infeasible boundary.
                x0 *= 0.999999
                if oracle.evaluate(x0)[1] > 1.0:
                    x0 *= 0.99
                def fun(x):
                    return -sign * oracle.evaluate(x)[0][q]
                def jac(x):
                    return -sign * oracle.evaluate(x)[4][q]
                constraints = [
                    {'type':'ineq', 'fun':lambda x: 1.0 - float(np.dot(x,x)),
                     'jac':lambda x: -2.0*np.asarray(x)},
                    {'type':'ineq', 'fun':lambda x: 1.0 - oracle.evaluate(x)[1],
                     'jac':lambda x: -oracle.evaluate(x)[4][3]},
                ]
                start_calls = oracle.calls
                res = minimize(fun, x0, jac=jac, method='SLSQP', bounds=[(-1.0, 1.0)] * len(x0), constraints=constraints,
                               options={'maxiter':args.maxiter,'ftol':args.ftol,'disp':False})
                v, r2, pmin, conservation, gradients = oracle.evaluate(res.x)
                initial_norm = float(np.linalg.norm(res.x))
                measurement_norm = float(np.sqrt(max(0.,r2)))
                valid = bool(initial_norm <= 1+1e-7 and r2 <= 1+1e-7 and
                             pmin > 0 and conservation <= 1e-6)
                row = dict(quantity=quantity, worst_index=index, sign=int(sign),
                           success=bool(res.success), status=int(res.status), message=str(res.message),
                           iterations=int(res.nit), oracle_calls=oracle.calls-start_calls,
                           initial_norm=initial_norm, measurement_norm=measurement_norm,
                           min_pressure_Pa=float(pmin), max_conservation_defect_kg=float(conservation),
                           terminal_signed=float(v[q]), terminal_abs=abs(float(v[q])),
                           F2B_linear_bound=linear_bound,
                           nonlinear_to_F2B_ratio=abs(float(v[q]))/linear_bound,
                           feasible=valid)
                rows.append(row)
                coords.extend(dict(quantity=quantity, sign=int(sign), coordinate=j, value=float(value))
                              for j,value in enumerate(res.x))
                print(f'{quantity:9s} sign={sign:+.0f} feasible={valid} '
                      f'success={res.success} iters={res.nit} '
                      f'|r|={measurement_norm:.10g} '
                      f'|target|={abs(v[q]):.12g} ratio={row["nonlinear_to_F2B_ratio"]:.9g}',flush=True)
        ROOT.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(ROOT/'f2d_nonlinear_local_optima.csv',index=False)
        pd.DataFrame(coords).to_csv(ROOT/'f2d_nonlinear_local_optima_directions.csv',index=False)
        print('\nBest feasible solutions by quantity (local, not certified global):')
        df=pd.DataFrame(rows)
        for quantity in LABELS:
            sub=df[(df.quantity==quantity)&(df.feasible)]
            if sub.empty:
                print(f'{quantity}: NO FEASIBLE SOLUTION')
            else:
                best=sub.loc[sub.terminal_abs.idxmax()]
                print(f'{quantity}: {best.terminal_abs:.12g} '
                      f'({best.nonlinear_to_F2B_ratio:.9g} x linear F2B), '
                      f'sign={best.sign:+d}, converged={best.success}')
        print(f'Oracle evaluations: {oracle.calls}')
    finally:
        oracle.close()

if __name__ == '__main__':
    main()
