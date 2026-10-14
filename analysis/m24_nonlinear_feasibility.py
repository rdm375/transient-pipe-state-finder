#!/usr/bin/env python3
"""M2.4-F2C-2: compare ray-feasible nonlinear deviations with F2B."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('benchmarks/results-m24')
feasible = pd.read_csv(root / 'adversarial_feasible_amplitudes.csv')
bounds = pd.read_csv(root / 'exact_operational_bounds.csv')
rows = []
for row in feasible.itertuples(index=False):
    matches = bounds[
        (bounds.quantity == row.quantity)
        & np.isclose(bounds.measurement_bound, row.measurement_bound)
        & np.isclose(bounds.time_s, 600.0)
        & (bounds.worst_index == row.worst_index)
    ]
    if len(matches) != 1:
        raise RuntimeError(f'No unique matching F2B bound for {row.quantity}')
    bound = float(matches.iloc[0].exact_bound)
    if bound <= 0 or not np.isfinite(bound):
        raise RuntimeError('Invalid F2B bound')
    rows.append({
        'quantity': row.quantity,
        'worst_index': row.worst_index,
        'alpha_feasible': row.alpha_feasible,
        'measurement_norm': row.measurement_norm,
        'nonlinear_terminal_signed': row.terminal_targeted_signed,
        'nonlinear_terminal_abs': row.terminal_targeted_abs,
        'linear_F2B_bound': bound,
        'nonlinear_to_linear_ratio': row.terminal_targeted_abs / bound,
        'minimum_pressure_Pa': row.min_pressure_Pa,
        'max_conservation_defect_kg': row.max_conservation_defect_kg,
    })
summary = pd.DataFrame(rows).sort_values('quantity')
if len(summary) != 3 or set(summary.quantity) != {'pressure', 'flow', 'inventory'}:
    raise RuntimeError('Expected pressure, flow, and inventory results')
if not np.isfinite(summary.select_dtypes(include='number').to_numpy()).all():
    raise RuntimeError('Nonfinite feasibility summary')
if (summary.measurement_norm > 1.0 + 1e-8).any():
    raise RuntimeError('Nonlinear measurement infeasibility')
if (summary.max_conservation_defect_kg > 1e-6).any():
    raise RuntimeError('Mass conservation failure')
output = root / 'adversarial_feasibility_summary.csv'
summary.to_csv(output, index=False)
print('\nM2.4-F2C-2 NONLINEAR RAY FEASIBILITY\n')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.10g}'))
print(f'\nSaved: {output}')
print('Interpretation: feasibility and comparisons hold along the three tested rays only.')
