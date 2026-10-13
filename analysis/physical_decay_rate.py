#!/usr/bin/env python3
"""M1.7d: Independent physical prediction of slow decay rate."""

import numpy as np
from scipy.linalg import eigvals
from scipy.sparse import diags
from scipy.sparse.linalg import eigs

# ---------------------------------------------------------
# Reference parameters from analytic_initial_state_observability.cpp
# ---------------------------------------------------------

n = 100
L = 100000.0
dx = L / n

dt = 15.0
theta = 0.65

diameter = 1.0
area = np.pi * diameter**2 / 4.0

temperature = 288.15
z = 0.9
gas_constant = 500.0

viscosity = 1.1e-5
roughness = 4.5e-5

inlet_pressure = 8.0e6
reference_flow = 100.0

c2 = z * gas_constant * temperature

observed_lambda = -1.472361268023e-3
observed_tau = -1.0 / observed_lambda

# ---------------------------------------------------------
# Swamee-Jain friction and derivative
# ---------------------------------------------------------

def friction_and_derivative(mdot):

    reynolds = (
        mdot * diameter
        / (area * viscosity)
    )

    relative_roughness = roughness / (3.7 * diameter)

    term = 5.74 * reynolds**(-0.9)

    argument = relative_roughness + term

    log_argument = np.log10(argument)

    friction = 0.25 / log_argument**2

    # d(argument)/d(mdot)
    dargument_dm = -0.9 * term / mdot

    dlog_dm = (
        dargument_dm
        / (argument * np.log(10.0))
    )

    df_dm = (
        -0.5 * dlog_dm
        / log_argument**3
    )

    return friction, df_dm


f, df_dm = friction_and_derivative(reference_flow)

reynolds = (
    reference_flow * diameter
    / (area * viscosity)
)

# ---------------------------------------------------------
# Reconstruct exact reference pressure profile formula
# ---------------------------------------------------------

coefficient = (
    f * c2 * reference_flow**2
    / (diameter * area**2)
)

x = np.linspace(0.0, L, n + 1)

pressure = np.sqrt(
    inlet_pressure**2 - coefficient * x
)

if not np.all(np.isfinite(pressure)):
    raise RuntimeError("Invalid reference pressure")

# Face pressures and densities.
p_face = 0.5 * (
    pressure[:-1] + pressure[1:]
)

rho_face = p_face / c2

# ---------------------------------------------------------
# Linearized friction coefficients
# ---------------------------------------------------------

a_face = (
    reference_flow
    / (diameter * area**2 * rho_face)
    * (f + 0.5 * reference_flow * df_dm)
)

source_face = (
    f * reference_flow**2
    / (2.0 * diameter * area**2 * rho_face)
)

b_face = -source_face / p_face

diffusivity_face = c2 / (area * a_face)

# ---------------------------------------------------------
# Constant-coefficient prediction
# ---------------------------------------------------------

a_mean = np.mean(a_face)

D_constant = c2 / (area * a_mean)

k1 = np.pi / (2.0 * L)

lambda_constant = -D_constant * k1**2

# ---------------------------------------------------------
# Conservative finite-volume diffusion operator
#
# Unknown pressure perturbations at nodes 1..n.
#
# Inlet:
#   delta p(0) = 0
#
# Outlet:
#   delta mdot(L) = 0
#
# Face relation:
#   delta mdot =
#      -(dp/dx + b*average(p))/a
#
# Node volumes:
#   dx at interior nodes
#   dx/2 at outlet
# ---------------------------------------------------------

def build_operator(include_pressure_friction):

    matrix = np.zeros((n, n))

    # Pressure perturbation at inlet is fixed to zero.
    # Unknown vector represents nodes 1..n.

    for face in range(n):

        a = a_face[face]

        b = (
            b_face[face]
            if include_pressure_friction
            else 0.0
        )

        # delta mdot(face) =
        # alpha*delta p(left)
        # + beta*delta p(right)

        alpha = 1.0 / (a * dx) - b / (2.0 * a)
        beta = -1.0 / (a * dx) - b / (2.0 * a)

        # Left node contribution.
        if face >= 1:

            left = face - 1

            volume_left = (
                dx / 2.0
                if left == n - 1
                else dx
            )

            # dp_left/dt = -c2/A *
            # (q_right-q_left)/volume
            factor = -c2 / (area * volume_left)

            matrix[left, left] += factor * alpha
            matrix[left, face] += factor * beta

        # Right node contribution.
        right = face

        volume_right = (
            dx / 2.0
            if right == n - 1
            else dx
        )

        # dp_right/dt = -c2/A *
        # (q_right-q_left)/volume
        factor = c2 / (area * volume_right)

        if face >= 1:
            matrix[right, face-1] += factor * alpha

        matrix[right, right] += factor * beta

    return matrix


def slowest_eigenvalue(matrix):

    values = eigvals(matrix)

    # Stable decay modes have negative real parts.
    stable = values[
        np.real(values) < 0.0
    ]

    if stable.size == 0:
        raise RuntimeError("No stable eigenvalues")

    index = np.argmax(np.real(stable))

    return stable[index]


operator_no_b = build_operator(False)
operator_with_b = build_operator(True)

lambda_variable = slowest_eigenvalue(operator_no_b)
lambda_with_b = slowest_eigenvalue(operator_with_b)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("M1.7d: Physical decay-rate prediction")
print()

print("Reference operating state")
print(f"Length:                {L:.6f} m")
print(f"Diameter:              {diameter:.6f} m")
print(f"Area:                  {area:.12f} m^2")
print(f"Sound speed:           {np.sqrt(c2):.9f} m/s")
print(f"Reynolds number:       {reynolds:.9e}")
print(f"Friction factor:       {f:.12e}")
print(f"df/dmdot:              {df_dm:.12e}")
print(f"Inlet pressure:        {pressure[0]:.9f} Pa")
print(f"Outlet pressure:       {pressure[-1]:.9f} Pa")

print()
print("Linearized coefficients")
print(
    "a_f range:",
    f"{a_face.min():.12e}",
    f"{a_face.max():.12e}"
)
print(
    "b_f range:",
    f"{b_face.min():.12e}",
    f"{b_face.max():.12e}"
)
print(
    "Diffusivity range:",
    f"{diffusivity_face.min():.9e}",
    f"{diffusivity_face.max():.9e}"
)

print()
print("Decay-rate comparison")
print()

models = [
    ("Observed full transient", observed_lambda),
    ("Constant diffusion", lambda_constant),
    ("Variable diffusion", lambda_variable),
    ("Variable diffusion + b_f", lambda_with_b),
]

for name, value in models:

    value = complex(value)

    lam = value.real

    tau = -1.0 / lam

    relative_error = (
        abs(lam - observed_lambda)
        / abs(observed_lambda)
    )

    print(name)
    print(f"  lambda = {lam:.12e} 1/s")
    print(f"  tau    = {tau:.9f} s")
    print(f"  error  = {100*relative_error:.6f}%")

    if abs(value.imag) > 1e-10:
        print(f"  imaginary = {value.imag:.12e}")

    print()

# ---------------------------------------------------------
# Discrete diffusion eigenmode
# ---------------------------------------------------------

values, vectors = np.linalg.eig(operator_with_b)

stable_indices = np.where(
    np.real(values) < 0.0
)[0]

index = stable_indices[
    np.argmax(np.real(values[stable_indices]))
]

mode = np.real(vectors[:, index])

if mode[-1] < 0:
    mode = -mode

mode /= np.max(np.abs(mode))

sine = np.sin(
    np.pi * x[1:] / (2.0 * L)
)

sine /= np.max(np.abs(sine))

relative_shape_error = (
    np.linalg.norm(mode - sine)
    / np.linalg.norm(sine)
)

print("Spatial mode comparison")
print(
    "Diffusion mode vs fundamental sine:",
    f"{relative_shape_error:.12e}"
)

print()
print("PASS M1.7d physical decay-rate analysis")
