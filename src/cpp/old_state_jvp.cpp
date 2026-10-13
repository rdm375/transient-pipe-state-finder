#include "state_finder/old_state_jvp.hpp"

#include "pipe_sim/physics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace state_finder {

void old_state_jvp(
    int n,
    std::span<const double> old_state,
    const pipe_sim::TransientParameters& par,
    std::span<const double> direction,
    std::span<double> result)
{
    if (n < 2)
        throw std::invalid_argument("Invalid cell count");

    const auto width = static_cast<std::size_t>(2*n + 2);

    if (old_state.size() != width ||
        direction.size() != width ||
        result.size() != width)
        throw std::invalid_argument(
            "Invalid old-state JVP dimensions");

    const double dx = par.dx;
    const double dt = par.dt;
    const double theta = par.theta;
    const double area = par.area;
    const double diameter = par.diameter;
    const double temperature = par.temperature;
    const double z = par.z;
    const double rs = par.gas_constant;
    const double viscosity = par.viscosity;
    const double roughness = par.roughness;

    if (!(std::isfinite(dx) && dx > 0.0 &&
          std::isfinite(dt) && dt > 0.0 &&
          std::isfinite(theta) &&
          theta >= 0.0 && theta <= 1.0 &&
          std::isfinite(area) && area > 0.0 &&
          std::isfinite(diameter) && diameter > 0.0 &&
          std::isfinite(temperature) && temperature > 0.0 &&
          std::isfinite(z) && z > 0.0 &&
          std::isfinite(rs) && rs > 0.0 &&
          std::isfinite(viscosity) && viscosity > 0.0 &&
          std::isfinite(roughness) && roughness >= 0.0))
        throw std::invalid_argument(
            "Invalid old-state JVP parameters");

    std::fill(result.begin(), result.end(), 0.0);

    const double cold = 1.0 - theta;

    const double crho = pipe_sim::density_derivative_cz(
        temperature, z, rs);

    // Inlet half-cell continuity.
    result[1] =
        (-area*dx*crho/(2.0*dt))*direction[0]
        - cold*direction[1]
        + cold*direction[2];

    for (int i = 0; i < n; ++i) {
        const int left = (i == 0) ? 0 : 2*i + 1;
        const int face = 2*i + 2;
        const int right = 2*i + 3;
        const int row = 2*i + 2;

        const double pl = old_state[left];
        const double pr = old_state[right];
        const double mdot = old_state[face];

        const double rhol = pipe_sim::density_cz(
            pl, temperature, z, rs);

        const double rhor = pipe_sim::density_cz(
            pr, temperature, z, rs);

        const double rhof = 0.5*(rhol + rhor);

        const double reynolds = pipe_sim::reynolds_mass(
            mdot, diameter, area, viscosity);

        const auto friction =
            pipe_sim::friction_sj_value_derivative(
                reynolds, roughness, diameter);

        const double dredm =
            pipe_sim::reynolds_derivative_mdot(
                mdot, diameter, area, viscosity);

        const double dfdm =
            friction.derivative_re*dredm;

        const double sf = pipe_sim::friction_source(
            friction.value, diameter, area, rhof, mdot);

        const double dsdm =
            pipe_sim::friction_source_derivative_mdot(
                friction.value,
                dfdm,
                diameter,
                area,
                rhof,
                mdot);

        const double dsdp =
            -sf*crho/(2.0*rhof);

        // Momentum equation: B*v.
        result[row] =
            cold*(-1.0/dx + dsdp)*direction[left]
            + cold*(1.0/dx + dsdp)*direction[right]
            + (-1.0/(area*dt) + cold*dsdm)
                *direction[face];

        // Continuity equation to the right.
        if (i < n-1) {
            result[row+1] =
                (-area*dx*crho/dt)*direction[right]
                - cold*direction[face]
                + cold*direction[face+2];
        }
        else {
            result[row+1] =
                (-area*dx*crho/(2.0*dt))
                    *direction[right]
                - cold*direction[face];
        }
    }

    // Row zero is the prescribed inlet-pressure
    // boundary condition and has no old-state
    // dependence.
    result[0] = 0.0;
}

} // namespace state_finder
