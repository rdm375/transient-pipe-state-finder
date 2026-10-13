#include "state_finder/old_state_vjp.hpp"

#include "pipe_sim/physics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace state_finder {

void old_state_vjp(
    int n,
    std::span<const double> old_state,
    const pipe_sim::TransientParameters& par,
    std::span<const double> lambda,
    std::span<double> result)
{
    const auto width = static_cast<std::size_t>(2*n + 2);

    if (n < 2 ||
        old_state.size() != width ||
        lambda.size() != width ||
        result.size() != width) {
        throw std::invalid_argument(
            "Invalid old-state VJP dimensions");
    }

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

    if (!(dx > 0.0 && dt > 0.0 &&
          theta >= 0.0 && theta <= 1.0 &&
          area > 0.0 && diameter > 0.0 &&
          temperature > 0.0 && z > 0.0 &&
          rs > 0.0 && viscosity > 0.0 &&
          roughness >= 0.0)) {
        throw std::invalid_argument(
            "Invalid old-state VJP parameters");
    }

    std::fill(result.begin(), result.end(), 0.0);

    const double cold = 1.0 - theta;

    const double crho = pipe_sim::density_derivative_cz(
        temperature, z, rs);

    // Inlet half-cell continuity.
    //
    // R[1] = A*dx/(2*dt)*(rho_new-rho_old)
    //      + theta*(new_face-new_inlet)
    //      + cold*(old_face-old_inlet).
    result[0] +=
        lambda[1] * (-area*dx*crho/(2.0*dt));

    result[1] += lambda[1] * (-cold);
    result[2] += lambda[1] * cold;

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
            friction.derivative_re * dredm;

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

        // Each endpoint contributes half of the face
        // density derivative.
        const double dsdp =
            -sf*crho/(2.0*rhof);

        // Old-time momentum residual.
        const double lm = lambda[row];

        result[left] +=
            lm * cold * (-1.0/dx + dsdp);

        result[right] +=
            lm * cold * (1.0/dx + dsdp);

        result[face] +=
            lm * (-1.0/(area*dt) + cold*dsdm);

        // Continuity to the right of this face.
        const double lc = lambda[row + 1];

        if (i < n-1) {
            result[right] +=
                lc * (-area*dx*crho/dt);

            result[face] += lc * (-cold);
            result[face + 2] += lc * cold;
        } else {
            // Outlet half-cell continuity.
            result[right] +=
                lc * (-area*dx*crho/(2.0*dt));

            result[face] += lc * (-cold);
        }
    }
}

} // namespace state_finder
