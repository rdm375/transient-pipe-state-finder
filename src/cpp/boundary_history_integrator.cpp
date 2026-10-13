#include "state_finder/boundary_history_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace state_finder {

BoundaryHistoryIntegrationResult integrate_boundary_history(
    int n,
    std::span<double> pressure,
    std::span<double> flow,
    double& inlet_flow,
    const AdjustableBoundaryHistory& boundary,
    int steps,
    const pipe_sim::TransientParameters& par,
    const pipe_sim::NewtonOptions& opt)
{
    if (n < 1 || steps < 0)
        throw std::invalid_argument(
            "Invalid boundary-history integration dimensions");

    if (pressure.size() != static_cast<std::size_t>(n + 1) ||
        flow.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument(
            "Invalid initial-state dimensions");

    const auto count = static_cast<std::size_t>(steps + 1);

    if (boundary.inlet_pressure.size() != count ||
        boundary.outlet_flow.size() != count)
        throw std::invalid_argument(
            "Boundary-history dimensions mismatch");

    for (double value : boundary.inlet_pressure) {
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument(
                "Invalid inlet-pressure history");
    }

    for (double value : boundary.outlet_flow) {
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Invalid outlet-flow history");
    }

    if (!std::isfinite(pressure[0]) ||
        pressure[0] != boundary.inlet_pressure[0])
        throw std::invalid_argument(
            "Initial inlet pressure must equal boundary history at t=0");

    for (double value : pressure) {
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument(
                "Invalid initial pressure");
    }

    for (double value : flow) {
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Invalid initial flow");
    }

    if (!std::isfinite(inlet_flow))
        throw std::invalid_argument(
            "Invalid initial inlet flow");

    BoundaryHistoryIntegrationResult result;

    auto& y = result.predicted;

    y.inlet_pressure = boundary.inlet_pressure;
    y.outlet_flow = boundary.outlet_flow;

    y.inlet_flow.assign(count, 0.0);
    y.outlet_pressure.assign(count, 0.0);

    result.linepack.assign(count, 0.0);
    result.newton_iterations.assign(count, 0);

    y.inlet_flow[0] = inlet_flow;
    y.outlet_pressure[0] = pressure[n];

    result.linepack[0] =
        pipe_sim::linepack_cz(n, pressure, par);

    pipe_sim::NewtonWorkspace workspace(n);

    std::vector<double> new_state(
        static_cast<std::size_t>(2*n + 2));

    std::vector<double> new_pressure(count > 0 ? n + 1 : 0);
    std::vector<double> new_flow(n);

    double max_mass_defect = 0.0;

    for (int k = 1; k <= steps; ++k) {
        const pipe_sim::TransientBoundary bc{
            inlet_flow,
            boundary.outlet_flow[k-1],
            boundary.outlet_flow[k],
            boundary.inlet_pressure[k]
        };

        const auto step = pipe_sim::transient_step(
            n,
            pressure,
            flow,
            bc,
            par,
            opt,
            new_state,
            workspace
        );

        if (step.info != 0) {
            result.integration = {
                step.info,
                k-1,
                max_mass_defect
            };

            return result;
        }

        new_pressure[0] = new_state[0];

        for (int i = 0; i < n; ++i) {
            new_flow[i] = new_state[2*i+2];
            new_pressure[i+1] = new_state[2*i+3];
        }

        y.inlet_flow[k] = new_state[1];
        y.outlet_pressure[k] = new_pressure[n];

        result.linepack[k] =
            pipe_sim::linepack_cz(n, new_pressure, par);

        result.newton_iterations[k] = step.iterations;

        const double balance_error = std::abs(
            (result.linepack[k] - result.linepack[k-1]) / par.dt
            - (
                par.theta *
                (y.inlet_flow[k] - y.outlet_flow[k])
                +
                (1.0 - par.theta) *
                (y.inlet_flow[k-1] - y.outlet_flow[k-1])
            )
        );

        max_mass_defect =
            std::max(max_mass_defect, balance_error);

        std::copy(
            new_pressure.begin(),
            new_pressure.end(),
            pressure.begin()
        );

        std::copy(
            new_flow.begin(),
            new_flow.end(),
            flow.begin()
        );

        inlet_flow = y.inlet_flow[k];
    }

    result.integration = {
        0,
        steps,
        max_mass_defect
    };

    return result;
}

} // namespace state_finder
