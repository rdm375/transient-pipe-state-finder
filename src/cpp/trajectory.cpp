#include "state_finder/trajectory.hpp"

#include "pipe_sim/integrate.hpp"
#include "pipe_sim/transient_step.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace state_finder {

std::span<const double> Trajectory::state(int k) const
{
    if (k < 0 || k > completed_steps)
        throw std::out_of_range("Trajectory step out of range");

    const std::size_t width = state_size();
    const std::size_t offset =
        static_cast<std::size_t>(k) * width;

    return {states.data() + offset, width};
}

Trajectory record_trajectory(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state)
{
    const int n = configuration.n;
    const int steps = configuration.steps;

    if (n < 2 || steps < 0)
        throw std::invalid_argument("Invalid trajectory dimensions");

    if (initial_state.pressure.size() !=
            static_cast<std::size_t>(n + 1) ||
        initial_state.flow.size() !=
            static_cast<std::size_t>(n)) {
        throw std::invalid_argument("Invalid initial-state dimensions");
    }

    const auto& par = configuration.parameters;
    const auto& demand = configuration.demand;
    const auto& opt = configuration.newton;

    // Match the integrator's validation before evaluating linepack.
    if (!std::isfinite(initial_state.inlet_flow) ||
        !std::isfinite(par.dx) || par.dx <= 0.0 ||
        !std::isfinite(par.dt) || par.dt <= 0.0 ||
        !std::isfinite(par.theta) ||
        par.theta < 0.0 || par.theta > 1.0 ||
        !std::isfinite(par.diameter) || par.diameter <= 0.0 ||
        !std::isfinite(par.area) || par.area <= 0.0 ||
        !std::isfinite(par.temperature) || par.temperature <= 0.0 ||
        !std::isfinite(par.z) || par.z <= 0.0 ||
        !std::isfinite(par.gas_constant) || par.gas_constant <= 0.0 ||
        !std::isfinite(par.viscosity) || par.viscosity <= 0.0 ||
        !std::isfinite(par.roughness) || par.roughness < 0.0 ||
        !std::isfinite(opt.residual_tolerance) ||
        opt.residual_tolerance <= 0.0 ||
        !std::isfinite(opt.step_tolerance) ||
        opt.step_tolerance < 0.0 ||
        opt.max_iterations < 0 ||
        !std::isfinite(demand.inlet_pressure) ||
        demand.inlet_pressure <= 0.0 ||
        !std::isfinite(demand.initial_outlet_flow) ||
        !std::isfinite(demand.final_outlet_flow) ||
        !std::isfinite(demand.ramp_duration) ||
        demand.ramp_duration < 0.0) {
        throw std::invalid_argument("Invalid trajectory parameters");
    }

    for (double value : initial_state.pressure) {
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument("Invalid initial pressure");
    }

    for (double value : initial_state.flow) {
        if (!std::isfinite(value))
            throw std::invalid_argument("Invalid initial flow");
    }

    // The initial inlet pressure is prescribed, not an
    // independent optimization variable.
    if (initial_state.pressure[0] != demand.inlet_pressure) {
        throw std::invalid_argument(
            "Initial inlet pressure differs from prescribed pressure");
    }

    Trajectory output;
    output.n = n;
    output.steps = steps;

    const std::size_t count =
        static_cast<std::size_t>(steps) + 1;
    const std::size_t width =
        static_cast<std::size_t>(2*n + 2);

    output.states.resize(count * width);
    output.time.resize(count);
    output.inlet_flow.resize(count);
    output.outlet_flow.resize(count);
    output.outlet_pressure.resize(count);
    output.linepack.resize(count);
    output.newton_iterations.resize(count);

    // Work on private copies: never mutate the candidate.
    std::vector<double> pressure = initial_state.pressure;
    std::vector<double> flow = initial_state.flow;
    double inlet_flow = initial_state.inlet_flow;

    pipe_sim::IntegrationWorkspace workspace(n);

    // Interleaved state at t=0.
    output.states[0] = pressure[0];
    output.states[1] = inlet_flow;

    for (int i = 0; i < n; ++i) {
        output.states[2*i + 2] = flow[i];
        output.states[2*i + 3] = pressure[i + 1];
    }

    double old_outlet_flow = demand.initial_outlet_flow;

    output.time[0] = 0.0;
    output.inlet_flow[0] = inlet_flow;
    output.outlet_flow[0] = old_outlet_flow;
    output.outlet_pressure[0] = pressure[n];
    output.linepack[0] =
        pipe_sim::linepack_cz(n, pressure, par);
    output.newton_iterations[0] = 0;

    for (int k = 1; k <= steps; ++k) {
        const double new_time =
            static_cast<double>(k) * par.dt;

        const double new_outlet_flow = pipe_sim::demand_ramp(
            new_time,
            demand.initial_outlet_flow,
            demand.final_outlet_flow,
            demand.ramp_duration);

        const pipe_sim::TransientBoundary boundary{
            inlet_flow,
            old_outlet_flow,
            new_outlet_flow,
            demand.inlet_pressure
        };

        const auto step = pipe_sim::transient_step(
            n,
            pressure,
            flow,
            boundary,
            par,
            opt,
            workspace.state,
            workspace.newton);

        if (step.info != 0) {
            output.info = step.info;
            output.completed_steps = k - 1;
            return output;
        }

        // Match integrate_transient() extraction order.
        workspace.new_pressure[0] = workspace.state[0];

        for (int i = 0; i < n; ++i) {
            workspace.new_flow[i] = workspace.state[2*i + 2];
            workspace.new_pressure[i + 1] =
                workspace.state[2*i + 3];
        }

        output.time[k] = new_time;
        output.inlet_flow[k] = workspace.state[1];
        output.outlet_flow[k] = new_outlet_flow;
        output.outlet_pressure[k] =
            workspace.new_pressure[n];

        output.linepack[k] = pipe_sim::linepack_cz(
            n, workspace.new_pressure, par);

        output.newton_iterations[k] = step.iterations;

        // Preserve the forward integrator's arithmetic grouping.
        const double balance_error = std::abs(
            (output.linepack[k] - output.linepack[k-1]) / par.dt
            - (par.theta *
                (output.inlet_flow[k] - output.outlet_flow[k])
              + (1.0 - par.theta) *
                (output.inlet_flow[k-1] -
                 output.outlet_flow[k-1])));

        output.max_mass_defect = std::max(
            output.max_mass_defect, balance_error);

        // Retain the complete converged Newton state.
        std::copy(
            workspace.state.begin(),
            workspace.state.end(),
            output.states.begin() +
                static_cast<std::size_t>(k) * width);

        std::copy(
            workspace.new_pressure.begin(),
            workspace.new_pressure.end(),
            pressure.begin());

        std::copy(
            workspace.new_flow.begin(),
            workspace.new_flow.end(),
            flow.begin());

        inlet_flow = output.inlet_flow[k];
        old_outlet_flow = new_outlet_flow;
        output.completed_steps = k;
    }

    return output;
}

} // namespace state_finder
