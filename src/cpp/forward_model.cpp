#include "state_finder/forward_model.hpp"

#include <stdexcept>
#include <vector>

namespace state_finder {

ForwardResult evaluate_forward(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state)
{
    const int n = configuration.n;
    const int steps = configuration.steps;

    if (n < 2 || steps < 0)
        throw std::invalid_argument("Invalid forward dimensions");

    if (initial_state.pressure.size() !=
            static_cast<std::size_t>(n + 1) ||
        initial_state.flow.size() !=
            static_cast<std::size_t>(n)) {
        throw std::invalid_argument("Invalid initial-state dimensions");
    }

    // The underlying integrator modifies these arrays.
    // Always use private copies.
    std::vector<double> pressure = initial_state.pressure;
    std::vector<double> flow = initial_state.flow;
    double inlet_flow = initial_state.inlet_flow;

    ForwardResult output;

    const auto count = static_cast<std::size_t>(steps + 1);

    output.time.resize(count);
    output.inlet_flow.resize(count);
    output.outlet_flow.resize(count);
    output.outlet_pressure.resize(count);
    output.linepack.resize(count);

    std::vector<int> iterations(count);

    const pipe_sim::IntegrationHistory history{
        output.time,
        output.inlet_flow,
        output.outlet_flow,
        output.outlet_pressure,
        output.linepack,
        iterations
    };

    pipe_sim::IntegrationWorkspace workspace(n);

    const auto result = pipe_sim::integrate_transient(
        n,
        pressure,
        flow,
        inlet_flow,
        configuration.demand,
        steps,
        configuration.parameters,
        configuration.newton,
        history,
        workspace
    );

    output.info = result.info;
    output.completed_steps = result.completed_steps;
    output.max_mass_defect = result.max_mass_defect;

    if (result.info != 0 || result.completed_steps != steps)
        return output;

    output.final_pressure = std::move(pressure);
    output.final_flow = std::move(flow);
    output.final_inlet_flow = inlet_flow;

    return output;
}

} // namespace state_finder
