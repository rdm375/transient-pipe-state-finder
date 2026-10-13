#include "state_finder/objective.hpp"

#include <cmath>
#include <stdexcept>

namespace state_finder {

ObjectiveResult evaluate_objective(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state,
    const Observations& observations)
{
    const auto count =
        static_cast<std::size_t>(configuration.steps + 1);

    if (observations.inlet_flow.size() != count ||
        observations.outlet_pressure.size() != count) {
        throw std::invalid_argument(
            "Observation dimensions do not match simulation");
    }

    if (!std::isfinite(observations.sigma_flow) ||
        !std::isfinite(observations.sigma_pressure) ||
        observations.sigma_flow <= 0.0 ||
        observations.sigma_pressure <= 0.0) {
        throw std::invalid_argument(
            "Observation scales must be finite and positive");
    }

    ObjectiveResult result;
    result.forward = evaluate_forward(
        configuration, initial_state);

    if (result.forward.info != 0 ||
        result.forward.completed_steps != configuration.steps) {
        throw std::runtime_error(
            "Forward simulation failed during objective evaluation");
    }

    const std::size_t first =
        observations.include_initial ? 0 : 1;

    double sum = 0.0;

    for (std::size_t k = first; k < count; ++k) {
        const double flow_residual =
            (result.forward.inlet_flow[k] -
             observations.inlet_flow[k]) /
            observations.sigma_flow;

        const double pressure_residual =
            (result.forward.outlet_pressure[k] -
             observations.outlet_pressure[k]) /
            observations.sigma_pressure;

        if (!std::isfinite(flow_residual) ||
            !std::isfinite(pressure_residual)) {
            throw std::runtime_error(
                "Nonfinite observation residual");
        }

        sum += flow_residual * flow_residual +
               pressure_residual * pressure_residual;
    }

    result.value = 0.5 * sum;

    if (!std::isfinite(result.value))
        throw std::runtime_error("Nonfinite objective value");

    return result;
}

} // namespace state_finder
