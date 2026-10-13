#include "state_finder/joint_inverse_objective.hpp"

#include <stdexcept>

namespace state_finder {

JointInverseObjectiveResult evaluate_joint_inverse_objective(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables,
    const BoundaryObservationRecord& observations)
{
    const int n = configuration.n;
    const int steps = configuration.steps;

    if (n < 2 || steps < 0)
        throw std::invalid_argument(
            "Invalid joint inverse dimensions");

    if (variables.initial.pressure.size() !=
            static_cast<std::size_t>(n+1) ||
        variables.initial.flow.size() !=
            static_cast<std::size_t>(n))
        throw std::invalid_argument(
            "Invalid joint initial-state dimensions");

    const auto count = static_cast<std::size_t>(steps+1);

    validate_boundary_observations(observations, count);

    if (variables.boundary.inlet_pressure.size() != count ||
        variables.boundary.outlet_flow.size() != count)
        throw std::invalid_argument(
            "Invalid joint boundary-history dimensions");

    // Enforce initial/boundary compatibility by construction.
    auto pressure = variables.initial.pressure;
    auto flow = variables.initial.flow;
    double inlet_flow = variables.initial.inlet_flow;

    pressure[0] = variables.boundary.inlet_pressure[0];

    JointInverseObjectiveResult result;

    result.trajectory = integrate_boundary_history(
        n,
        pressure,
        flow,
        inlet_flow,
        variables.boundary,
        steps,
        configuration.parameters,
        configuration.newton
    );

    if (result.trajectory.integration.info != 0 ||
        result.trajectory.integration.completed_steps != steps)
        throw std::runtime_error(
            "Joint inverse forward integration failed");

    FourChannelPrediction prediction;
    prediction.boundary = result.trajectory.predicted;

    result.measurement = evaluate_four_channel_objective(
        observations,
        prediction
    );

    return result;
}

} // namespace state_finder
