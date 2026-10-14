#pragma once

#include "state_finder/joint_trajectory.hpp"

#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

namespace state_finder {

// Convert a recorded interleaved state into the initial-state
// representation used by the existing joint trajectory solver.
inline InitialState initial_state_from_joint_state(
    std::span<const double> state,
    int n)
{
    if (n < 2 ||
        state.size() != static_cast<std::size_t>(2*n+2))
        throw std::invalid_argument(
            "Invalid continuation state dimension");

    InitialState result;

    result.pressure.resize(n+1);
    result.flow.resize(n);

    result.pressure[0] = state[0];
    result.inlet_flow = state[1];

    for (int i = 0; i < n; ++i) {
        result.flow[i] = state[2*i+2];
        result.pressure[i+1] = state[2*i+3];
    }

    for (double pressure : result.pressure) {
        if (!std::isfinite(pressure) || pressure <= 0.0)
            throw std::invalid_argument(
                "Invalid continuation pressure");
    }

    if (!std::isfinite(result.inlet_flow))
        throw std::invalid_argument(
            "Invalid continuation inlet flow");

    for (double flow : result.flow) {
        if (!std::isfinite(flow))
            throw std::invalid_argument(
                "Invalid continuation flow");
    }

    return result;
}

// Continue an existing trajectory under prescribed future
// inlet-pressure and outlet-flow histories.
//
// The first history entries must equal the boundary values
// at the end of the previous trajectory.
inline JointTrajectory continue_joint_trajectory(
    const ForwardConfiguration& configuration,
    std::span<const double> terminal_state,
    const AdjustableBoundaryHistory& future_boundary)
{
    if (future_boundary.inlet_pressure.size() !=
            static_cast<std::size_t>(configuration.steps+1) ||
        future_boundary.outlet_flow.size() !=
            static_cast<std::size_t>(configuration.steps+1))
        throw std::invalid_argument(
            "Invalid forecast boundary history dimension");

    const auto initial =
        initial_state_from_joint_state(
            terminal_state, configuration.n);

    if (future_boundary.inlet_pressure[0] !=
            initial.pressure[0])
        throw std::invalid_argument(
            "Forecast inlet pressure discontinuity");

    JointInverseVariables variables;
    variables.initial = initial;
    variables.boundary = future_boundary;

    return record_joint_trajectory(configuration, variables);
}

} // namespace state_finder
