#pragma once

#include "state_finder/objective.hpp"

#include <vector>

namespace state_finder {

// Free initial-state variables, in this order:
// p[1], ..., p[n], flow[0], ..., flow[n-1], inlet_flow.
std::vector<double> pack_initial_state(
    const InitialState& state);

InitialState unpack_initial_state(
    const InitialState& reference,
    const std::vector<double>& variables);

struct DirectionalDifference {
    double step = 0.0;
    double objective_plus = 0.0;
    double objective_minus = 0.0;
    double derivative = 0.0;
};

DirectionalDifference central_directional_difference(
    const ForwardConfiguration& configuration,
    const InitialState& reference,
    const Observations& observations,
    const std::vector<double>& direction,
    double step);

} // namespace state_finder
