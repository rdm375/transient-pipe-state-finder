#pragma once

#include "state_finder/forward_model.hpp"
#include "state_finder/observations.hpp"

namespace state_finder {

struct ObjectiveResult {
    double value = 0.0;
    ForwardResult forward;
};

ObjectiveResult evaluate_objective(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state,
    const Observations& observations);

} // namespace state_finder
