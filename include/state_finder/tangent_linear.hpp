#pragma once

#include "state_finder/forward_model.hpp"
#include "state_finder/scaled_initial_state.hpp"

#include <vector>

namespace state_finder {

struct TangentLinearResult {
    // Perturbations at t=0,...,steps.
    std::vector<double> inlet_flow;
    std::vector<double> outlet_pressure;

    // Final interleaved state perturbation.
    std::vector<double> final_state;

    double max_linear_backward_error = 0.0;
};

// Propagate one dimensionless initial-state direction.
//
// Direction ordering:
// p[1..n], flow[0..n-1], inlet_flow.
//
// Prescribed boundary histories remain fixed.
TangentLinearResult evaluate_tangent_linear(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state,
    const std::vector<double>& scaled_direction,
    InitialStateScales scales = {});

} // namespace state_finder
