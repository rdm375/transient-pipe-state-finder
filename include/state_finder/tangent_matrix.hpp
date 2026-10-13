#pragma once

#include "state_finder/forward_model.hpp"
#include "state_finder/scaled_initial_state.hpp"

#include <vector>

namespace state_finder {

struct TangentMatrixResult {
    int steps = 0;
    int columns = 0;

    // Row-major observation sensitivity matrix:
    //
    // row 2*k:     inlet mass flow
    // row 2*k + 1: outlet pressure
    //
    // Both channels are normalized by their
    // respective observation scales.
    //
    // Shape: 2*(steps+1) x (2*n+1).
    std::vector<double> observations;

    double max_linear_backward_error = 0.0;
};

TangentMatrixResult evaluate_tangent_matrix(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state,
    double sigma_flow = 1.0,
    double sigma_pressure = 1.0e4,
    InitialStateScales scales = {});

} // namespace state_finder
