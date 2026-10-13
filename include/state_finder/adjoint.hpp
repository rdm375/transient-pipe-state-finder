#pragma once

#include "state_finder/objective.hpp"
#include "state_finder/trajectory.hpp"

#include <vector>

namespace state_finder {

struct AdjointResult {
    double objective = 0.0;

    // Ordering:
    // p[1..n], face_flow[0..n-1], inlet_flow.
    std::vector<double> gradient;

    // Largest normwise relative backward error
    // reported by the LAPACK transpose solves.
    double max_linear_backward_error = 0.0;

    // Optional diagnostics: one entry per time step.
    std::vector<double> linear_backward_errors;
};

// Evaluate the objective and its discrete-adjoint gradient.
// The initial inlet pressure is prescribed, not optimized.
AdjointResult evaluate_adjoint(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state,
    const Observations& observations);

} // namespace state_finder
