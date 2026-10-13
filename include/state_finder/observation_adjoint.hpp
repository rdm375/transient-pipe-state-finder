#pragma once

#include "state_finder/trajectory.hpp"

#include <span>
#include <vector>

namespace state_finder {

struct ObservationAdjointResult {
    // Physical initial-state gradient:
    // p[1..n], face_flow[0..n-1], inlet_flow.
    std::vector<double> gradient;

    double max_linear_backward_error = 0.0;
};

// Apply the transpose of the physical observation
// sensitivity operator to noise-normalized forcing.
//
// forcing[2*k]   : inlet-flow forcing at time k
// forcing[2*k+1] : outlet-pressure forcing at time k
//
// Only k >= start_step contributes.
//
// Returns H_physical^T * R^(-1/2) * forcing.
// The caller applies B^(1/2) to obtain K^T * forcing.
ObservationAdjointResult apply_observation_adjoint(
    const ForwardConfiguration& configuration,
    const Trajectory& trajectory,
    std::span<const double> forcing,
    int start_step = 0,
    double sigma_flow = 0.01,
    double sigma_pressure = 10.0);

} // namespace state_finder
