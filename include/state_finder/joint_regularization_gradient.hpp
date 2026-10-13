#pragma once

#include "state_finder/joint_scaled_problem.hpp"

#include <span>
#include <vector>

namespace state_finder {

// Analytic gradient of the three quadratic regularization
// terms in dimensionless joint optimization coordinates.
//
// The initial-state block has 2*n+1 variables.
// Each boundary-history block has steps+1 variables.
std::vector<double> joint_regularization_gradient(
    int n,
    int steps,
    std::span<const double> x,
    JointRegularization regularization);

} // namespace state_finder
