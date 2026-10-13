#pragma once

#include <span>

namespace state_finder {

// Explicit boundary-input derivatives of the implicit
// transient residual at one time step.
//
// Input ordering:
//   [inlet_pressure_new, outlet_flow_old, outlet_flow_new]
//
// These operators exclude the dependence on the old
// internal state, which is handled by old_state_jvp/vjp.

void boundary_residual_jvp(
    int n,
    double theta,
    double inlet_pressure_new_direction,
    double outlet_flow_old_direction,
    double outlet_flow_new_direction,
    std::span<double> result);

// Accumulate C^T * lambda into the three boundary
// input gradients, using the same ordering.
void boundary_residual_vjp(
    int n,
    double theta,
    std::span<const double> lambda,
    double& inlet_pressure_new_gradient,
    double& outlet_flow_old_gradient,
    double& outlet_flow_new_gradient);

} // namespace state_finder
