#pragma once

#include "state_finder/joint_scaled_problem.hpp"
#include "state_finder/joint_trajectory.hpp"

#include <span>

#include <vector>

namespace state_finder {

struct JointAdjointResult {
    double measurement_objective = 0.0;

    // Gradient of the four-channel measurement objective
    // in the dimensionless joint optimization coordinates.
    //
    // Ordering:
    //   initial pressures p[1..n]
    //   initial internal flows m[0..n-1]
    //   initial inlet flow
    //   inlet-pressure history [0..steps]
    //   outlet-flow history [0..steps]
    std::vector<double> gradient;

    double max_linear_backward_error = 0.0;
};


// Apply the transpose of the normalized four-channel
// observation Jacobian to arbitrary observation forcing.
//
// Input ordering matches normalized_joint_tangent_residuals():
// [pin, min, pout, mout] at each included time sample.
//
// The result is in dimensionless joint optimization
// coordinates, matching pack_joint_variables().
//
// The nonlinear trajectory is reused; no forward
// integration is performed by this overload.
JointAdjointResult apply_joint_observation_adjoint(
    const ForwardConfiguration& configuration,
    const JointTrajectory& trajectory,
    std::span<const double> whitened_forcing,
    const BoundaryObservationRecord& observations,
    JointVariableScales scales = {});

// Exact discrete adjoint of the implemented implicit
// transient equations, up to forward/Newton solve tolerance.
//
// This returns the measurement gradient only.
// Regularization is handled separately.
JointAdjointResult evaluate_joint_adjoint(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables,
    const BoundaryObservationRecord& observations,
    JointVariableScales scales = {});

} // namespace state_finder
