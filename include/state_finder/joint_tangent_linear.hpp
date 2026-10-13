#pragma once

#include "state_finder/joint_trajectory.hpp"
#include "state_finder/joint_scaled_problem.hpp"

#include <span>
#include <vector>

namespace state_finder {

struct JointTangentResult {
    // Physical four-channel perturbations at t=0,...,steps.
    BoundaryMeasurements response;
    std::vector<double> final_state;
    double max_linear_backward_error = 0.0;
};

// Apply the joint observation Jacobian to one dimensionless
// optimization-space direction, reusing a nonlinear trajectory.
// Direction ordering matches pack_joint_variables().
JointTangentResult evaluate_joint_tangent_linear(
    const ForwardConfiguration& configuration,
    const JointTrajectory& trajectory,
    std::span<const double> scaled_direction,
    JointVariableScales scales = {});

// Convenience overload; records a nonlinear trajectory once.
JointTangentResult evaluate_joint_tangent_linear(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables,
    std::span<const double> scaled_direction,
    JointVariableScales scales = {});

// Whiten and flatten the four-channel tangent response in the
// same [pin, min, pout, mout] order and initial-sample policy
// as FourChannelObjectiveResult::residuals.
std::vector<double> normalized_joint_tangent_residuals(
    const JointTangentResult& tangent,
    const BoundaryObservationRecord& observations);

} // namespace state_finder
