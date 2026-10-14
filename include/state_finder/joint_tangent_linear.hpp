#pragma once

#include "state_finder/joint_trajectory.hpp"
#include "state_finder/joint_scaled_problem.hpp"

#include <span>
#include <stdexcept>
#include <vector>

namespace state_finder {

struct JointTangentResult {
    // Physical four-channel perturbations at t=0,...,steps.
    BoundaryMeasurements response;

    // Time-major physical tangent states.
    // Layout matches JointTrajectory::states:
    // [p0, inlet_flow, face_flow0, p1, ..., face_flow(n-1), pn]
    // at each timestep, including t=0.
    std::vector<double> states;

    // Retained for backward compatibility.
    std::vector<double> final_state;

    double max_linear_backward_error = 0.0;

    [[nodiscard]]
    std::span<const double> state(int k, int n) const
    {
        if (n < 2 || k < 0)
            throw std::out_of_range(
                "Invalid tangent state index");

        const auto width =
            static_cast<std::size_t>(2*n+2);

        const auto offset =
            static_cast<std::size_t>(k)*width;

        if (offset > states.size() ||
            width > states.size()-offset)
            throw std::out_of_range(
                "Tangent state index out of range");

        return std::span<const double>(
            states.data()+offset, width);
    }
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
