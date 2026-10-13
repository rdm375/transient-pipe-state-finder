#pragma once

#include "state_finder/joint_inverse_objective.hpp"

#include <span>
#include <vector>

namespace state_finder {

// Shared nonlinear trajectory for joint statefinding.
//
// Interleaved state ordering follows the existing
// pipe simulator:
//
// [p0, inlet_flow, face_flow0, p1, ..., face_flow(n-1), pn]
//
// Initial compatibility:
// p0(0) = boundary.inlet_pressure[0].
struct JointTrajectory {
    int n = 0;
    int steps = 0;

    std::vector<double> states;
    BoundaryMeasurements predicted;

    [[nodiscard]]
    std::span<const double> state(int k) const;
};

// Record the complete nonlinear joint trajectory.
//
// Throws if a timestep fails. The result is suitable
// for reuse by tangent-linear and adjoint operators.
JointTrajectory record_joint_trajectory(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables);

} // namespace state_finder
