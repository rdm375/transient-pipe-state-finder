#pragma once

#include "state_finder/joint_scaled_problem.hpp"

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
