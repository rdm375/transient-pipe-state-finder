#pragma once

#include "state_finder/boundary_history_integrator.hpp"
#include "state_finder/forward_model.hpp"
#include "state_finder/four_channel_objective.hpp"

namespace state_finder {

// Optimization variables for the four-measurement problem.
//
// initial.pressure[0] is not an independent optimization
// variable: it is replaced by boundary.inlet_pressure[0]
// before the forward integration.
struct JointInverseVariables {
    InitialState initial;
    AdjustableBoundaryHistory boundary;
};

struct JointInverseObjectiveResult {
    FourChannelObjectiveResult measurement;
    BoundaryHistoryIntegrationResult trajectory;
};

// Full nonlinear measurement objective.
//
// Regularization is intentionally separate, so the
// existing quadratic regularizers can be reused.
JointInverseObjectiveResult evaluate_joint_inverse_objective(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables,
    const BoundaryObservationRecord& observations);

} // namespace state_finder
