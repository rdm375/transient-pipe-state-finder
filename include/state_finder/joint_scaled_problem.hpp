#pragma once

#include "state_finder/joint_inverse_objective.hpp"
#include "state_finder/optimizer.hpp"
#include "state_finder/scaled_initial_state.hpp"

#include <cstddef>
#include <vector>

namespace state_finder {

struct JointVariableScales {
    InitialStateScales initial{};
    double inlet_pressure = 1.0e4;
    double outlet_flow = 1.0;
};

struct JointRegularization {
    // Penalties act on dimensionless deviations from baseline.
    double initial_weight = 1.0e-4;
    double boundary_weight = 1.0e-2;
    double boundary_smoothness = 1.0e-2;
};

struct JointObjectiveBreakdown {
    double measurement = 0.0;
    double initial_regularization = 0.0;
    double boundary_regularization = 0.0;
    double boundary_smoothness = 0.0;
    double total = 0.0;
};

std::size_t joint_variable_dimension(
    int n, int steps);

std::vector<double> pack_joint_variables(
    const JointInverseVariables& variables,
    const JointInverseVariables& baseline,
    JointVariableScales scales = {});

JointInverseVariables unpack_joint_variables(
    const std::vector<double>& x,
    const JointInverseVariables& baseline,
    JointVariableScales scales = {});

class JointScaledProblem {
public:
    JointScaledProblem(
        ForwardConfiguration configuration,
        BoundaryObservationRecord observations,
        JointInverseVariables baseline,
        JointVariableScales scales = {},
        JointRegularization regularization = {});

    JointObjectiveBreakdown evaluate(
        const std::vector<double>& x) const;

    OptimizationEvaluation evaluate_for_optimizer(
        const std::vector<double>& x,
        bool need_gradient) const;

    std::vector<double> finite_difference_gradient(
        const std::vector<double>& x,
        double relative_step = 1.0e-4) const;

private:
    ForwardConfiguration configuration_;
    BoundaryObservationRecord observations_;
    JointInverseVariables baseline_;
    JointVariableScales scales_;
    JointRegularization regularization_;
};

} // namespace state_finder
