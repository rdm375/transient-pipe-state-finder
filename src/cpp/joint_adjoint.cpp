#include "state_finder/joint_adjoint.hpp"

#include "state_finder/boundary_derivatives.hpp"
#include "state_finder/four_channel_objective.hpp"
#include "state_finder/old_state_vjp.hpp"
#include "state_finder/transpose_solver.hpp"

#include "pipe_sim/transient.hpp"
#include "pipe_sim/transient_step.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace state_finder {
namespace {

struct RecordedJointTrajectory {
    std::vector<double> states;
    BoundaryMeasurements predicted;
};

RecordedJointTrajectory record_joint_trajectory(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables)
{
    const int n = configuration.n;
    const int steps = configuration.steps;

    if (n < 2 || steps < 0)
        throw std::invalid_argument(
            "Invalid joint trajectory dimensions");

    const auto count =
        static_cast<std::size_t>(steps + 1);

    const auto width =
        static_cast<std::size_t>(2*n + 2);

    if (variables.initial.pressure.size() !=
            static_cast<std::size_t>(n+1) ||
        variables.initial.flow.size() !=
            static_cast<std::size_t>(n) ||
        variables.boundary.inlet_pressure.size() != count ||
        variables.boundary.outlet_flow.size() != count)
        throw std::invalid_argument(
            "Joint trajectory dimension mismatch");

    RecordedJointTrajectory result;

    result.states.resize(count*width);

    auto& y = result.predicted;

    y.inlet_pressure = variables.boundary.inlet_pressure;
    y.outlet_flow = variables.boundary.outlet_flow;
    y.inlet_flow.resize(count);
    y.outlet_pressure.resize(count);

    auto pressure = variables.initial.pressure;
    auto flow = variables.initial.flow;

    pressure[0] = variables.boundary.inlet_pressure[0];

    double inlet_flow = variables.initial.inlet_flow;

    // Initial interleaved state.
    result.states[0] = pressure[0];
    result.states[1] = inlet_flow;

    for (int i = 0; i < n; ++i) {
        result.states[2*i+2] = flow[i];
        result.states[2*i+3] = pressure[i+1];
    }

    y.inlet_flow[0] = inlet_flow;
    y.outlet_pressure[0] = pressure[n];

    pipe_sim::NewtonWorkspace workspace(n);

    std::vector<double> state(width);
    std::vector<double> next_pressure(n+1);
    std::vector<double> next_flow(n);

    for (int k = 1; k <= steps; ++k) {
        const pipe_sim::TransientBoundary boundary{
            inlet_flow,
            variables.boundary.outlet_flow[k-1],
            variables.boundary.outlet_flow[k],
            variables.boundary.inlet_pressure[k]
        };

        const auto step = pipe_sim::transient_step(
            n,
            pressure,
            flow,
            boundary,
            configuration.parameters,
            configuration.newton,
            state,
            workspace);

        if (step.info != 0)
            throw std::runtime_error(
                "Joint adjoint forward trajectory failed");

        next_pressure[0] = state[0];

        for (int i = 0; i < n; ++i) {
            next_flow[i] = state[2*i+2];
            next_pressure[i+1] = state[2*i+3];
        }

        y.inlet_flow[k] = state[1];
        y.outlet_pressure[k] = state[width-1];

        std::copy(
            state.begin(),
            state.end(),
            result.states.begin() +
                static_cast<std::size_t>(k)*width);

        pressure.swap(next_pressure);
        flow.swap(next_flow);
        inlet_flow = state[1];
    }

    return result;
}

} // namespace

JointAdjointResult evaluate_joint_adjoint(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables,
    const BoundaryObservationRecord& observations,
    JointVariableScales scales)
{
    const int n = configuration.n;
    const int steps = configuration.steps;
    const int width = 2*n+2;

    const auto count =
        static_cast<std::size_t>(steps+1);

    validate_boundary_observations(observations, count);

    const auto trajectory =
        record_joint_trajectory(configuration, variables);

    const FourChannelPrediction prediction{
        trajectory.predicted
    };

    const auto objective =
        evaluate_four_channel_objective(
            observations, prediction);

    const auto direct =
        four_channel_observation_gradient(
            observations, prediction);

    JointAdjointResult result;

    result.measurement_objective = objective.total;

    const std::size_t state_count =
        static_cast<std::size_t>(2*n+1);

    const std::size_t dimension =
        joint_variable_dimension(n, steps);

    result.gradient.assign(dimension, 0.0);

    // Physical-coordinate gradient of the complete
    // initial interleaved state, including p0.
    std::vector<double> initial_gradient(width, 0.0);

    // Physical-coordinate boundary-history gradients.
    std::vector<double> gp(count, 0.0);
    std::vector<double> gq(count, 0.0);

    for (int k = 0; k <= steps; ++k) {
        gp[k] = direct.inlet_pressure[k];
        gq[k] = direct.outlet_flow[k];
    }

    // Adjoint carry = B_(k+1)^T lambda_(k+1).
    std::vector<double> carry(width, 0.0);
    std::vector<double> rhs(width, 0.0);
    std::vector<double> previous_carry(width, 0.0);

    constexpr int kl = 2;
    constexpr int ku = 1;
    constexpr int ldab = 6;

    std::vector<double> jacobian(
        static_cast<std::size_t>(ldab)*width);

    for (int k = steps; k >= 1; --k) {
        std::fill(rhs.begin(), rhs.end(), 0.0);

        // Local observation gradient with respect
        // to the interleaved state u_k.
        rhs[1] = direct.inlet_flow[k];
        rhs[width-1] = direct.outlet_pressure[k];

        for (int j = 0; j < width; ++j)
            rhs[j] -= carry[j];

        const auto offset =
            static_cast<std::size_t>(k)*width;

        const std::span<const double> new_state(
            trajectory.states.data()+offset, width);

        const std::span<const double> old_state(
            trajectory.states.data()+offset-width, width);

        pipe_sim::assemble_jacobian_banded(
            n,
            new_state,
            configuration.parameters,
            jacobian,
            ldab);

        const auto solved = solve_transposed_banded(
            width,
            jacobian,
            ldab,
            kl,
            ku,
            rhs);

        result.max_linear_backward_error = std::max(
            result.max_linear_backward_error,
            solved.relative_backward_error);

        // Constraint contribution to the boundary
        // gradient is -C^T lambda.
        double bp = 0.0;
        double bq_old = 0.0;
        double bq_new = 0.0;

        boundary_residual_vjp(
            n,
            configuration.parameters.theta,
            solved.solution,
            bp,
            bq_old,
            bq_new);

        gp[k] -= bp;
        gq[k-1] -= bq_old;
        gq[k] -= bq_new;

        old_state_vjp(
            n,
            old_state,
            configuration.parameters,
            solved.solution,
            previous_carry);

        carry.swap(previous_carry);
    }

    // At t=0, the local observation gradient
    // contributes directly to the initial state.
    initial_gradient[1] = direct.inlet_flow[0];
    initial_gradient[width-1] =
        direct.outlet_pressure[0];

    for (int j = 0; j < width; ++j)
        initial_gradient[j] -= carry[j];

    // Initial p0 is not independent:
    // p0(0) = inlet_pressure_history[0].
    gp[0] += initial_gradient[0];

    // Convert physical initial-state derivatives
    // into dimensionless optimization coordinates.
    for (int i = 0; i < n; ++i) {
        result.gradient[i] =
            initial_gradient[2*i+3] *
            scales.initial.pressure;

        result.gradient[n+i] =
            initial_gradient[2*i+2] *
            scales.initial.flow;
    }

    result.gradient[2*n] =
        initial_gradient[1] *
        scales.initial.flow;

    // Scale boundary-history derivatives.
    for (std::size_t k = 0; k < count; ++k) {
        result.gradient[state_count+k] =
            gp[k]*scales.inlet_pressure;

        result.gradient[state_count+count+k] =
            gq[k]*scales.outlet_flow;
    }

    for (double value : result.gradient) {
        if (!std::isfinite(value))
            throw std::runtime_error(
                "Nonfinite joint adjoint gradient");
    }

    return result;
}

} // namespace state_finder
