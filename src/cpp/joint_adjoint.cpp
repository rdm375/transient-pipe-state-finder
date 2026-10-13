#include "state_finder/joint_adjoint.hpp"
#include "state_finder/joint_trajectory.hpp"

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


JointAdjointResult apply_joint_observation_adjoint_physical(
    const ForwardConfiguration& configuration,
    const JointTrajectory& trajectory,
    const BoundaryMeasurements& direct,
    JointVariableScales scales)
{
    const int n = configuration.n;
    const int steps = configuration.steps;
    const int width = 2*n+2;

    const auto count =
        static_cast<std::size_t>(steps+1);

    if (trajectory.n != n ||
        trajectory.steps != steps)
        throw std::invalid_argument(
            "Joint adjoint trajectory mismatch");

    const auto valid = [count](
        const std::vector<double>& values)
    {
        if (values.size() != count)
            throw std::invalid_argument(
                "Joint adjoint forcing dimension mismatch");

        for (double value : values)
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "Nonfinite joint adjoint forcing");
    };

    valid(direct.inlet_pressure);
    valid(direct.inlet_flow);
    valid(direct.outlet_pressure);
    valid(direct.outlet_flow);

    JointAdjointResult result;


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


JointAdjointResult apply_joint_observation_adjoint(
    const ForwardConfiguration& configuration,
    const JointTrajectory& trajectory,
    std::span<const double> whitened_forcing,
    const BoundaryObservationRecord& observations,
    JointVariableScales scales)
{
    const int steps = configuration.steps;

    if (steps < 0)
        throw std::invalid_argument(
            "Invalid joint adjoint step count");

    const auto count =
        static_cast<std::size_t>(steps + 1);

    validate_boundary_observations(observations, count);

    const auto first =
        observations.include_initial
        ? std::size_t{0}
        : std::size_t{1};

    if (whitened_forcing.size() != 4*(count-first))
        throw std::invalid_argument(
            "Joint adjoint normalized forcing dimension mismatch");

    const auto& s = observations.scales;

    BoundaryMeasurements direct;

    direct.inlet_pressure.assign(count, 0.0);
    direct.inlet_flow.assign(count, 0.0);
    direct.outlet_pressure.assign(count, 0.0);
    direct.outlet_flow.assign(count, 0.0);

    std::size_t row = 0;

    for (std::size_t k = first; k < count; ++k) {
        direct.inlet_pressure[k] =
            whitened_forcing[row++] / s.inlet_pressure;

        direct.inlet_flow[k] =
            whitened_forcing[row++] / s.inlet_flow;

        direct.outlet_pressure[k] =
            whitened_forcing[row++] / s.outlet_pressure;

        direct.outlet_flow[k] =
            whitened_forcing[row++] / s.outlet_flow;
    }

    return apply_joint_observation_adjoint_physical(
        configuration, trajectory, direct, scales);
}

JointAdjointResult evaluate_joint_adjoint(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables,
    const BoundaryObservationRecord& observations,
    JointVariableScales scales)
{
    const auto trajectory =
        record_joint_trajectory(configuration, variables);

    const FourChannelPrediction prediction{
        trajectory.predicted
    };

    const auto objective =
        evaluate_four_channel_objective(
            observations, prediction);

    const auto& residuals = objective.residuals;

    auto result = apply_joint_observation_adjoint(
        configuration,
        trajectory,
        residuals,
        observations,
        scales);

    result.measurement_objective = objective.total;

    return result;
}

} // namespace state_finder
