#include "state_finder/tangent_linear.hpp"

#include "state_finder/old_state_jvp.hpp"
#include "state_finder/trajectory.hpp"
#include "state_finder/transpose_solver.hpp"

#include "pipe_sim/transient.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace state_finder {

TangentLinearResult evaluate_tangent_linear(
    const ForwardConfiguration& configuration,
    const Trajectory& trajectory,
    const std::vector<double>& scaled_direction,
    InitialStateScales scales)
{
    const int n = configuration.n;
    const int steps = configuration.steps;
    const int width = 2*n + 2;

    if (n < 2 || steps < 0 ||
        scaled_direction.size() !=
            static_cast<std::size_t>(2*n+1) ||
        !std::isfinite(scales.pressure) ||
        !std::isfinite(scales.flow) ||
        scales.pressure <= 0.0 ||
        scales.flow <= 0.0)
        throw std::invalid_argument(
            "Invalid tangent-linear dimensions or scales");

    for (double value : scaled_direction) {
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Nonfinite tangent direction");
    }

    if (trajectory.n != n ||
        trajectory.steps != steps ||
        trajectory.info != 0 ||
        trajectory.completed_steps != steps ||
        trajectory.states.size() !=
            static_cast<std::size_t>(steps+1)*width)
        throw std::invalid_argument(
            "Invalid tangent reference trajectory");

    // Interleaved physical perturbation at t=0.
    std::vector<double> direction(width, 0.0);

    // Prescribed inlet pressure: delta p0 = 0.
    direction[0] = 0.0;

    direction[1] =
        scales.flow*scaled_direction[2*n];

    for (int i = 0; i < n; ++i) {
        direction[2*i+2] =
            scales.flow*scaled_direction[n+i];

        direction[2*i+3] =
            scales.pressure*scaled_direction[i];
    }

    TangentLinearResult output;

    output.inlet_flow.resize(steps+1);
    output.outlet_pressure.resize(steps+1);

    output.inlet_flow[0] = direction[1];
    output.outlet_pressure[0] = direction[width-1];

    constexpr int kl = 2;
    constexpr int ku = 1;
    constexpr int ldab = 6;

    std::vector<double> jacobian(
        static_cast<std::size_t>(ldab)*width);

    std::vector<double> rhs(width);

    for (int k = 1; k <= steps; ++k) {
        // B_k * delta(u_{k-1}).
        old_state_jvp(
            n,
            trajectory.state(k-1),
            configuration.parameters,
            direction,
            rhs);

        for (double& value : rhs)
            value = -value;

        // A_k = dR_k / d(u_k).
        pipe_sim::assemble_jacobian_banded(
            n,
            trajectory.state(k),
            configuration.parameters,
            jacobian,
            ldab);

        // A_k * delta(u_k) = -B_k * delta(u_{k-1}).
        const auto solved = solve_banded(
            width, jacobian, ldab, kl, ku, rhs);

        output.max_linear_backward_error = std::max(
            output.max_linear_backward_error,
            solved.relative_backward_error);

        direction = solved.solution;

        // The prescribed inlet-pressure perturbation
        // should remain zero.
        if (!std::isfinite(direction[0]) ||
            std::abs(direction[0]) > 1.0e-8)
            throw std::runtime_error(
                "Prescribed inlet-pressure tangent drift");

        output.inlet_flow[k] = direction[1];
        output.outlet_pressure[k] = direction[width-1];

        for (double value : direction) {
            if (!std::isfinite(value))
                throw std::runtime_error(
                    "Nonfinite tangent-linear state");
        }
    }

    output.final_state = std::move(direction);

    return output;
}


TangentLinearResult evaluate_tangent_linear(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state,
    const std::vector<double>& scaled_direction,
    InitialStateScales scales)
{
    const auto trajectory =
        record_trajectory(configuration, initial_state);

    return evaluate_tangent_linear(
        configuration,
        trajectory,
        scaled_direction,
        scales);
}

} // namespace state_finder
