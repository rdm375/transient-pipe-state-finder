#include "state_finder/adjoint.hpp"

#include "state_finder/old_state_vjp.hpp"
#include "state_finder/transpose_solver.hpp"

#include "pipe_sim/transient.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace state_finder {

AdjointResult evaluate_adjoint(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state,
    const Observations& observations)
{
    const int n = configuration.n;
    const int steps = configuration.steps;
    const int width = 2*n + 2;

    if (steps < 0 || n < 2)
        throw std::invalid_argument("Invalid adjoint dimensions");

    const auto count = static_cast<std::size_t>(steps + 1);

    if (observations.inlet_flow.size() != count ||
        observations.outlet_pressure.size() != count)
        throw std::invalid_argument("Observation dimension mismatch");

    const double sm = observations.sigma_flow;
    const double sp = observations.sigma_pressure;

    if (!std::isfinite(sm) || sm <= 0.0 ||
        !std::isfinite(sp) || sp <= 0.0)
        throw std::invalid_argument("Invalid observation scales");

    const auto trajectory =
        record_trajectory(configuration, initial_state);

    if (trajectory.info != 0 ||
        trajectory.completed_steps != steps)
        throw std::runtime_error("Forward trajectory failed");

    AdjointResult output;
    output.gradient.assign(2*n + 1, 0.0);
    output.linear_backward_errors.assign(steps, 0.0);

    // g_k = partial j_k / partial u_k.
    // Store all local observation gradients.
    std::vector<double> local_gradients(
        count * static_cast<std::size_t>(width), 0.0);

    const int first = observations.include_initial ? 0 : 1;
    double sum = 0.0;

    for (int k = first; k <= steps; ++k) {
        const double dm =
            trajectory.inlet_flow[k] - observations.inlet_flow[k];

        const double dp =
            trajectory.outlet_pressure[k] -
            observations.outlet_pressure[k];

        const double rm = dm / sm;
        const double rp = dp / sp;

        if (!std::isfinite(rm) || !std::isfinite(rp))
            throw std::runtime_error("Nonfinite observation residual");

        // Same accumulation order as evaluate_objective().
        sum += rm*rm + rp*rp;

        const auto offset =
            static_cast<std::size_t>(k) * width;

        local_gradients[offset + 1] = rm / sm;
        local_gradients[offset + width - 1] = rp / sp;
    }

    output.objective = 0.5*sum;

    if (!std::isfinite(output.objective))
        throw std::runtime_error("Nonfinite objective");

    // Carry = B_(k+1)^T lambda_(k+1).
    std::vector<double> carry(width, 0.0);
    std::vector<double> rhs(width, 0.0);
    std::vector<double> previous_carry(width, 0.0);

    constexpr int kl = 2;
    constexpr int ku = 1;
    constexpr int ldab = 6;

    std::vector<double> jacobian(
        static_cast<std::size_t>(ldab) * width);

    for (int k = steps; k >= 1; --k) {
        const auto offset =
            static_cast<std::size_t>(k) * width;

        for (int j = 0; j < width; ++j) {
            rhs[j] = local_gradients[offset + j] - carry[j];
        }

        // A_k = dR_k/du_k, using the forward solver's
        // existing analytic banded Jacobian.
        pipe_sim::assemble_jacobian_banded(
            n,
            trajectory.state(k),
            configuration.parameters,
            jacobian,
            ldab);

        const auto solve = solve_transposed_banded(
            width, jacobian, ldab, kl, ku, rhs);

        output.linear_backward_errors[k-1] =
            solve.relative_backward_error;

        output.max_linear_backward_error = std::max(
            output.max_linear_backward_error,
            solve.relative_backward_error);

        // B_k^T lambda_k.
        old_state_vjp(
            n,
            trajectory.state(k-1),
            configuration.parameters,
            solve.solution,
            previous_carry);

        carry.swap(previous_carry);
    }

    // Full interleaved initial-state gradient.
    std::vector<double> initial_gradient(width);

    for (int j = 0; j < width; ++j)
        initial_gradient[j] = local_gradients[j] - carry[j];

    // Convert to the permanent optimization ordering:
    // p[1..n], face_flow[0..n-1], inlet_flow.
    //
    // p[0] is fixed by the inlet-pressure boundary condition.
    for (int i = 0; i < n; ++i) {
        output.gradient[i] =
            initial_gradient[2*i + 3];

        output.gradient[n+i] =
            initial_gradient[2*i + 2];
    }

    output.gradient[2*n] = initial_gradient[1];

    for (double value : output.gradient) {
        if (!std::isfinite(value))
            throw std::runtime_error("Nonfinite adjoint gradient");
    }

    return output;
}

} // namespace state_finder
