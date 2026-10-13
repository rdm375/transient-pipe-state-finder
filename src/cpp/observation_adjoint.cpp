#include "state_finder/observation_adjoint.hpp"

#include "state_finder/old_state_vjp.hpp"
#include "state_finder/transpose_solver.hpp"

#include "pipe_sim/transient.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace state_finder {

ObservationAdjointResult apply_observation_adjoint(
    const ForwardConfiguration& configuration,
    const Trajectory& trajectory,
    std::span<const double> forcing,
    int start_step,
    double sigma_flow,
    double sigma_pressure)
{
    const int n = configuration.n;
    const int steps = configuration.steps;
    const int width = 2*n + 2;

    if (n < 2 || steps < 0)
        throw std::invalid_argument(
            "Invalid observation-adjoint dimensions");

    if (start_step < 0 || start_step > steps)
        throw std::invalid_argument(
            "Invalid observation-window start");

    if (!(std::isfinite(sigma_flow) &&
          sigma_flow > 0.0) ||
        !(std::isfinite(sigma_pressure) &&
          sigma_pressure > 0.0))
        throw std::invalid_argument(
            "Invalid observation-noise scales");

    if (trajectory.n != n ||
        trajectory.steps != steps ||
        trajectory.info != 0 ||
        trajectory.completed_steps != steps ||
        trajectory.states.size() !=
            static_cast<std::size_t>(steps+1)*width)
        throw std::invalid_argument(
            "Invalid recorded trajectory");

    if (forcing.size() !=
        static_cast<std::size_t>(2)*(steps+1))
        throw std::invalid_argument(
            "Invalid observation-forcing dimension");

    for (double value : forcing) {
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Nonfinite observation forcing");
    }

    ObservationAdjointResult output;
    output.gradient.assign(2*n+1, 0.0);

    // The adjoint uses the forward residual Jacobians.
    //
    // Local observation derivatives are nonzero only
    // at inlet flow (interleaved index 1) and outlet
    // pressure (interleaved index width-1).
    //
    // g_k = C^T R^(-1/2) w_k.
    //
    // We construct these directly, without subtracting
    // synthetic observations from reference values.

    std::vector<double> carry(width, 0.0);
    std::vector<double> rhs(width, 0.0);
    std::vector<double> previous_carry(width, 0.0);

    constexpr int kl = 2;
    constexpr int ku = 1;
    constexpr int ldab = 6;

    std::vector<double> jacobian(
        static_cast<std::size_t>(ldab)*width);

    for (int k = steps; k >= 1; --k) {
        // rhs = g_k - carry.
        for (int j = 0; j < width; ++j)
            rhs[j] = -carry[j];

        if (k >= start_step) {
            rhs[1] +=
                forcing[2*k]/sigma_flow;

            rhs[width-1] +=
                forcing[2*k+1]/sigma_pressure;
        }

        pipe_sim::assemble_jacobian_banded(
            n,
            trajectory.state(k),
            configuration.parameters,
            jacobian,
            ldab);

        const auto solve =
            solve_transposed_banded(
                width,
                jacobian,
                ldab,
                kl,
                ku,
                rhs);

        output.max_linear_backward_error =
            std::max(
                output.max_linear_backward_error,
                solve.relative_backward_error);

        // carry = B_k^T lambda_k.
        old_state_vjp(
            n,
            trajectory.state(k-1),
            configuration.parameters,
            solve.solution,
            previous_carry);

        carry.swap(previous_carry);
    }

    // At time zero:
    //
    // gradient = g_0 - carry.
    std::vector<double> initial_gradient(width);

    for (int j = 0; j < width; ++j)
        initial_gradient[j] = -carry[j];

    if (start_step == 0) {
        initial_gradient[1] +=
            forcing[0]/sigma_flow;

        initial_gradient[width-1] +=
            forcing[1]/sigma_pressure;
    }

    // Convert from interleaved physical-state ordering
    // to permanent packed initial-state ordering.
    //
    // Fixed inlet pressure p[0] is excluded.
    for (int i = 0; i < n; ++i) {
        output.gradient[i] =
            initial_gradient[2*i+3];

        output.gradient[n+i] =
            initial_gradient[2*i+2];
    }

    output.gradient[2*n] =
        initial_gradient[1];

    for (double value : output.gradient) {
        if (!std::isfinite(value))
            throw std::runtime_error(
                "Nonfinite observation-adjoint gradient");
    }

    return output;
}

} // namespace state_finder
