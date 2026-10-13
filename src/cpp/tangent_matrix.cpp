#include "state_finder/tangent_matrix.hpp"

#include "state_finder/banded_multi_rhs.hpp"
#include "state_finder/old_state_jvp.hpp"
#include "state_finder/trajectory.hpp"

#include "pipe_sim/transient.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace state_finder {

TangentMatrixResult evaluate_tangent_matrix(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state,
    double sigma_flow,
    double sigma_pressure,
    InitialStateScales scales)
{
    const int n = configuration.n;
    const int steps = configuration.steps;

    if (n < 2 || steps < 0 ||
        !std::isfinite(sigma_flow) ||
        !std::isfinite(sigma_pressure) ||
        !std::isfinite(scales.flow) ||
        !std::isfinite(scales.pressure) ||
        sigma_flow <= 0.0 ||
        sigma_pressure <= 0.0 ||
        scales.flow <= 0.0 ||
        scales.pressure <= 0.0)
        throw std::invalid_argument(
            "Invalid tangent-matrix parameters");

    const int width = 2*n + 2;
    const int columns = 2*n + 1;
    const int rows = 2*(steps+1);

    const auto trajectory =
        record_trajectory(configuration, initial_state);

    if (trajectory.info != 0 ||
        trajectory.completed_steps != steps)
        throw std::runtime_error(
            "Tangent-matrix forward trajectory failed");

    TangentMatrixResult output;
    output.steps = steps;
    output.columns = columns;

    output.observations.assign(
        static_cast<std::size_t>(rows)*columns,
        0.0);

    // State sensitivity matrix S:
    //
    // width x columns, column-major.
    //
    // Column j represents the derivative of the
    // physical interleaved state with respect to
    // dimensionless initial-state coordinate j.
    std::vector<double> S(
        static_cast<std::size_t>(width)*columns,
        0.0);

    // Initial pressure columns.
    for (int i = 0; i < n; ++i)
        S[static_cast<std::size_t>(i)*width + 2*i+3] =
            scales.pressure;

    // Initial internal face-flow columns.
    for (int i = 0; i < n; ++i)
        S[static_cast<std::size_t>(n+i)*width + 2*i+2] =
            scales.flow;

    // Initial inlet-flow column.
    S[static_cast<std::size_t>(2*n)*width + 1] =
        scales.flow;

    auto extract_observations = [&](int k) {
        const auto row =
            static_cast<std::size_t>(2*k)*columns;

        for (int j = 0; j < columns; ++j) {
            const auto offset =
                static_cast<std::size_t>(j)*width;

            output.observations[row+j] =
                S[offset+1]/sigma_flow;

            output.observations[row+columns+j] =
                S[offset+width-1]/sigma_pressure;
        }
    };

    extract_observations(0);

    constexpr int kl = 2;
    constexpr int ku = 1;
    constexpr int ldab = 6;

    std::vector<double> jacobian(
        static_cast<std::size_t>(ldab)*width);

    std::vector<double> rhs(S.size());

    for (int k = 1; k <= steps; ++k) {
        // Assemble -B_k*S_(k-1), column by column.
        for (int j = 0; j < columns; ++j) {
            const auto offset =
                static_cast<std::size_t>(j)*width;

            old_state_jvp(
                n,
                trajectory.state(k-1),
                configuration.parameters,
                std::span<const double>(
                    S.data()+offset, width),
                std::span<double>(
                    rhs.data()+offset, width));

            for (int i = 0; i < width; ++i)
                rhs[offset+i] = -rhs[offset+i];
        }

        pipe_sim::assemble_jacobian_banded(
            n,
            trajectory.state(k),
            configuration.parameters,
            jacobian,
            ldab);

        const auto solved = solve_banded_multi_rhs(
            width,
            jacobian,
            ldab,
            kl,
            ku,
            columns,
            rhs);

        output.max_linear_backward_error = std::max(
            output.max_linear_backward_error,
            solved.relative_backward_error);

        S = solved.solution;

        // The prescribed inlet pressure must have
        // zero sensitivity to every initial-state
        // optimization variable.
        for (int j = 0; j < columns; ++j) {
            const auto offset =
                static_cast<std::size_t>(j)*width;

            if (!std::isfinite(S[offset]) ||
                std::abs(S[offset]) > 1.0e-8)
                throw std::runtime_error(
                    "Tangent-matrix inlet-pressure drift");
        }

        for (double value : S) {
            if (!std::isfinite(value))
                throw std::runtime_error(
                    "Nonfinite tangent-matrix entry");
        }

        extract_observations(k);
    }

    return output;
}

} // namespace state_finder
