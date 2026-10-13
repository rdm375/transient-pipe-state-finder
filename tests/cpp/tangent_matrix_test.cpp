#include "state_finder/tangent_linear.hpp"
#include "state_finder/tangent_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace state_finder;

ForwardConfiguration make_configuration(
    int n,
    int steps)
{
    const double area = std::acos(-1.0)/4.0;

    return {
        n,
        steps,
        pipe_sim::TransientParameters{
            1000.0, 15.0, 0.65,
            1.0, area, 288.15,
            0.9, 500.0, 1.1e-5, 4.5e-5
        },
        pipe_sim::DemandRamp{
            8.0e6, 100.0, 105.0, 600.0
        },
        pipe_sim::NewtonOptions{
            1.0e-9, 1.0e-12, 30, false
        }
    };
}

InitialState make_state(int n)
{
    InitialState state{
        std::vector<double>(n+1),
        std::vector<double>(n),
        100.0
    };

    for (int i = 0; i <= n; ++i)
        state.pressure[i] = 8.0e6 - 1500.0*i;

    for (int i = 0; i < n; ++i)
        state.flow[i] = 100.0;

    return state;
}

void run_case(int n, int steps)
{
    const auto config =
        make_configuration(n, steps);

    auto state = make_state(n);

    for (int i = 1; i <= n; ++i)
        state.pressure[i] +=
            1000.0*std::sin(0.31*i);

    for (int i = 0; i < n; ++i)
        state.flow[i] +=
            0.5*std::cos(0.47*(i+1));

    state.inlet_flow += 0.7;

    const int columns = 2*n+1;

    const auto matrix =
        evaluate_tangent_matrix(config, state);

    if (matrix.columns != columns ||
        matrix.steps != steps)
        throw std::runtime_error(
            "Tangent-matrix dimensions incorrect");

    double max_error = 0.0;

    // Check first/middle/last pressure and flow
    // coordinates, plus the inlet-flow coordinate.
    std::vector<int> selected{
        0,
        n/2,
        n-1,
        n,
        n+n/2,
        2*n-1,
        2*n
    };

    std::sort(selected.begin(), selected.end());
    selected.erase(
        std::unique(selected.begin(), selected.end()),
        selected.end());

    for (int j : selected) {
        std::vector<double> direction(columns, 0.0);
        direction[j] = 1.0;

        const auto tangent =
            evaluate_tangent_linear(
                config, state, direction);

        for (int k = 0; k <= steps; ++k) {
            const auto row =
                static_cast<std::size_t>(2*k)*columns;

            const double dm =
                matrix.observations[row+j];

            const double dp =
                matrix.observations[row+columns+j];

            const double expected_dm =
                tangent.inlet_flow[k];

            const double expected_dp =
                tangent.outlet_pressure[k]/1.0e4;

            const double error_dm =
                std::abs(dm-expected_dm) /
                std::max(1.0, std::abs(expected_dm));

            const double error_dp =
                std::abs(dp-expected_dp) /
                std::max(1.0, std::abs(expected_dp));

            max_error = std::max(
                {max_error, error_dm, error_dp});
        }
    }

    std::cout
        << std::setprecision(17)
        << "n=" << n
        << " steps=" << steps
        << " checked_columns=" << selected.size()
        << " max_relative_error=" << max_error
        << " max_backward_error="
        << matrix.max_linear_backward_error
        << '\n';

    if (max_error > 1.0e-11)
        throw std::runtime_error(
            "Matrix/vector tangent mismatch");

    if (matrix.max_linear_backward_error > 1.0e-12)
        throw std::runtime_error(
            "Matrix tangent linear-solve backward error");
}

} // namespace

int main()
{
    try {
        for (int n : {2, 10, 100}) {
            for (int steps : {0, 1, 12, 80})
                run_case(n, steps);
        }

        std::cout
            << "PASS tangent sensitivity matrix\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
