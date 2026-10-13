#include "state_finder/trajectory.hpp"
#include "state_finder/old_state_jvp.hpp"
#include "state_finder/banded_multi_rhs.hpp"

#include "pipe_sim/transient.hpp"

// Reuse the benchmark configuration and reference-state
// construction without duplicating their physical parameters.
//
// This is an exploratory diagnostic, not production code.
#define main observability_benchmark_main
#include "../tests/cpp/analytic_initial_state_observability.cpp"
#undef main

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    using namespace state_finder;

    try {
        const auto config = configuration();
        const auto initial = reference_state(config);

        const auto trajectory =
            record_trajectory(config, initial);

        if (trajectory.info != 0 ||
            trajectory.completed_steps != config.steps)
            throw std::runtime_error("Trajectory failed");

        const int n = config.n;
        const int width = 2*n + 2;
        const int columns = width - 1;

        // Independent physical state coordinates:
        // u[1], u[2], ..., u[width-1].
        //
        // u[0] is prescribed inlet pressure.

        for (int selected_step : {20,40,80,200,400,600,800}) {
        const auto old_state =
            trajectory.state(selected_step - 1);

        const auto new_state =
            trajectory.state(selected_step);

        std::vector<double> jacobian(
            static_cast<std::size_t>(6)*width);

        pipe_sim::assemble_jacobian_banded(
            n,
            new_state,
            config.parameters,
            jacobian,
            6);

        // Multiple RHS, column-major.
        std::vector<double> rhs(
            static_cast<std::size_t>(width)*columns,
            0.0);

        std::vector<double> direction(width, 0.0);
        std::vector<double> result(width, 0.0);

        for (int j = 0; j < columns; ++j) {
            std::fill(
                direction.begin(),
                direction.end(),
                0.0);

            direction[j+1] = 1.0;

            old_state_jvp(
                n,
                old_state,
                config.parameters,
                direction,
                result);

            for (int i = 0; i < width; ++i)
                rhs[
                    static_cast<std::size_t>(j)*width+i
                ] = -result[i];
        }

        constexpr int kl = 2;
        constexpr int ku = 1;
        constexpr int ldab = 6;

        const auto solved = solve_banded_multi_rhs(
            width,
            jacobian,
            ldab,
            kl,
            ku,
            columns,
            rhs);

        std::filesystem::create_directories(
            "benchmarks/results-m17e");

        const std::string filename =
            "benchmarks/results-m17e/T_" +
            std::to_string(selected_step) + ".csv";

        std::ofstream file(filename);

        if (!file)
            throw std::runtime_error(
                "Cannot open operator CSV");

        file << std::setprecision(17);

        // Extract independent physical coordinates.
        // Write matrix in row-major CSV order.
        for (int i = 0; i < columns; ++i) {
            for (int j = 0; j < columns; ++j) {
                if (j)
                    file << ',';

                file << solved.solution[
                    static_cast<std::size_t>(j)*width+i+1
                ];
            }

            file << '\n';
        }

        if (!file)
            throw std::runtime_error(
                "Operator CSV write failed");

        std::cout << std::setprecision(17);

        std::cout
            << "M1.7e discrete modal operator\n"
            << "step=" << selected_step << '\n'
            << "time=" << selected_step*config.parameters.dt
            << '\n'
            << "dimension=" << columns << '\n'
            << "dt=" << config.parameters.dt << '\n'
            << "max_linear_backward_error="
            << solved.relative_backward_error << '\n'
            << "output=" << filename << '\n'
            << "PASS discrete modal operator\n";

        }
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return 1;
    }
}
