#include "state_finder/tangent_linear.hpp"
#include "state_finder/adjoint.hpp"
#include "state_finder/observation_adjoint.hpp"
#include "state_finder/trajectory.hpp"
#include "state_finder/forward_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace state_finder;

constexpr int n = 100;
constexpr int steps = 800;
constexpr int dimension = 2*n + 1;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

ForwardConfiguration configuration()
{
    const double area = std::acos(-1.0)/4.0;

    return {
        n,
        steps,
        pipe_sim::TransientParameters{
            100000.0/n,
            15.0,
            0.65,
            1.0,
            area,
            288.15,
            0.9,
            500.0,
            1.1e-5,
            4.5e-5
        },
        pipe_sim::DemandRamp{
            8.0e6,
            100.0,
            105.0,
            600.0
        },
        pipe_sim::NewtonOptions{
            1.0e-9,
            1.0e-12,
            30,
            false
        }
    };
}

InitialState reference_state(
    const ForwardConfiguration& config)
{
    const auto& par = config.parameters;

    InitialState state{
        std::vector<double>(n+1),
        std::vector<double>(n, 100.0),
        100.0
    };

    const double mdot = 100.0;

    const double reynolds =
        mdot*par.diameter/
        (par.area*par.viscosity);

    const double argument =
        par.roughness/(3.7*par.diameter) +
        5.74/std::pow(reynolds, 0.9);

    const double friction =
        0.25/std::pow(std::log10(argument), 2.0);

    const double coefficient =
        friction*par.z*par.gas_constant*
        par.temperature*mdot*mdot/
        (par.diameter*par.area*par.area);

    for (int i = 0; i <= n; ++i) {
        const double squared =
            config.demand.inlet_pressure*
            config.demand.inlet_pressure -
            coefficient*i*par.dx;

        require(
            squared > 0.0,
            "Invalid reference pressure"
        );

        state.pressure[i] = std::sqrt(squared);
    }

    return state;
}

std::vector<double> read_direction(
    const std::string& filename)
{
    std::ifstream file(filename);

    require(
        static_cast<bool>(file),
        "Cannot open predictive direction"
    );

    std::vector<double> direction;

    double value;

    while (file >> value)
        direction.push_back(value);

    require(
        direction.size() == dimension,
        "Predictive direction must have 201 entries"
    );

    for (double x : direction)
        require(
            std::isfinite(x),
            "Nonfinite predictive direction"
        );

    return direction;
}

} // namespace



int main(int argc, char** argv)
{
    try {
        require(argc == 3,
                "Usage: executable START_STEP OUTPUT_TXT");

        const int start_step = std::stoi(argv[1]);

        require(
            start_step >= 0 && start_step <= steps,
            "Invalid observation window");

        const auto config = configuration();
        const auto truth = reference_state(config);

        const auto trajectory =
            record_trajectory(config, truth);

        require(
            trajectory.info == 0 &&
            trajectory.completed_steps == steps,
            "Reference trajectory failed");

        constexpr double sigma_m = 0.01;
        constexpr double sigma_p = 10.0;

        const InitialStateScales scales{
            1.0e5, 10.0
        };

        std::vector<double> prior(dimension, 1.0);

        for (int j = 0; j < n; ++j)
            prior[j] = 1.0e4;

        std::mt19937_64 rng(20261013);
        std::normal_distribution<double> normal(0.0, 1.0);

        std::vector<double> v(dimension);
        std::vector<double> w(2*(steps+1));

        for (double& x : v)
            x = normal(rng);

        for (double& x : w)
            x = normal(rng);

        std::vector<double> direction(dimension);

        for (int j = 0; j < dimension; ++j) {
            const double state_scale =
                j < n ? scales.pressure : scales.flow;

            direction[j] =
                prior[j]*v[j]/state_scale;
        }

        const auto shared =
            evaluate_tangent_linear(
                config, trajectory, direction, scales);

        const auto legacy =
            evaluate_tangent_linear(
                config, truth, direction, scales);

        double tangent_difference = 0.0;
        double tangent_reference = 0.0;
        double lhs = 0.0;

        for (int k = 0; k <= steps; ++k) {
            const double dm =
                shared.inlet_flow[k] -
                legacy.inlet_flow[k];

            const double dp =
                shared.outlet_pressure[k] -
                legacy.outlet_pressure[k];

            tangent_difference += dm*dm + dp*dp;

            tangent_reference +=
                shared.inlet_flow[k]*shared.inlet_flow[k] +
                shared.outlet_pressure[k]*shared.outlet_pressure[k];

            if (k >= start_step) {
                lhs +=
                    (shared.inlet_flow[k]/sigma_m)*w[2*k];

                lhs +=
                    (shared.outlet_pressure[k]/sigma_p)*
                    w[2*k+1];
            }
        }

        const double tangent_relative_error =
            std::sqrt(tangent_difference)/
            std::max(std::sqrt(tangent_reference), 1.0);

        const auto adjoint =
            apply_observation_adjoint(
                config,
                trajectory,
                w,
                start_step,
                sigma_m,
                sigma_p);

        std::vector<double> ktw(dimension);

        double rhs = 0.0;

        for (int j = 0; j < dimension; ++j) {
            ktw[j] = prior[j]*adjoint.gradient[j];
            rhs += v[j]*ktw[j];
        }

        const double relative_error =
            std::abs(lhs-rhs)/
            std::max({
                1.0,
                std::abs(lhs),
                std::abs(rhs)
            });

        std::ofstream file(argv[2]);

        require(
            static_cast<bool>(file),
            "Cannot open output file");

        file << std::setprecision(17);

        for (double x : ktw)
            file << x << '\n';

        require(
            static_cast<bool>(file),
            "Failed writing transpose result");

        std::ofstream forcing_file(
            std::string(argv[2]) + ".forcing");

        require(
            static_cast<bool>(forcing_file),
            "Cannot open forcing output");

        forcing_file << std::setprecision(17);

        for (double x : w)
            forcing_file << x << '\n';

        require(
            static_cast<bool>(forcing_file),
            "Failed writing forcing vector");

        std::cout << std::setprecision(17)
                  << "start_step=" << start_step << '\n'
                  << "lhs=" << lhs << '\n'
                  << "rhs=" << rhs << '\n'
                  << "dot_relative_error="
                  << relative_error << '\n'
                  << "tangent_relative_error="
                  << tangent_relative_error << '\n'
                  << "tangent_backward_error="
                  << shared.max_linear_backward_error << '\n'
                  << "adjoint_backward_error="
                  << adjoint.max_linear_backward_error << '\n';

        require(
            tangent_relative_error < 1.0e-13,
            "Shared tangent disagrees with original");

        require(
            relative_error < 1.0e-10,
            "Direct adjoint dot-product identity failed");

        std::cout
            << "PASS M1.8c independent transpose identity\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL M1.8c transpose verification: "
            << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
