#include "state_finder/tangent_linear.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
        require(
            argc == 3,
            "Usage: executable DIRECTION_TXT OUTPUT_CSV"
        );

        const auto direction =
            read_direction(argv[1]);

        const auto config = configuration();
        const auto truth = reference_state(config);

        const InitialStateScales scales{
            1.0e5,
            10.0
        };

        const auto start =
            std::chrono::steady_clock::now();

        const auto result =
            evaluate_tangent_linear(
                config,
                truth,
                direction,
                scales
            );

        const auto stop =
            std::chrono::steady_clock::now();

        const double elapsed =
            std::chrono::duration<double>(
                stop-start
            ).count();

        require(
            result.inlet_flow.size() == steps+1 &&
            result.outlet_pressure.size() == steps+1,
            "Unexpected tangent response length"
        );

        const std::filesystem::path output(argv[2]);

        if (!output.parent_path().empty())
            std::filesystem::create_directories(
                output.parent_path()
            );

        std::ofstream file(output);

        require(
            static_cast<bool>(file),
            "Cannot open output CSV"
        );

        file << std::setprecision(17);

        file
            << "time_s,inlet_flow_kg_s,"
            << "outlet_pressure_pa\n";

        for (int k = 0; k <= steps; ++k) {
            file
                << k*config.parameters.dt << ','
                << result.inlet_flow[k] << ','
                << result.outlet_pressure[k] << '\n';
        }

        require(
            static_cast<bool>(file),
            "Failed writing response CSV"
        );

        std::cout
            << std::setprecision(17)
            << "method=single_direction_tangent\n"
            << "cells=" << n << '\n'
            << "steps=" << steps << '\n'
            << "direction_dimension=" << dimension << '\n'
            << "response_channels=2\n"
            << "elapsed_seconds=" << elapsed << '\n'
            << "max_linear_backward_error="
            << result.max_linear_backward_error << '\n'
            << "output=" << output.string() << '\n'
            << "PASS M1.7h direct response\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL M1.7h: "
            << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
