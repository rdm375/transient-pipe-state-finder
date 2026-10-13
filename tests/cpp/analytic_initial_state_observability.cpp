#include "state_finder/forward_model.hpp"
#include "state_finder/scaled_initial_state.hpp"
#include "state_finder/observations.hpp"
#include "state_finder/tangent_matrix.hpp"

#include <algorithm>
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
constexpr int variables = 2*n + 1;

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

        require(squared > 0.0,
                "Invalid reference pressure");

        state.pressure[i] = std::sqrt(squared);
    }

    return state;
}

ForwardResult run(
    const ForwardConfiguration& config,
    const InitialState& state)
{
    auto result = evaluate_forward(config, state);

    require(
        result.info == 0 &&
        result.completed_steps == config.steps,
        "Forward integration failed");

    require(
        result.inlet_flow.size() ==
            static_cast<std::size_t>(steps+1) &&
        result.outlet_pressure.size() ==
            static_cast<std::size_t>(steps+1) &&
        result.outlet_flow.size() ==
            static_cast<std::size_t>(steps+1),
        "Unexpected trajectory dimensions");

    return result;
}

} // namespace


int main(int argc, char** argv)
{
    try {
        require(
            argc <= 2,
            "Usage: executable [OUTPUT_DIR]");

        const std::string output =
            argc == 2
            ? argv[1]
            : "benchmarks/results-m16b-analytic";

        std::filesystem::create_directories(output);

        const auto config = configuration();
        const auto truth = reference_state(config);

        const InitialStateScales scales{
            1.0e5,
            10.0
        };

        const auto reference = run(config, truth);

        // Preserve the four-channel observation
        // specification used by the FD experiment.
        BoundaryObservationRecord record;

        record.measured.inlet_pressure.assign(
            steps+1,
            config.demand.inlet_pressure);

        record.measured.inlet_flow =
            reference.inlet_flow;

        record.measured.outlet_pressure =
            reference.outlet_pressure;

        record.measured.outlet_flow =
            reference.outlet_flow;

        record.include_initial = true;

        validate_boundary_observations(
            record, steps+1);

        validate_current_boundary_roles(
            record.roles);

        const auto observations =
            make_current_observations(
                record, steps+1);

        require(
            observations.inlet_flow ==
                reference.inlet_flow &&
            observations.outlet_pressure ==
                reference.outlet_pressure,
            "Observation conversion failed");

        std::cout
            << std::setprecision(17)
            << "method=analytic_tangent_matrix\n"
            << "cells=" << n << '\n'
            << "steps=" << steps << '\n'
            << "variables=" << variables << '\n';

        const auto result =
            evaluate_tangent_matrix(
                config,
                truth,
                record.scales.inlet_flow,
                record.scales.outlet_pressure,
                scales);

        require(
            result.steps == steps &&
            result.columns == variables,
            "Unexpected tangent-matrix dimensions");

        require(
            result.observations.size() ==
                static_cast<std::size_t>(
                    2*(steps+1)*variables),
            "Unexpected sensitivity matrix size");

        for (double value : result.observations)
            require(
                std::isfinite(value),
                "Nonfinite analytic sensitivity");

        std::cout
            << "max_linear_backward_error="
            << result.max_linear_backward_error
            << '\n';

        for (int horizon : {80,100,200,400,800}) {
            const std::string filename =
                output + "/H_" +
                std::to_string(horizon) + ".csv";

            std::ofstream file(filename);

            require(
                static_cast<bool>(file),
                "Cannot open output CSV");

            file << std::setprecision(17);

            const int rows = 2*(horizon+1);

            for (int row = 0; row < rows; ++row) {
                const auto offset =
                    static_cast<std::size_t>(row)*
                    variables;

                for (int j = 0; j < variables; ++j) {
                    if (j != 0)
                        file << ',';

                    file << result.observations[
                        offset+j];
                }

                file << '\n';
            }

            require(
                static_cast<bool>(file),
                "CSV output failed");

            std::cout
                << "horizon=" << horizon
                << " rows=" << rows
                << " columns=" << variables
                << " file=" << filename
                << '\n';
        }

        std::cout
            << "PASS analytic observability matrices\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
