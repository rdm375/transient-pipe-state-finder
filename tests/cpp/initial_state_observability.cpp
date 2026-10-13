#include "state_finder/forward_model.hpp"
#include "state_finder/scaled_initial_state.hpp"
#include "state_finder/observations.hpp"

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
            argc <= 3,
            "Usage: executable [OUTPUT_DIR] [FD_STEP]");

        const std::string output =
            argc >= 2
            ? argv[1]
            : "benchmarks/results-m16b";

        const double h =
            argc >= 3
            ? std::stod(argv[2])
            : 1.0e-5;

        require(
            std::isfinite(h) &&
            h > 0.0 &&
            h <= 1.0e-2,
            "Invalid finite-difference step");

        std::cout << std::setprecision(17)
                  << "finite_difference_step="
                  << h << '\n';

        std::filesystem::create_directories(output);

        const auto config = configuration();
        const auto truth = reference_state(config);

        const InitialStateScales scales{
            1.0e5,
            10.0
        };

        const auto reference = run(config, truth);

        // All four measured boundary histories.
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

        // Column-major storage:
        // H_flow[k,j] and H_pressure[k,j].
        //
        // Columns correspond to dimensionless initial
        // state coordinates.
        //
        // Rows are divided by the observation scales.
        std::vector<double> H_flow(
            static_cast<std::size_t>(steps+1)*variables);

        std::vector<double> H_pressure(
            static_cast<std::size_t>(steps+1)*variables);

        const std::vector<double> zero(variables, 0.0);

        for (int j = 0; j < variables; ++j) {
            auto plus = zero;
            auto minus = zero;

            plus[j] = h;
            minus[j] = -h;

            const auto state_plus =
                unpack_scaled_state(
                    plus, truth, scales);

            const auto state_minus =
                unpack_scaled_state(
                    minus, truth, scales);

            const auto forward_plus =
                run(config, state_plus);

            const auto forward_minus =
                run(config, state_minus);

            for (int k = 0; k <= steps; ++k) {
                const auto index =
                    static_cast<std::size_t>(j)*(steps+1)
                    + k;

                H_flow[index] =
                    (forward_plus.inlet_flow[k] -
                     forward_minus.inlet_flow[k]) /
                    (2.0*h*record.scales.inlet_flow);

                H_pressure[index] =
                    (forward_plus.outlet_pressure[k] -
                     forward_minus.outlet_pressure[k]) /
                    (2.0*h*record.scales.outlet_pressure);

                require(
                    std::isfinite(H_flow[index]) &&
                    std::isfinite(H_pressure[index]),
                    "Nonfinite sensitivity");
            }

            if ((j+1)%20 == 0 || j+1 == variables) {
                std::cout
                    << "columns_completed="
                    << j+1 << '/'
                    << variables << '\n';
            }
        }

        // Write one matrix per horizon.
        // Each row: one time sample, one response channel.
        // First flow, then pressure, in alternating order.
        for (int horizon : {80,100,200,400,800}) {
            const std::string filename =
                output + "/H_" +
                std::to_string(horizon) + ".csv";

            std::ofstream file(filename);

            require(
                file.good(),
                "Unable to open sensitivity output");

            file << std::setprecision(17);

            for (int k = 0; k <= horizon; ++k) {
                for (int channel = 0;
                     channel < 2;
                     ++channel)
                {
                    for (int j = 0; j < variables; ++j) {
                        if (j)
                            file << ',';

                        const auto index =
                            static_cast<std::size_t>(j)*
                            (steps+1) + k;

                        file << (
                            channel == 0
                            ? H_flow[index]
                            : H_pressure[index]);
                    }

                    file << '\n';
                }
            }

            require(
                file.good(),
                "Sensitivity matrix write failed");

            std::cout
                << "wrote=" << filename
                << " rows=" << 2*(horizon+1)
                << " columns=" << variables
                << '\n';
        }

        std::cout
            << "PASS M1.6b sensitivity construction\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
