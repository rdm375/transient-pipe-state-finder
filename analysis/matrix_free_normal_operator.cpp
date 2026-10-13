#include "state_finder/tangent_linear.hpp"
#include "state_finder/adjoint.hpp"
#include "state_finder/forward_model.hpp"

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
            argc == 4,
            "Usage: executable START_STEP INPUT_TXT OUTPUT_TXT"
        );

        const int start_step = std::stoi(argv[1]);

        require(
            start_step >= 0 && start_step <= steps,
            "Invalid observation-window start"
        );

        const auto v = read_direction(argv[2]);

        const auto config = configuration();
        const auto truth = reference_state(config);

        constexpr double sigma_m = 0.01;
        constexpr double sigma_p = 10.0;

        const InitialStateScales scales{
            1.0e5,
            10.0
        };

        // Physical prior standard deviations.
        std::vector<double> prior(dimension, 1.0);

        for (int j = 0; j < n; ++j)
            prior[j] = 1.0e4;

        // Convert prior-standardized coordinates
        // into the tangent solver's dimensionless
        // initial-state coordinates.
        std::vector<double> direction(dimension);

        for (int j = 0; j < dimension; ++j) {
            const double state_scale =
                j < n ? scales.pressure : scales.flow;

            direction[j] =
                prior[j]*v[j]/state_scale;
        }

        // Forward application: K*v.
        const auto tangent =
            evaluate_tangent_linear(
                config,
                truth,
                direction,
                scales
            );

        const auto reference =
            evaluate_forward(config, truth);

        require(
            reference.info == 0 &&
            reference.completed_steps == steps,
            "Reference trajectory failed"
        );

        // Synthetic observations implement
        // the adjoint forcing K*v.
        Observations observations;

        observations.sigma_flow = sigma_m;
        observations.sigma_pressure = sigma_p;
        observations.include_initial = true;

        observations.inlet_flow.resize(steps+1);
        observations.outlet_pressure.resize(steps+1);

        for (int k = 0; k <= steps; ++k) {
            double wm = 0.0;
            double wp = 0.0;

            if (k >= start_step) {
                wm = tangent.inlet_flow[k]/sigma_m;
                wp = tangent.outlet_pressure[k]/sigma_p;
            }

            observations.inlet_flow[k] =
                reference.inlet_flow[k]-sigma_m*wm;

            observations.outlet_pressure[k] =
                reference.outlet_pressure[k]-sigma_p*wp;
        }

        // Transpose application: K^T*(K*v).
        const auto adjoint =
            evaluate_adjoint(
                config,
                truth,
                observations
            );

        require(
            adjoint.gradient.size() == dimension,
            "Unexpected adjoint gradient"
        );

        std::ofstream file(argv[3]);

        require(
            static_cast<bool>(file),
            "Cannot open output"
        );

        file << std::setprecision(17);

        for (int j = 0; j < dimension; ++j) {
            const double value =
                prior[j]*adjoint.gradient[j];

            require(
                std::isfinite(value),
                "Nonfinite normal-operator result"
            );

            file << value << '\n';
        }

        require(
            static_cast<bool>(file),
            "Failed writing normal-operator result"
        );

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL matrix-free normal operator: "
            << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
