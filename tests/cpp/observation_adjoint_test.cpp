#include "state_finder/observation_adjoint.hpp"
#include "state_finder/adjoint.hpp"
#include "state_finder/forward_model.hpp"
#include "state_finder/trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main()
{
    try {
        constexpr int n = 100;
        constexpr int steps = 800;

        const double area = std::acos(-1.0)/4.0;

        const state_finder::ForwardConfiguration config{
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

        // Same steady-state construction as the
        // analytic observability benchmark.
        state_finder::InitialState truth{
            std::vector<double>(n+1),
            std::vector<double>(n, 100.0),
            100.0
        };

        const auto& par = config.parameters;

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

            truth.pressure[i] = std::sqrt(squared);
        }

        const auto trajectory =
            state_finder::record_trajectory(
                config,
                truth);

        require(
            trajectory.info == 0 &&
            trajectory.completed_steps == steps,
            "Reference trajectory failed");

        std::mt19937_64 rng(20261013);
        std::normal_distribution<double> normal(0.0, 1.0);

        std::vector<double> forcing(2*(steps+1));

        for (double& value : forcing)
            value = normal(rng);

        constexpr double sigma_m = 0.01;
        constexpr double sigma_p = 10.0;

        // Compare multiple observation windows.
        for (int start_step : {0, 80, 160, 400}) {
            const auto direct =
                state_finder::apply_observation_adjoint(
                    config,
                    trajectory,
                    forcing,
                    start_step,
                    sigma_m,
                    sigma_p);

            state_finder::Observations observations;

            observations.sigma_flow = sigma_m;
            observations.sigma_pressure = sigma_p;
            observations.include_initial = true;

            observations.inlet_flow.resize(steps+1);
            observations.outlet_pressure.resize(steps+1);

            for (int k = 0; k <= steps; ++k) {
                const double wm =
                    k >= start_step
                    ? forcing[2*k]
                    : 0.0;

                const double wp =
                    k >= start_step
                    ? forcing[2*k+1]
                    : 0.0;

                observations.inlet_flow[k] =
                    trajectory.inlet_flow[k]
                    -sigma_m*wm;

                observations.outlet_pressure[k] =
                    trajectory.outlet_pressure[k]
                    -sigma_p*wp;
            }

            const auto legacy =
                state_finder::evaluate_adjoint(
                    config,
                    truth,
                    observations);

            require(
                direct.gradient.size() ==
                legacy.gradient.size(),
                "Gradient dimension mismatch");

            double difference_squared = 0.0;
            double reference_squared = 0.0;

            for (std::size_t j = 0;
                 j < direct.gradient.size(); ++j) {
                const double difference =
                    direct.gradient[j] -
                    legacy.gradient[j];

                difference_squared +=
                    difference*difference;

                reference_squared +=
                    direct.gradient[j]*
                    direct.gradient[j];
            }

            const double relative_error =
                std::sqrt(difference_squared)/
                std::max(
                    std::sqrt(reference_squared),
                    1.0e-30);

            std::cout << std::setprecision(17)
                      << "start_step=" << start_step
                      << " relative_error="
                      << relative_error
                      << " backward_error="
                      << direct.max_linear_backward_error
                      << '\n';

            require(
                relative_error < 1.0e-8,
                "Direct/legacy adjoint mismatch");
        }

        std::cout
            << "PASS M1.8c-1 direct observation adjoint\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL M1.8c-1: "
            << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
