#include "state_finder/adjoint.hpp"
#include "state_finder/finite_difference.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace state_finder;

void run_case(bool include_initial)
{
    constexpr int n = 10;
    constexpr int steps = 12;

    const double area = std::acos(-1.0)/4.0;

    const ForwardConfiguration config{
        n,
        steps,
        pipe_sim::TransientParameters{
            1000.0, 15.0, 0.65,
            1.0, area, 288.15, 0.9,
            500.0, 1.1e-5, 4.5e-5
        },
        pipe_sim::DemandRamp{
            8.0e6, 100.0, 105.0, 600.0
        },
        pipe_sim::NewtonOptions{
            1.0e-9, 1.0e-12, 30, false
        }
    };

    InitialState truth{
        std::vector<double>(n+1),
        std::vector<double>(n, 100.0),
        100.0
    };

    for (int i = 0; i <= n; ++i)
        truth.pressure[i] = 8.0e6-1500.0*i;

    const auto reference =
        evaluate_forward(config, truth);

    if (reference.info != 0 ||
        reference.completed_steps != steps)
        throw std::runtime_error(
            "Reference forward simulation failed");

    Observations observations;
    observations.inlet_flow = reference.inlet_flow;
    observations.outlet_pressure =
        reference.outlet_pressure;
    observations.sigma_flow = 1.0;
    observations.sigma_pressure = 1.0e4;
    observations.include_initial = include_initial;

    auto candidate = truth;

    for (int i = 1; i <= n; ++i)
        candidate.pressure[i] +=
            1000.0*std::sin(0.31*(i+1));

    for (int i = 0; i < n; ++i)
        candidate.flow[i] +=
            0.5*std::cos(0.47*(i+1));

    candidate.inlet_flow += 0.7;

    const auto adjoint =
        evaluate_adjoint(config, candidate, observations);

    const std::vector<int> indices{
        0, n/2, n-1,
        n, n+n/2, 2*n-1,
        2*n
    };

    for (int index : indices) {
        std::vector<double> direction(
            static_cast<std::size_t>(2*n+1), 0.0);

        const double physical_scale =
            index < n ? 1.0e5 : 10.0;

        direction[index] = physical_scale;

        const double analytic =
            adjoint.gradient[index]*physical_scale;

        double best = std::numeric_limits<double>::infinity();
        double best_step = 0.0;

        for (double h : {
            1.0e-2, 3.0e-3, 1.0e-3,
            3.0e-4, 1.0e-4, 3.0e-5,
            1.0e-5, 3.0e-6, 1.0e-6
        }) {
            const auto fd =
                central_directional_difference(
                    config,
                    candidate,
                    observations,
                    direction,
                    h);

            const double error =
                std::abs(fd.derivative-analytic)/
                std::max(1.0, std::abs(analytic));

            if (error < best) {
                best = error;
                best_step = h;
            }
        }

        std::cout
            << std::setprecision(16)
            << "include_initial=" << include_initial
            << " variable=" << index
            << " analytic_scaled=" << analytic
            << " best_relative_error=" << best
            << " best_step=" << best_step
            << '\n';

        // Diagnostic smoke-test threshold only.
        // This is not the production derivative tolerance.
        if (best > 1.0e-4)
            throw std::runtime_error(
                "Componentwise adjoint verification failed");
    }
}

} // namespace

int main()
{
    try {
        run_case(false);
        run_case(true);

        std::cout
            << "PASS componentwise adjoint verification\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
