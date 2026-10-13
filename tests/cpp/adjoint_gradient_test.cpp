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

ForwardConfiguration make_configuration()
{
    const double diameter = 1.0;
    const double area = std::acos(-1.0)/4.0;

    return {
        10,
        12,
        pipe_sim::TransientParameters{
            1000.0, 15.0, 0.65,
            diameter, area, 288.15,
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

InitialState make_initial_state(int n)
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

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Dot-product dimensions differ");

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i]*b[i];

    return result;
}

void run_case(bool include_initial, int seed)
{
    const auto config = make_configuration();

    // Generate synthetic observations from a reference state.
    const auto truth = make_initial_state(config.n);
    const auto reference = evaluate_forward(config, truth);

    if (reference.info != 0 ||
        reference.completed_steps != config.steps)
        throw std::runtime_error("Reference simulation failed");

    Observations observations;
    observations.inlet_flow = reference.inlet_flow;
    observations.outlet_pressure = reference.outlet_pressure;
    observations.sigma_flow = 1.0;
    observations.sigma_pressure = 1.0e4;
    observations.include_initial = include_initial;

    // Perturb the initial state to create a nonzero gradient.
    auto candidate = truth;

    for (int i = 1; i <= config.n; ++i) {
        candidate.pressure[i] +=
            1000.0*std::sin(0.31*(i+seed));
    }

    for (int i = 0; i < config.n; ++i) {
        candidate.flow[i] +=
            0.5*std::cos(0.47*(i+seed));
    }

    candidate.inlet_flow += 0.7;

    const auto adjoint = evaluate_adjoint(
        config, candidate, observations);

    const auto direct = evaluate_objective(
        config, candidate, observations);

    if (adjoint.objective != direct.value)
        throw std::runtime_error(
            "Adjoint objective differs from forward objective");

    const auto variables = pack_initial_state(candidate);

    if (adjoint.gradient.size() != variables.size())
        throw std::runtime_error("Gradient dimension mismatch");

    // Physically scaled direction:
    // pressure components ~1e5 Pa,
    // mass-flow components ~10 kg/s.
    std::vector<double> direction(variables.size());

    for (std::size_t i = 0; i < direction.size(); ++i) {
        const bool pressure =
            i < static_cast<std::size_t>(config.n);

        direction[i] =
            (pressure ? 1.0e5 : 10.0) *
            std::sin(0.43*(i+1+seed));
    }

    const double analytic =
        dot(adjoint.gradient, direction);

    std::cout
        << std::setprecision(16)
        << "include_initial=" << include_initial
        << " seed=" << seed
        << " objective=" << adjoint.objective
        << " adjoint_directional=" << analytic
        << " max_linear_backward_error="
        << adjoint.max_linear_backward_error
        << '\n';

    double best_error = std::numeric_limits<double>::infinity();

    for (double h : {
        1.0e-2, 3.0e-3, 1.0e-3,
        3.0e-4, 1.0e-4, 3.0e-5,
        1.0e-5, 3.0e-6, 1.0e-6
    }) {
        const auto fd = central_directional_difference(
            config, candidate, observations, direction, h);

        const double scale =
            std::max(1.0, std::abs(analytic));

        const double error =
            std::abs(fd.derivative-analytic)/scale;

        best_error = std::min(best_error, error);

        std::cout
            << "  h=" << h
            << " FD=" << fd.derivative
            << " relative_error=" << error
            << '\n';
    }

    std::cout << "  best_relative_error="
              << best_error << '\n';

    // This is a diagnostic threshold for the full
    // finite-difference experiment, not a tolerance
    // for analytic residual derivatives.
    if (best_error > 1.0e-4)
        throw std::runtime_error(
            "Adjoint directional derivative verification failed");

    if (adjoint.max_linear_backward_error > 1.0e-12)
        throw std::runtime_error(
            "Adjoint transpose solve backward error too large");
}

} // namespace

int main()
{
    try {
        for (bool include_initial : {false, true}) {
            for (int seed : {1, 2, 3})
                run_case(include_initial, seed);
        }

        std::cout << "PASS complete discrete adjoint\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
