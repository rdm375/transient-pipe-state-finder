#include "state_finder/adjoint.hpp"
#include "state_finder/forward_model.hpp"
#include "state_finder/scaled_initial_state.hpp"
#include "state_finder/tangent_linear.hpp"

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

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Dot dimension mismatch");

    double sum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        sum += a[i]*b[i];

    return sum;
}

void run_case(
    int n,
    int steps,
    bool include_initial)
{
    const auto config =
        make_configuration(n, steps);

    const auto truth = make_state(n);

    const auto reference =
        evaluate_forward(config, truth);

    if (reference.info != 0 ||
        reference.completed_steps != steps)
        throw std::runtime_error(
            "Reference forward solve failed");

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
            1000.0*std::sin(0.31*i);

    for (int i = 0; i < n; ++i)
        candidate.flow[i] +=
            0.5*std::cos(0.47*(i+1));

    candidate.inlet_flow += 0.7;

    const auto forward =
        evaluate_forward(config, candidate);

    if (forward.info != 0 ||
        forward.completed_steps != steps)
        throw std::runtime_error(
            "Candidate forward solve failed");

    const InitialStateScales scales{1.0e5, 10.0};

    std::vector<double> direction(2*n+1);

    for (int i = 0; i < 2*n+1; ++i)
        direction[i] =
            std::sin(0.43*(i+1));

    const auto tangent =
        evaluate_tangent_linear(
            config, candidate, direction, scales);

    const auto adjoint =
        evaluate_adjoint(
            config, candidate, observations);

    const auto scaled_gradient =
        scale_adjoint_gradient(
            adjoint.gradient, n, scales);

    const double adjoint_directional =
        dot(scaled_gradient, direction);

    double tangent_directional = 0.0;

    const int first =
        include_initial ? 0 : 1;

    for (int k = first; k <= steps; ++k) {
        const double rm =
            forward.inlet_flow[k] -
            observations.inlet_flow[k];

        const double rp =
            forward.outlet_pressure[k] -
            observations.outlet_pressure[k];

        tangent_directional +=
            rm*tangent.inlet_flow[k] /
                (observations.sigma_flow*
                 observations.sigma_flow);

        tangent_directional +=
            rp*tangent.outlet_pressure[k] /
                (observations.sigma_pressure*
                 observations.sigma_pressure);
    }

    const double scale = std::max(
        {1.0,
         std::abs(adjoint_directional),
         std::abs(tangent_directional)});

    const double relative_error =
        std::abs(
            tangent_directional-adjoint_directional
        )/scale;

    std::cout
        << std::setprecision(17)
        << "n=" << n
        << " steps=" << steps
        << " include_initial=" << include_initial
        << " tangent=" << tangent_directional
        << " adjoint=" << adjoint_directional
        << " relative_error=" << relative_error
        << " max_backward_error="
        << tangent.max_linear_backward_error
        << '\n';

    if (relative_error > 1.0e-9)
        throw std::runtime_error(
            "Tangent/adjoint directional mismatch");

    if (tangent.max_linear_backward_error > 1.0e-12)
        throw std::runtime_error(
            "Tangent linear solve backward error too large");

    // At zero steps, propagation must reproduce
    // the initial physical perturbation.
    if (steps == 0) {
        const double expected_inlet =
            scales.flow*direction[2*n];

        const double expected_outlet =
            scales.pressure*direction[n-1];

        if (std::abs(
                tangent.inlet_flow[0]-expected_inlet
            ) > 1.0e-12 ||
            std::abs(
                tangent.outlet_pressure[0]-expected_outlet
            ) > 1.0e-9)
            throw std::runtime_error(
                "Initial tangent mapping incorrect");
    }
}

} // namespace

int main()
{
    try {
        for (int n : {2, 10, 100}) {
            for (int steps : {0, 1, 12, 80}) {
                for (bool include_initial : {false, true})
                    run_case(n, steps, include_initial);
            }
        }

        std::cout
            << "PASS tangent-linear propagation\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
