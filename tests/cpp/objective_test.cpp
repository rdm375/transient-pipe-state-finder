#include "state_finder/objective.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

state_finder::ForwardConfiguration make_configuration()
{
    using namespace pipe_sim;

    constexpr int n = 10;

    const double diameter = 1.0;
    const double area = std::acos(-1.0) / 4.0;

    return {
        n,
        20,
        TransientParameters{
            100000.0 / n,
            15.0,
            0.65,
            diameter,
            area,
            288.15,
            0.9,
            500.0,
            1.1e-5,
            4.5e-5
        },
        DemandRamp{
            8.0e6,
            100.0,
            105.0,
            600.0
        },
        NewtonOptions{
            1.0e-9,
            1.0e-12,
            30,
            false
        }
    };
}

state_finder::InitialState make_initial_state(
    const state_finder::ForwardConfiguration& config)
{
    const auto& par = config.parameters;
    const auto& demand = config.demand;

    const int n = config.n;
    const double inlet_flow = 100.0;

    state_finder::InitialState state{
        std::vector<double>(n + 1),
        std::vector<double>(n, inlet_flow),
        inlet_flow
    };

    const double reynolds =
        inlet_flow * par.diameter /
        (par.area * par.viscosity);

    const double friction_arg =
        par.roughness / (3.7 * par.diameter) +
        5.74 / std::pow(reynolds, 0.9);

    const double friction =
        0.25 / std::pow(std::log10(friction_arg), 2.0);

    const double coefficient =
        friction * par.z * par.gas_constant *
        par.temperature * inlet_flow * inlet_flow /
        (par.diameter * par.area * par.area);

    for (int i = 0; i <= n; ++i) {
        state.pressure[i] = std::sqrt(
            demand.inlet_pressure * demand.inlet_pressure -
            coefficient * i * par.dx);
    }

    return state;
}

void test_objective()
{
    using namespace state_finder;

    const auto config = make_configuration();
    const auto truth = make_initial_state(config);

    const auto synthetic = evaluate_forward(config, truth);

    require(synthetic.info == 0, "Synthetic forward run failed");
    require(synthetic.completed_steps == config.steps,
            "Synthetic run incomplete");

    Observations observations;
    observations.inlet_flow = synthetic.inlet_flow;
    observations.outlet_pressure = synthetic.outlet_pressure;
    observations.sigma_flow = 1.0;
    observations.sigma_pressure = 1.0e4;

    const auto exact =
        evaluate_objective(config, truth, observations);

    require(exact.value == 0.0,
            "True initial state must have zero misfit");

    const auto repeated =
        evaluate_objective(config, truth, observations);

    require(repeated.value == exact.value,
            "Objective is not deterministic");

    // Interior pressure perturbation.
    auto perturbed_pressure = truth;
    perturbed_pressure.pressure[config.n / 2] += 1.0e4;

    const auto pressure_result =
        evaluate_objective(
            config, perturbed_pressure, observations);

    require(pressure_result.value > 0.0,
            "Pressure perturbation produced zero misfit");

    // Interior mass-flow perturbation.
    auto perturbed_flow = truth;
    perturbed_flow.flow[config.n / 2] += 1.0;

    const auto flow_result =
        evaluate_objective(
            config, perturbed_flow, observations);

    require(flow_result.value > 0.0,
            "Flow perturbation produced zero misfit");

    // Verify the original state was not modified.
    const auto truth_again = make_initial_state(config);

    require(truth.pressure == truth_again.pressure,
            "Initial pressure state was modified");

    require(truth.flow == truth_again.flow,
            "Initial flow state was modified");

    require(truth.inlet_flow == truth_again.inlet_flow,
            "Initial inlet flow was modified");

    std::cout
        << "PASS objective zero misfit: " << exact.value << '\n'
        << "PASS pressure sensitivity: "
        << pressure_result.value << '\n'
        << "PASS flow sensitivity: "
        << flow_result.value << '\n'
        << "PASS deterministic evaluation\n"
        << "PASS initial-state preservation\n";
}

} // namespace

int main()
{
    try {
        test_objective();
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
