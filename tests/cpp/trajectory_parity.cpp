#include "state_finder/trajectory.hpp"

#include <algorithm>
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

state_finder::ForwardConfiguration configuration(int steps)
{
    const double diameter = 1.0;
    const double area = std::acos(-1.0) / 4.0;

    return {
        100,
        steps,
        pipe_sim::TransientParameters{
            1000.0,
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

state_finder::InitialState initial_state(
    const state_finder::ForwardConfiguration& config)
{
    const auto& par = config.parameters;

    state_finder::InitialState state{
        std::vector<double>(config.n + 1),
        std::vector<double>(config.n, 100.0),
        100.0
    };

    const double reynolds =
        100.0 * par.diameter /
        (par.area * par.viscosity);

    const double friction_arg =
        par.roughness / (3.7 * par.diameter) +
        5.74 / std::pow(reynolds, 0.9);

    const double friction =
        0.25 / std::pow(std::log10(friction_arg), 2.0);

    const double coefficient =
        friction * par.z * par.gas_constant *
        par.temperature * 100.0 * 100.0 /
        (par.diameter * par.area * par.area);

    for (int i = 0; i <= config.n; ++i) {
        state.pressure[i] = std::sqrt(
            config.demand.inlet_pressure *
                config.demand.inlet_pressure -
            coefficient * i * par.dx);
    }

    return state;
}

void check_exact(
    const std::vector<double>& actual,
    const std::vector<double>& expected,
    const char* name)
{
    require(actual.size() == expected.size(),
            "History dimensions differ");

    if (actual != expected) {
        std::cerr << "Mismatch in " << name << '\n';
        throw std::runtime_error("Trajectory parity failure");
    }
}

void run_case(int steps)
{
    using namespace state_finder;

    const auto config = configuration(steps);
    const auto truth = initial_state(config);

    const auto reference = evaluate_forward(config, truth);
    const auto trajectory = record_trajectory(config, truth);

    require(reference.info == 0, "Reference integration failed");
    require(trajectory.info == 0, "Trajectory integration failed");
    require(trajectory.completed_steps == steps,
            "Incomplete trajectory");

    check_exact(trajectory.time, reference.time, "time");
    check_exact(trajectory.inlet_flow,
                reference.inlet_flow, "inlet flow");
    check_exact(trajectory.outlet_flow,
                reference.outlet_flow, "outlet flow");
    check_exact(trajectory.outlet_pressure,
                reference.outlet_pressure, "outlet pressure");
    check_exact(trajectory.linepack,
                reference.linepack, "linepack");

    require(
        trajectory.max_mass_defect ==
            reference.max_mass_defect,
        "Mass conservation diagnostic mismatch");

    const auto initial = trajectory.state(0);
    const auto final = trajectory.state(steps);

    require(
        trajectory.states.size() ==
            static_cast<std::size_t>(steps + 1) *
            static_cast<std::size_t>(2*config.n + 2),
        "Incorrect trajectory storage size");

    require(initial[0] == truth.pressure[0],
            "Initial inlet pressure mismatch");
    require(initial[1] == truth.inlet_flow,
            "Initial inlet flow mismatch");

    for (int i = 0; i < config.n; ++i) {
        require(initial[2*i + 2] == truth.flow[i],
                "Initial interior flow mismatch");
        require(initial[2*i + 3] == truth.pressure[i + 1],
                "Initial pressure profile mismatch");
    }

    require(final[1] == reference.final_inlet_flow,
            "Final inlet flow mismatch");

    for (int i = 0; i < config.n; ++i) {
        require(final[2*i + 2] == reference.final_flow[i],
                "Final interior flow mismatch");
        require(final[2*i + 3] ==
                    reference.final_pressure[i + 1],
                "Final pressure profile mismatch");
    }

    // The recorder must not modify the candidate initial state.
    const auto truth_again = initial_state(config);

    require(truth.pressure == truth_again.pressure,
            "Candidate pressure modified");
    require(truth.flow == truth_again.flow,
            "Candidate flow modified");
    require(truth.inlet_flow == truth_again.inlet_flow,
            "Candidate inlet flow modified");

    std::cout
        << "PASS trajectory parity"
        << " cells=" << config.n
        << " steps=" << steps
        << " state_width=" << trajectory.state_size()
        << " stored_doubles=" << trajectory.states.size()
        << '\n';
}

} // namespace

int main()
{
    try {
        run_case(80);
        run_case(8000);
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
