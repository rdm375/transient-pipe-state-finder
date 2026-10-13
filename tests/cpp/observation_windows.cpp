#include "state_finder/forward_model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using namespace state_finder;

InitialState make_initial_state(
    const ForwardConfiguration& config)
{
    const int n = config.n;
    const auto& par = config.parameters;

    InitialState state{
        std::vector<double>(n+1),
        std::vector<double>(n, 100.0),
        100.0
    };

    const double mdot = 100.0;

    const double reynolds =
        mdot*par.diameter/(par.area*par.viscosity);

    const double argument =
        par.roughness/(3.7*par.diameter)
        + 5.74/std::pow(reynolds, 0.9);

    const double friction =
        0.25/std::pow(std::log10(argument), 2);

    const double coefficient =
        friction*par.z*par.gas_constant*
        par.temperature*mdot*mdot/
        (par.diameter*par.area*par.area);

    const double inlet_pressure =
        config.demand.inlet_pressure;

    for (int i = 0; i <= n; ++i) {
        const double squared =
            inlet_pressure*inlet_pressure
            - coefficient*i*par.dx;

        if (!(squared > 0.0))
            throw std::runtime_error(
                "Invalid reference pressure profile");

        state.pressure[i] = std::sqrt(squared);
    }

    return state;
}

void report_window(
    const ForwardResult& reference,
    const ForwardResult& candidate,
    int first,
    int last,
    double sigma_flow,
    double sigma_pressure)
{
    double objective = 0.0;
    double max_flow = 0.0;
    double max_pressure = 0.0;

    for (int k = first; k <= last; ++k) {
        const double df =
            candidate.inlet_flow[k] -
            reference.inlet_flow[k];

        const double dp =
            candidate.outlet_pressure[k] -
            reference.outlet_pressure[k];

        objective += 0.5*(
            std::pow(df/sigma_flow, 2)
            + std::pow(dp/sigma_pressure, 2));

        max_flow =
            std::max(max_flow, std::abs(df));

        max_pressure =
            std::max(max_pressure, std::abs(dp));
    }

    std::cout
        << first << ','
        << last << ','
        << objective << ','
        << max_flow << ','
        << max_pressure
        << '\n';
}

} // namespace

int main()
{
    try {
        constexpr int n = 100;
        constexpr int steps = 8000;

        const double area = std::acos(-1.0)/4.0;

        const ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                1000.0,
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

        const auto truth = make_initial_state(config);

        auto candidate = truth;

        for (int i = 1; i <= n; ++i)
            candidate.pressure[i] +=
                1000.0*std::sin(0.031*i);

        for (int i = 0; i < n; ++i)
            candidate.flow[i] +=
                0.5*std::cos(0.047*(i+1));

        candidate.inlet_flow += 0.7;

        const auto reference =
            evaluate_forward(config, truth);

        const auto perturbed =
            evaluate_forward(config, candidate);

        if (reference.info != 0 ||
            perturbed.info != 0 ||
            reference.completed_steps != steps ||
            perturbed.completed_steps != steps)
            throw std::runtime_error(
                "Forward integration did not complete");

        std::cout << std::setprecision(17);

        std::cout
            << "reference_completed_steps="
            << reference.completed_steps << '\n'
            << "candidate_completed_steps="
            << perturbed.completed_steps << '\n'
            << "reference_mass_defect="
            << reference.max_mass_defect << '\n'
            << "candidate_mass_defect="
            << perturbed.max_mass_defect << '\n';

        std::cout
            << "first,last,objective,"
            << "max_inlet_flow_difference,"
            << "max_outlet_pressure_difference\n";

        report_window(
            reference, perturbed,
            0, 80, 1.0, 1.0e4);

        report_window(
            reference, perturbed,
            81, 800, 1.0, 1.0e4);

        report_window(
            reference, perturbed,
            801, 8000, 1.0, 1.0e4);

        double max_final_pressure = 0.0;
        double max_final_flow = 0.0;

        for (int i = 0; i <= n; ++i)
            max_final_pressure = std::max(
                max_final_pressure,
                std::abs(
                    perturbed.final_pressure[i] -
                    reference.final_pressure[i]));

        for (int i = 0; i < n; ++i)
            max_final_flow = std::max(
                max_final_flow,
                std::abs(
                    perturbed.final_flow[i] -
                    reference.final_flow[i]));

        std::cout
            << "max_final_pressure_difference="
            << max_final_pressure << '\n'
            << "max_final_flow_difference="
            << max_final_flow << '\n'
            << "final_inlet_flow_difference="
            << std::abs(
                perturbed.final_inlet_flow -
                reference.final_inlet_flow)
            << '\n';

        std::cout
            << "PASS observation-window diagnostics\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
