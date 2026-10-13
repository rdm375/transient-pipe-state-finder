#include "state_finder/steady_state_initializer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace state_finder;

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

}

int main()
{
    try {
        constexpr int n = 20;
        constexpr double length = 50000.0;
        constexpr double diameter = 0.6;
        constexpr double mass_flow = 100.0;

        const double pi = std::acos(-1.0);
        const double area = pi*diameter*diameter/4.0;

        ForwardConfiguration config{
            n,
            60,
            pipe_sim::TransientParameters{
                length/n,
                10.0,
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
                8.0e6, mass_flow, mass_flow, 1.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9, 1.0e-12, 30, false
            }
        };

        const auto state =
            initialize_steady_state(config, mass_flow);

        require(
            state.pressure.size() == n+1,
            "Incorrect pressure dimension");

        require(
            state.flow.size() == n,
            "Incorrect flow dimension");

        require(
            state.pressure.front() == 8.0e6,
            "Incorrect inlet pressure");

        require(
            state.inlet_flow == mass_flow,
            "Incorrect inlet flow");

        for (int i = 0; i < n; ++i) {
            require(
                state.pressure[i] > state.pressure[i+1],
                "Pressure profile is not decreasing");

            require(
                state.flow[i] == mass_flow,
                "Mass flow is not constant");
        }

        require(
            std::all_of(
                state.pressure.begin(),
                state.pressure.end(),
                [](double p) {
                    return std::isfinite(p) && p > 0.0;
                }),
            "Nonphysical pressure");

        // Check the pressure-squared relation.
        const auto& par = config.parameters;

        const double reynolds =
            mass_flow*par.diameter /
            (par.area*par.viscosity);

        const double argument =
            par.roughness/(3.7*par.diameter) +
            5.74/std::pow(reynolds, 0.9);

        const double friction =
            0.25/std::pow(std::log10(argument), 2.0);

        const double coefficient =
            friction*par.z*par.gas_constant*
            par.temperature*mass_flow*mass_flow /
            (par.diameter*par.area*par.area);

        for (int i = 0; i <= n; ++i) {
            const double expected_squared =
                8.0e6*8.0e6 -
                coefficient*i*par.dx;

            const double actual_squared =
                state.pressure[i]*state.pressure[i];

            const double relative_error =
                std::abs(
                    actual_squared - expected_squared) /
                expected_squared;

            require(
                relative_error < 1.0e-14,
                "Pressure-squared relation mismatch");
        }

        std::cout
            << "outlet_pressure_Pa="
            << state.pressure.back() << '\n'
            << "PASS steady-state initializer\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';

        return 1;
    }
}
