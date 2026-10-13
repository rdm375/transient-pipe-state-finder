#include "state_finder/boundary_history_integrator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace state_finder;
using namespace pipe_sim;

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
        constexpr int n = 10;
        constexpr int steps = 8;

        TransientParameters par{
            1000.0, 15.0, 0.65,
            0.5, 0.19634954084936207,
            300.0, 0.9, 518.3,
            1.1e-5, 1.0e-5
        };

        NewtonOptions opt{
            1.0e-9, 1.0e-12, 30, false
        };

        AdjustableBoundaryHistory boundary;

        boundary.inlet_pressure.assign(
            steps + 1, 8.0e6
        );

        boundary.outlet_flow.assign(
            steps + 1, 100.0
        );

        std::vector<double> pressure(n+1, 8.0e6);
        std::vector<double> flow(n, 100.0);

        double inlet_flow = 100.0;

        const auto result = integrate_boundary_history(
            n,
            pressure,
            flow,
            inlet_flow,
            boundary,
            steps,
            par,
            opt
        );

        require(result.integration.info == 0,
                "Boundary-history integration failed");

        require(result.integration.completed_steps == steps,
                "Incorrect completed-step count");

        require(
            result.predicted.inlet_pressure.size() == steps+1,
            "Incorrect inlet-pressure history size"
        );

        require(
            result.predicted.inlet_flow.size() == steps+1,
            "Incorrect inlet-flow history size"
        );

        require(
            result.predicted.outlet_pressure.size() == steps+1,
            "Incorrect outlet-pressure history size"
        );

        require(
            result.predicted.outlet_flow.size() == steps+1,
            "Incorrect outlet-flow history size"
        );

        require(
            std::isfinite(result.integration.max_mass_defect),
            "Nonfinite mass-conservation defect"
        );

        std::cout
            << "PASS boundary-history integrator\n"
            << "cells=" << n << '\n'
            << "steps=" << steps << '\n'
            << "max_mass_defect="
            << result.integration.max_mass_defect << '\n';

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
