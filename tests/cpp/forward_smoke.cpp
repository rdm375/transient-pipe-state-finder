#include "pipe_sim/integrate.hpp"
#include "pipe_sim/transient.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    constexpr int n = 10;
    constexpr int steps = 0;

    pipe_sim::TransientParameters par{
        100.0,       // dx
        1.0,         // dt
        0.65,        // theta
        0.5,         // diameter
        0.19634954084936207, // area
        288.15,      // temperature
        0.9,         // z
        518.3,       // gas constant
        1.1e-5,      // viscosity
        1.0e-5       // roughness
    };

    std::vector<double> pressure(n + 1, 5.0e6);
    std::vector<double> flow(n, 10.0);

    double inlet_flow = 10.0;

    pipe_sim::DemandRamp demand{
        5.0e6,
        10.0,
        10.0,
        0.0
    };

    pipe_sim::NewtonOptions options{
        1.0e-8,   // residual_tolerance
        1.0e-10,  // step_tolerance
        30,       // max_iterations
        false     // verbose
    };

    std::vector<double> time(steps + 1);
    std::vector<double> inlet_history(steps + 1);
    std::vector<double> outlet_history(steps + 1);
    std::vector<double> pressure_history(steps + 1);
    std::vector<double> linepack_history(steps + 1);
    std::vector<int> iterations(steps + 1);

    pipe_sim::IntegrationHistory history{
        time,
        inlet_history,
        outlet_history,
        pressure_history,
        linepack_history,
        iterations
    };

    pipe_sim::IntegrationWorkspace workspace(n);

    const auto result = pipe_sim::integrate_transient(
        n,
        pressure,
        flow,
        inlet_flow,
        demand,
        steps,
        par,
        options,
        history,
        workspace
    );

    if (result.info != 0) {
        std::cerr << "Forward integration failed: "
                  << result.info << '\n';
        return 1;
    }

    if (!std::isfinite(linepack_history[0]) ||
        linepack_history[0] <= 0.0) {
        std::cerr << "Invalid initial linepack\n";
        return 1;
    }

    std::cout << "Forward solver integration: PASS\n";
    std::cout << "Initial linepack: "
              << linepack_history[0] << " kg\n";
    std::cout << "Initial inlet flow: "
              << inlet_history[0] << " kg/s\n";
    std::cout << "Initial outlet pressure: "
              << pressure_history[0] << " Pa\n";

    return 0;
}
