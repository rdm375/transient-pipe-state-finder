#include "state_finder/joint_inverse_objective.hpp"

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

bool close(double a, double b, double tol = 1e-9)
{
    return std::abs(a-b) <=
        tol * std::max({1.0, std::abs(a), std::abs(b)});
}

} // namespace

int main()
{
    try {
        constexpr int n = 10;
        constexpr int steps = 8;

        const double pi = std::acos(-1.0);

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                1000.0,
                15.0,
                0.65,
                1.0,
                pi/4.0,
                288.15,
                0.9,
                500.0,
                1.1e-5,
                4.5e-5
            },
            pipe_sim::DemandRamp{
                8.0e6, 100.0, 100.0, 1.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9, 1.0e-12, 30, false
            }
        };

        JointInverseVariables truth;

        truth.initial.pressure.assign(n+1, 8.0e6);
        truth.initial.flow.assign(n, 100.0);
        truth.initial.inlet_flow = 100.0;

        truth.boundary.inlet_pressure.assign(
            steps+1, 8.0e6
        );

        truth.boundary.outlet_flow.assign(
            steps+1, 100.0
        );

        // Generate physically consistent synthetic data.
        auto pressure = truth.initial.pressure;
        auto flow = truth.initial.flow;
        double inlet_flow = truth.initial.inlet_flow;

        const auto reference = integrate_boundary_history(
            n,
            pressure,
            flow,
            inlet_flow,
            truth.boundary,
            steps,
            config.parameters,
            config.newton
        );

        require(reference.integration.info == 0,
                "Reference integration failed");

        BoundaryObservationRecord observations;

        observations.include_initial = true;
        observations.measured = reference.predicted;

        observations.scales.inlet_pressure = 1.0e4;
        observations.scales.inlet_flow = 1.0;
        observations.scales.outlet_pressure = 1.0e4;
        observations.scales.outlet_flow = 1.0;

        const auto exact = evaluate_joint_inverse_objective(
            config, truth, observations
        );

        require(close(exact.measurement.total, 0.0),
                "Exact synthetic objective is nonzero");

        // Independently perturb each adjustable input.
        auto candidate = truth;

        candidate.boundary.inlet_pressure[3] += 1000.0;

        const auto pressure_test =
            evaluate_joint_inverse_objective(
                config, candidate, observations
            );

        require(pressure_test.measurement.inlet_pressure > 0.0,
                "Inlet-pressure residual is inactive");

        candidate = truth;
        candidate.boundary.outlet_flow[3] += 0.5;

        const auto flow_test =
            evaluate_joint_inverse_objective(
                config, candidate, observations
            );

        require(flow_test.measurement.outlet_flow > 0.0,
                "Outlet-flow residual is inactive");

        // Perturb an initial-state variable and verify
        // that the predicted observation channels respond.
        candidate = truth;
        candidate.initial.inlet_flow += 0.5;

        const auto state_test =
            evaluate_joint_inverse_objective(
                config, candidate, observations
            );

        require(state_test.measurement.inlet_flow > 0.0,
                "Inlet-flow observation is inactive");

        candidate = truth;
        candidate.initial.pressure[n] += 1000.0;

        const auto outlet_pressure_test =
            evaluate_joint_inverse_objective(
                config, candidate, observations
            );

        require(
            outlet_pressure_test.measurement.outlet_pressure > 0.0,
            "Outlet-pressure observation is inactive"
        );

        // Verify compatibility: the duplicated initial
        // inlet-pressure entry is not an independent DOF.
        candidate = truth;
        candidate.initial.pressure[0] += 5000.0;

        const auto compatibility_test =
            evaluate_joint_inverse_objective(
                config, candidate, observations
            );

        require(close(compatibility_test.measurement.total, 0.0),
                "Initial inlet-pressure compatibility failed");

        std::cout
            << "exact_objective=" << exact.measurement.total << '\n'
            << "perturbed_inlet_pressure_objective="
            << pressure_test.measurement.total << '\n'
            << "perturbed_outlet_flow_objective="
            << flow_test.measurement.total << '\n'
            << "perturbed_initial_inlet_flow_objective="
            << state_test.measurement.total << '\n'
            << "perturbed_initial_outlet_pressure_objective="
            << outlet_pressure_test.measurement.total << '\n'
            << "compatibility_objective="
            << compatibility_test.measurement.total << '\n'
            << "PASS joint four-measurement inverse objective\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
