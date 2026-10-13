#include "state_finder/joint_adjoint.hpp"
#include "state_finder/joint_tangent_linear.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace state_finder;

namespace {

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Dot-product size mismatch");

    return std::inner_product(
        a.begin(), a.end(), b.begin(), 0.0);
}

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

}

int main()
{
    try {
        constexpr int n = 6;
        constexpr int steps = 5;

        const double pi = std::acos(-1.0);

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                1000.0, 15.0, 0.65,
                1.0, pi/4.0,
                288.15, 0.9, 500.0,
                1.1e-5, 4.5e-5
            },
            pipe_sim::DemandRamp{
                8.0e6, 100.0, 100.0, 1.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9, 1.0e-12, 30, false
            }
        };

        JointInverseVariables variables;

        variables.initial.pressure.assign(
            n+1, 8.0e6);

        variables.initial.flow.assign(
            n, 100.0);

        variables.initial.inlet_flow = 100.0;

        variables.boundary.inlet_pressure.assign(
            steps+1, 8.0e6);

        variables.boundary.outlet_flow.assign(
            steps+1, 100.0);

        // Nontrivial reference trajectory.
        variables.initial.pressure[2] += 1200.0;
        variables.initial.flow[3] -= 0.15;
        variables.initial.inlet_flow += 0.2;

        variables.boundary.inlet_pressure[0] += 500.0;
        variables.boundary.inlet_pressure[2] -= 300.0;
        variables.boundary.outlet_flow[3] += 0.2;

        const auto trajectory =
            record_joint_trajectory(config, variables);

        BoundaryObservationRecord observations;
        observations.measured = trajectory.predicted;

        const auto dimension =
            joint_variable_dimension(n, steps);

        std::vector<double> direction(dimension);

        for (std::size_t i = 0; i < dimension; ++i)
            direction[i] =
                std::sin(0.31*(i+1)) +
                0.2*std::cos(0.17*(i+1));

        double worst_error = 0.0;
        double worst_backward_error = 0.0;

        for (bool include_initial : {true, false}) {
            observations.include_initial = include_initial;

            const auto tangent =
                evaluate_joint_tangent_linear(
                    config, trajectory, direction);

            const auto Sv =
                normalized_joint_tangent_residuals(
                    tangent, observations);

            std::vector<double> forcing(Sv.size());

            for (std::size_t i = 0; i < forcing.size(); ++i)
                forcing[i] =
                    std::cos(0.23*(i+1)) -
                    0.3*std::sin(0.41*(i+1));

            const auto adjoint =
                apply_joint_observation_adjoint(
                    config,
                    trajectory,
                    forcing,
                    observations);

            const double lhs = dot(Sv, forcing);
            const double rhs =
                dot(direction, adjoint.gradient);

            const double error =
                std::abs(lhs-rhs) /
                std::max({
                    1.0,
                    std::abs(lhs),
                    std::abs(rhs)
                });

            worst_error = std::max(
                worst_error, error);

            worst_backward_error = std::max(
                worst_backward_error,
                adjoint.max_linear_backward_error);

            std::cout
                << "include_initial=" << include_initial
                << " lhs=" << lhs
                << " rhs=" << rhs
                << " relative_error=" << error
                << '\n';

            require(
                error < 1.0e-10,
                "Joint tangent/adjoint transpose identity failed");
        }

        std::cout
            << "joint_dimension=" << dimension << '\n'
            << "worst_transpose_relative_error="
            << worst_error << '\n'
            << "max_linear_backward_error="
            << worst_backward_error << '\n'
            << "PASS joint observation adjoint\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
