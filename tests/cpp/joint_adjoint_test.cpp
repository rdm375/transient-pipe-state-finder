#include "state_finder/joint_adjoint.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace state_finder;

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    return std::inner_product(
        a.begin(), a.end(), b.begin(), 0.0);
}

} // namespace

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

        JointInverseVariables truth;

        truth.initial.pressure.assign(n+1, 8.0e6);
        truth.initial.flow.assign(n, 100.0);
        truth.initial.inlet_flow = 100.0;

        truth.boundary.inlet_pressure.assign(
            steps+1, 8.0e6);

        truth.boundary.outlet_flow.assign(
            steps+1, 100.0);

        auto pressure = truth.initial.pressure;
        auto flow = truth.initial.flow;
        double inlet = truth.initial.inlet_flow;

        const auto reference = integrate_boundary_history(
            n, pressure, flow, inlet,
            truth.boundary, steps,
            config.parameters, config.newton);

        require(reference.integration.info == 0,
                "Reference trajectory failed");

        BoundaryObservationRecord observations;

        observations.include_initial = true;
        observations.measured = reference.predicted;

        JointInverseVariables baseline = truth;

        baseline.initial.pressure[2] += 1500.0;
        baseline.initial.pressure[4] -= 800.0;

        baseline.initial.flow[1] += 0.25;
        baseline.initial.flow[4] -= 0.15;

        baseline.initial.inlet_flow += 0.3;

        baseline.boundary.inlet_pressure[0] += 500.0;
        baseline.boundary.inlet_pressure[2] -= 700.0;
        baseline.boundary.inlet_pressure[4] += 400.0;

        baseline.boundary.outlet_flow[0] += 0.15;
        baseline.boundary.outlet_flow[3] -= 0.25;

        baseline.initial.pressure[0] =
            baseline.boundary.inlet_pressure[0];

        JointVariableScales scales;

        JointRegularization reg;

        // Measurement-only comparison.
        reg.initial_weight = 0.0;
        reg.boundary_weight = 0.0;
        reg.boundary_smoothness = 0.0;

        JointScaledProblem problem(
            config, observations, baseline, scales, reg);

        const std::size_t dimension =
            joint_variable_dimension(n, steps);

        std::vector<double> x(dimension, 0.0);

        const auto candidate =
            unpack_joint_variables(x, baseline, scales);

        const auto adjoint = evaluate_joint_adjoint(
            config, candidate, observations, scales);

        const auto fd =
            problem.finite_difference_gradient(x, 1.0e-4);

        require(adjoint.gradient.size() == dimension,
                "Adjoint dimension mismatch");

        double max_absolute = 0.0;
        double max_relative = 0.0;

        for (std::size_t i = 0; i < dimension; ++i) {
            const double a = adjoint.gradient[i];
            const double b = fd[i];

            max_absolute = std::max(
                max_absolute, std::abs(a-b));

            const double relative =
                std::abs(a-b) /
                std::max({1.0, std::abs(a), std::abs(b)});

            max_relative = std::max(
                max_relative, relative);
        }

        std::vector<double> direction(dimension);

        for (std::size_t i = 0; i < dimension; ++i)
            direction[i] =
                std::sin(0.37*(i+1));

        const double norm = std::sqrt(
            dot(direction, direction));

        for (double& value : direction)
            value /= norm;

        constexpr double h = 1.0e-4;

        auto xp = x;
        auto xm = x;

        for (std::size_t i = 0; i < dimension; ++i) {
            xp[i] += h*direction[i];
            xm[i] -= h*direction[i];
        }

        const double directional_fd =
            (problem.evaluate(xp).total -
             problem.evaluate(xm).total) /
            (2.0*h);

        const double directional_adjoint =
            dot(adjoint.gradient, direction);

        const double directional_error =
            std::abs(
                directional_fd-directional_adjoint) /
            std::max({
                1.0,
                std::abs(directional_fd),
                std::abs(directional_adjoint)
            });

        std::cout
            << "joint_dimension=" << dimension << '\n'
            << "measurement_objective="
            << adjoint.measurement_objective << '\n'
            << "max_absolute_gradient_error="
            << max_absolute << '\n'
            << "max_relative_gradient_error="
            << max_relative << '\n'
            << "directional_fd="
            << directional_fd << '\n'
            << "directional_adjoint="
            << directional_adjoint << '\n'
            << "directional_relative_error="
            << directional_error << '\n'
            << "max_linear_backward_error="
            << adjoint.max_linear_backward_error << '\n';

        require(max_relative < 1.0e-3,
                "Joint adjoint component gradient mismatch");

        require(directional_error < 1.0e-3,
                "Joint adjoint directional gradient mismatch");

        std::cout
            << "PASS joint discrete adjoint gradient\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
