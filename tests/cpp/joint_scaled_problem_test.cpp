#include "state_finder/joint_scaled_problem.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
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
    require(a.size() == b.size(), "Dot-product size mismatch");

    double sum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        sum += a[i]*b[i];

    return sum;
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
                1.0, pi/4.0, 288.15,
                0.9, 500.0, 1.1e-5, 4.5e-5
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
        double inlet_flow = truth.initial.inlet_flow;

        const auto reference = integrate_boundary_history(
            n, pressure, flow, inlet_flow,
            truth.boundary, steps,
            config.parameters, config.newton);

        require(reference.integration.info == 0,
                "Reference integration failed");

        BoundaryObservationRecord observations;
        observations.include_initial = true;
        observations.measured = reference.predicted;

        // The baseline is intentionally imperfect.
        auto baseline = truth;

        baseline.initial.pressure[3] += 2000.0;
        baseline.initial.flow[2] += 0.3;
        baseline.initial.inlet_flow += 0.2;

        baseline.boundary.inlet_pressure[2] += 500.0;
        baseline.boundary.outlet_flow[3] += 0.2;

        JointScaledProblem problem(
            config, observations, baseline);

        const auto dimension =
            joint_variable_dimension(n, steps);

        require(dimension == 25,
                "Incorrect joint variable dimension");

        const std::vector<double> zero(dimension, 0.0);

        const auto initial = problem.evaluate(zero);

        require(initial.measurement > 0.0,
                "Initial measurement objective must be positive");

        // Verify packing/unpacking of the known truth.
        const auto truth_x =
            pack_joint_variables(truth, baseline);

        const auto recovered =
            unpack_joint_variables(truth_x, baseline);

        require(
            std::abs(
                recovered.initial.pressure[3] -
                truth.initial.pressure[3]) < 1e-8,
            "Initial-pressure packing mismatch");

        require(
            std::abs(
                recovered.boundary.outlet_flow[3] -
                truth.boundary.outlet_flow[3]) < 1e-12,
            "Boundary-flow packing mismatch");

        // Compare finite-difference gradient with an
        // independently evaluated directional derivative.
        const auto gradient =
            problem.finite_difference_gradient(zero);

        std::vector<double> direction(dimension);

        for (std::size_t i = 0; i < dimension; ++i)
            direction[i] = std::sin(0.31*(i+1));

        const double direction_norm =
            std::sqrt(dot(direction, direction));

        for (double& value : direction)
            value /= direction_norm;

        const double h = 1.0e-3;

        auto plus = zero;
        auto minus = zero;

        for (std::size_t i = 0; i < dimension; ++i) {
            plus[i] += h*direction[i];
            minus[i] -= h*direction[i];
        }

        const double directional_fd =
            (problem.evaluate(plus).total -
             problem.evaluate(minus).total)/(2.0*h);

        const double directional_gradient =
            dot(gradient, direction);

        const double error =
            std::abs(directional_fd - directional_gradient) /
            std::max({
                1.0,
                std::abs(directional_fd),
                std::abs(directional_gradient)
            });

        std::cout
            << "joint_dimension=" << dimension << '\n'
            << "initial_objective=" << initial.total << '\n'
            << "directional_fd=" << directional_fd << '\n'
            << "directional_gradient=" << directional_gradient << '\n'
            << "directional_relative_error=" << error << '\n';

        require(error < 1.0e-3,
                "Joint directional gradient verification failed");

        // Verify that a short nonlinear optimization
        // actually reduces the complete objective.
        OptimizationOptions options;
        options.max_iterations = 5;

        const auto callback =
            [&](const std::vector<double>& x,
                bool need_gradient)
            {
                return problem.evaluate_for_optimizer(
                    x, need_gradient);
            };

        const auto result = minimize_lbfgs(
            callback, zero, options);

        std::cout
            << "optimization_initial_objective="
            << result.initial_objective << '\n'
            << "optimization_final_objective="
            << result.final_objective << '\n'
            << "optimization_iterations="
            << result.iterations << '\n';

        require(
            result.final_objective < result.initial_objective,
            "Joint optimization failed to reduce objective");

        std::cout
            << "PASS scaled joint inverse problem\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
