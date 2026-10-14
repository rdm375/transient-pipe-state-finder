#include "state_finder/correlated_joint_gauss_newton.hpp"

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
        throw std::runtime_error("Dot dimension mismatch");

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

        JointInverseVariables baseline;

        baseline.initial.pressure.assign(n+1, 8.0e6);
        baseline.initial.flow.assign(n, 100.0);
        baseline.initial.inlet_flow = 100.0;

        baseline.boundary.inlet_pressure.assign(
            steps+1, 8.0e6);

        baseline.boundary.outlet_flow.assign(
            steps+1, 100.0);

        const auto reference =
            record_joint_trajectory(config, baseline);

        const auto dimension =
            joint_variable_dimension(n, steps);

        std::vector<double> x(dimension);
        std::vector<double> direction(dimension);

        for (std::size_t i = 0; i < dimension; ++i) {
            x[i] = 0.03*std::sin(0.17*(i+1));

            direction[i] =
                std::sin(0.31*(i+1)) +
                0.2*std::cos(0.19*(i+1));
        }

        for (bool include_initial : {true, false}) {
            BoundaryObservationRecord observations;
            observations.include_initial = include_initial;
            observations.measured = reference.predicted;

            // Introduce a nonzero observation residual.
            for (std::size_t k = 0;
                 k < observations.measured.inlet_pressure.size();
                 ++k)
            {
                observations.measured.inlet_pressure[k] +=
                    100.0*std::sin(0.4*(k+1));

                observations.measured.outlet_pressure[k] -=
                    75.0*std::cos(0.3*(k+1));

                observations.measured.inlet_flow[k] +=
                    0.01*std::sin(0.5*(k+1));

                observations.measured.outlet_flow[k] -=
                    0.01*std::cos(0.2*(k+1));
            }

            const std::size_t time_count =
                include_initial ? steps+1 : steps;

            TemporalMeasurementCovariance covariance(
                time_count, 15.0, 120.0);

            const auto evaluate =
                [&](const std::vector<double>& point,
                    bool gradient)
                {
                    return evaluate_correlated_joint_objective(
                        config,
                        observations,
                        baseline,
                        point,
                        covariance,
                        {},
                        gradient);
                };

            const auto analytical = evaluate(x, true);

            require(
                analytical.valid,
                "Invalid analytical objective evaluation");

            require(
                analytical.gradient.size() == dimension,
                "Analytical gradient dimension mismatch");

            const double directional =
                dot(analytical.gradient, direction);

            double best_error = 1.0e100;

            for (double h : {
                1.0e-2,
                1.0e-3,
                1.0e-4
            }) {
                auto plus = x;
                auto minus = x;

                for (std::size_t i = 0;
                     i < dimension;
                     ++i)
                {
                    plus[i] += h*direction[i];
                    minus[i] -= h*direction[i];
                }

                const auto fp = evaluate(plus, false);
                const auto fm = evaluate(minus, false);

                require(
                    fp.valid && fm.valid,
                    "Invalid finite-difference evaluation");

                const double finite_difference =
                    (fp.objective-fm.objective)/(2.0*h);

                const double error =
                    std::abs(
                        finite_difference-directional) /
                    std::max({
                        1.0,
                        std::abs(finite_difference),
                        std::abs(directional)
                    });

                best_error = std::min(best_error, error);

                std::cout
                    << "include_initial=" << include_initial
                    << " h=" << h
                    << " objective=" << analytical.objective
                    << " fd=" << finite_difference
                    << " adjoint=" << directional
                    << " relative_error=" << error
                    << '\n';
            }

            require(
                best_error < 1.0e-5,
                "Correlated objective gradient FD failed");
        }

        std::cout
            << "PASS M27 correlated nonlinear objective gradient\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
