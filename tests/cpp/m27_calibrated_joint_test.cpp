#include "state_finder/calibrated_joint_gauss_newton.hpp"

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
    require(a.size() == b.size(), "Dot dimension mismatch");

    return std::inner_product(
        a.begin(), a.end(), b.begin(), 0.0);
}

double relative_error(double a, double b)
{
    return std::abs(a-b) /
        std::max({1.0, std::abs(a), std::abs(b)});
}

} // namespace

int main()
{
    try {
        constexpr int n = 6;
        constexpr int steps = 5;
        constexpr double damping = 1.0e-3;

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

        const auto state_dimension =
            joint_variable_dimension(n, steps);

        const auto dimension = state_dimension+2;

        std::vector<double> z(dimension);
        std::vector<double> u(dimension);
        std::vector<double> v(dimension);

        for (std::size_t i = 0; i < dimension; ++i) {
            z[i] = 0.02*std::sin(0.17*(i+1));
            u[i] = std::sin(0.31*(i+1));
            v[i] = std::cos(0.23*(i+1));
        }

        // Explicitly exercise the calibration coordinates.
        z[state_dimension] = 0.7;
        z[state_dimension+1] = -0.4;

        u[state_dimension] = 0.6;
        u[state_dimension+1] = -0.3;

        v[state_dimension] = -0.5;
        v[state_dimension+1] = 0.8;

        for (bool include_initial : {true, false}) {
            BoundaryObservationRecord observations;
            observations.include_initial = include_initial;
            observations.measured = reference.predicted;

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

            for (double rho : {0.0, 0.5}) {
                const PressureCalibrationPrior prior{
                    2.0, rho
                };

                const auto evaluate =
                    [&](const std::vector<double>& point,
                        bool gradient)
                    {
                        return evaluate_calibrated_joint_objective(
                            config,
                            observations,
                            baseline,
                            point,
                            covariance,
                            prior,
                            {},
                            gradient);
                    };

                const auto analytical = evaluate(z, true);

                require(
                    analytical.valid,
                    "Invalid augmented objective");

                require(
                    analytical.gradient.size() == dimension,
                    "Augmented gradient dimension mismatch");

                const double directional =
                    dot(analytical.gradient, u);

                double best_error = 1.0e100;

                for (double h : {
                    1.0e-2,
                    1.0e-3,
                    1.0e-4
                }) {
                    auto plus = z;
                    auto minus = z;

                    for (std::size_t i = 0;
                         i < dimension;
                         ++i)
                    {
                        plus[i] += h*u[i];
                        minus[i] -= h*u[i];
                    }

                    const auto fp = evaluate(plus, false);
                    const auto fm = evaluate(minus, false);

                    require(
                        fp.valid && fm.valid,
                        "Invalid augmented FD evaluation");

                    const double fd =
                        (fp.objective-fm.objective)/(2.0*h);

                    best_error = std::min(
                        best_error,
                        relative_error(fd, directional));
                }

                require(
                    best_error < 1.0e-5,
                    "Augmented gradient FD mismatch");

                const auto H =
                    make_calibrated_joint_gauss_newton_operator(
                        config,
                        observations,
                        baseline,
                        z,
                        damping,
                        covariance,
                        prior);

                const auto Hu = H(u);
                const auto Hv = H(v);

                const double symmetry_error =
                    relative_error(
                        dot(u, Hv),
                        dot(v, Hu));

                require(
                    symmetry_error < 1.0e-10,
                    "Augmented operator symmetry mismatch");

                const std::vector<double> x(
                    z.begin(),
                    z.begin()+state_dimension);

                const auto variables =
                    unpack_joint_variables(x, baseline);

                const auto trajectory =
                    record_joint_trajectory(config, variables);

                const std::vector<double> vx(
                    v.begin(),
                    v.begin()+state_dimension);

                const auto tangent =
                    evaluate_joint_tangent_linear(
                        config, trajectory, vx);

                auto response =
                    normalized_joint_tangent_residuals(
                        tangent, observations);

                const std::span<const double> vb(
                    v.data()+state_dimension, 2);

                const auto bias_response =
                    calibration_response_kpa(
                        vb, response.size());

                for (std::size_t i = 0;
                     i < response.size();
                     ++i)
                {
                    response[i] += bias_response[i];
                }

                const double expected_curvature =
                    covariance.quadratic_form(response)
                    +prior.quadratic_form(vb)
                    +damping*dot(v, v);

                const double actual_curvature =
                    dot(v, Hv);

                const double curvature_error =
                    relative_error(
                        actual_curvature,
                        expected_curvature);

                require(
                    curvature_error < 1.0e-10,
                    "Augmented operator curvature mismatch");

                require(
                    actual_curvature > 0.0,
                    "Augmented operator nonpositive curvature");

                // Zero bias must reproduce the original
                // measurement objective and state gradient.
                auto zero_bias = z;

                zero_bias[state_dimension] = 0.0;
                zero_bias[state_dimension+1] = 0.0;

                const std::vector<double> zero_x(
                    zero_bias.begin(),
                    zero_bias.begin()+state_dimension);

                const auto original =
                    evaluate_correlated_joint_objective(
                        config,
                        observations,
                        baseline,
                        zero_x,
                        covariance,
                        {},
                        true);

                const auto augmented =
                    evaluate(zero_bias, true);

                require(
                    original.valid && augmented.valid,
                    "Zero-bias objective evaluation failed");

                require(
                    relative_error(
                        original.objective,
                        augmented.objective) < 1.0e-12,
                    "Zero-bias objective mismatch");

                for (std::size_t i = 0;
                     i < state_dimension;
                     ++i)
                {
                    require(
                        relative_error(
                            original.gradient[i],
                            augmented.gradient[i]) < 1.0e-12,
                        "Zero-bias state gradient mismatch");
                }

                std::cout
                    << "include_initial=" << include_initial
                    << " rho=" << rho
                    << " gradient_error=" << best_error
                    << " symmetry_error=" << symmetry_error
                    << " curvature_error=" << curvature_error
                    << '\n';
            }
        }

        std::cout
            << "PASS: M27-B2 augmented objective and operator\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
