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

double relative_error(double a, double b)
{
    return std::abs(a-b) /
        std::max({1.0, std::abs(a), std::abs(b)});
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

        const auto dimension =
            joint_variable_dimension(n, steps);

        std::vector<double> x(dimension);
        std::vector<double> u(dimension);
        std::vector<double> v(dimension);

        for (std::size_t i = 0; i < dimension; ++i) {
            x[i] = 0.02*std::sin(0.17*(i+1));
            u[i] = std::sin(0.31*(i+1));
            v[i] = std::cos(0.23*(i+1));
        }

        const auto variables =
            unpack_joint_variables(x, baseline);

        const auto trajectory =
            record_joint_trajectory(config, variables);

        for (bool include_initial : {true, false}) {
            BoundaryObservationRecord observations;
            observations.include_initial = include_initial;
            observations.measured = reference.predicted;

            const std::size_t time_count =
                include_initial ? steps+1 : steps;

            TemporalMeasurementCovariance covariance(
                time_count, 15.0, 120.0);

            const auto tangent_u =
                evaluate_joint_tangent_linear(
                    config, trajectory, u);

            const auto tangent_v =
                evaluate_joint_tangent_linear(
                    config, trajectory, v);

            const auto Su =
                normalized_joint_tangent_residuals(
                    tangent_u, observations);

            const auto Sv =
                normalized_joint_tangent_residuals(
                    tangent_v, observations);

            std::vector<double> w(Sv.size());

            for (std::size_t i = 0; i < w.size(); ++i)
                w[i] =
                    std::cos(0.23*(i+1)) -
                    0.3*std::sin(0.41*(i+1));

            const auto Cinv_w =
                covariance.apply_inverse(w);

            const auto adjoint =
                apply_joint_observation_adjoint(
                    config,
                    trajectory,
                    Cinv_w,
                    observations);

            const double transpose_error =
                relative_error(
                    dot(Sv, Cinv_w),
                    dot(v, adjoint.gradient));

            const auto H =
                make_correlated_joint_gauss_newton_operator(
                    config,
                    observations,
                    baseline,
                    x,
                    damping,
                    covariance);

            const auto Hu = H(u);
            const auto Hv = H(v);

            const double symmetry_error =
                relative_error(
                    dot(u, Hv),
                    dot(v, Hu));

            const double curvature =
                dot(v, Hv);

            const double expected_curvature =
                dot(Sv, covariance.apply_inverse(Sv))
                + damping*dot(v, v);

            const double curvature_error =
                relative_error(
                    curvature,
                    expected_curvature);

            TemporalMeasurementCovariance independent(
                time_count, 15.0, 0.0);

            const auto H_independent =
                make_correlated_joint_gauss_newton_operator(
                    config,
                    observations,
                    baseline,
                    x,
                    damping,
                    independent);

            const double independent_curvature =
                dot(v, H_independent(v));

            const double independent_expected =
                4.0*dot(Sv, Sv)
                + damping*dot(v, v);

            const double independent_error =
                relative_error(
                    independent_curvature,
                    independent_expected);

            std::cout
                << "include_initial=" << include_initial
                << " transpose_error=" << transpose_error
                << " symmetry_error=" << symmetry_error
                << " curvature_error=" << curvature_error
                << " independent_error=" << independent_error
                << " curvature=" << curvature
                << '\n';

            require(
                transpose_error < 1.0e-10,
                "Weighted adjoint identity failed");

            require(
                symmetry_error < 1.0e-10,
                "Correlated operator symmetry failed");

            require(
                curvature_error < 1.0e-10,
                "Correlated operator curvature mismatch");

            require(
                independent_error < 1.0e-10,
                "Independent-noise limit failed");

            require(
                curvature > 0.0,
                "Correlated operator is not positive definite");
        }

        std::cout
            << "PASS M27 correlated Gauss-Newton operator\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
