#include "state_finder/joint_gauss_newton_operator.hpp"

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
    if (a.size() != b.size())
        throw std::runtime_error("Dot dimension mismatch");

    return std::inner_product(
        a.begin(), a.end(), b.begin(), 0.0);
}

double norm(const std::vector<double>& x)
{
    return std::sqrt(dot(x, x));
}

double relative_error(double a, double b)
{
    return std::abs(a-b) /
        std::max({1.0, std::abs(a), std::abs(b)});
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

        baseline.initial.pressure.assign(
            n+1, 8.0e6);

        baseline.initial.flow.assign(
            n, 100.0);

        baseline.initial.inlet_flow = 100.0;

        baseline.boundary.inlet_pressure.assign(
            steps+1, 8.0e6);

        baseline.boundary.outlet_flow.assign(
            steps+1, 100.0);

        BoundaryObservationRecord observations;
        observations.include_initial = true;

        const auto reference =
            record_joint_trajectory(config, baseline);

        observations.measured = reference.predicted;

        JointRegularization regularization;
        regularization.initial_weight = 1.0e-4;
        regularization.boundary_weight = 1.0e-2;
        regularization.boundary_smoothness = 1.0e-2;

        constexpr double damping = 1.0e-3;

        const auto dimension =
            joint_variable_dimension(n, steps);

        std::vector<double> x(dimension, 0.0);

        // Nontrivial linearization point.
        for (std::size_t i = 0; i < dimension; ++i)
            x[i] = 0.02*std::sin(0.17*(i+1));

        const auto A =
            make_joint_gauss_newton_operator(
                config,
                observations,
                baseline,
                x,
                damping,
                {},
                regularization);

        std::vector<double> u(dimension);
        std::vector<double> v(dimension);

        for (std::size_t i = 0; i < dimension; ++i) {
            u[i] =
                std::sin(0.31*(i+1));

            v[i] =
                std::cos(0.23*(i+1));
        }

        const auto Au = A(u);
        const auto Av = A(v);

        // 1. Symmetry.
        const double lhs = dot(u, Av);
        const double rhs = dot(v, Au);

        const double symmetry_error =
            relative_error(lhs, rhs);

        // 2. Positive definiteness.
        const double curvature = dot(v, Av);

        // 3. Explicit normal-product equivalence.
        const auto variables =
            unpack_joint_variables(x, baseline);

        const auto trajectory =
            record_joint_trajectory(config, variables);

        const auto tangent =
            evaluate_joint_tangent_linear(
                config, trajectory, v);

        const auto Sv =
            normalized_joint_tangent_residuals(
                tangent, observations);

        const auto reg =
            joint_regularization_gradient(
                n, steps, v, regularization);

        const double expected_curvature =
            dot(Sv, Sv) +
            dot(v, reg) +
            damping*dot(v, v);

        const double curvature_error =
            relative_error(
                curvature, expected_curvature);

        // 4. Solve a manufactured system with PCG.
        const auto b = A(v);

        CGOptions options;
        options.max_iterations = 1000;
        options.relative_tolerance = 1.0e-13;
        options.absolute_tolerance = 1.0e-14;

        const auto cg =
            solve_matrix_free_cg(A, b, options);

        const auto recovered = cg.solution;

        const auto Arecovered = A(recovered);

        std::vector<double> residual(dimension);
        std::vector<double> error(dimension);

        for (std::size_t i = 0; i < dimension; ++i) {
            residual[i] = Arecovered[i]-b[i];
            error[i] = recovered[i]-v[i];
        }

        const double residual_ratio =
            norm(residual)/std::max(1.0, norm(b));

        const double solution_error =
            norm(error)/std::max(1.0, norm(v));

        std::cout
            << "joint_dimension=" << dimension << '\n'
            << "symmetry_relative_error="
            << symmetry_error << '\n'
            << "curvature=" << curvature << '\n'
            << "curvature_relative_error="
            << curvature_error << '\n'
            << "pcg_converged=" << cg.converged << '\n'
            << "pcg_iterations=" << cg.iterations << '\n'
            << "pcg_residual_ratio="
            << residual_ratio << '\n'
            << "pcg_solution_relative_error="
            << solution_error << '\n';

        require(
            symmetry_error < 1.0e-11,
            "Joint GN symmetry failed");

        require(
            curvature > 0.0,
            "Joint GN positive definiteness failed");

        require(
            curvature_error < 1.0e-11,
            "Joint GN normal-product identity failed");

        require(
            cg.converged,
            "Joint GN PCG failed to converge");

        require(
            residual_ratio < 1.0e-8,
            "Joint GN PCG residual too large");

        require(
            solution_error < 1.0e-6,
            "Joint GN PCG solution error too large");

        std::cout
            << "PASS joint matrix-free Gauss-Newton operator\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
