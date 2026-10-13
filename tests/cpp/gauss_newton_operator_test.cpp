#include "state_finder/gauss_newton_operator.hpp"
#include "state_finder/matrix_free_cg.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    using namespace state_finder;

    constexpr int n = 100;
    constexpr int steps = 80;

    const double area = std::acos(-1.0)/4.0;

    const ForwardConfiguration config{
        n,
        steps,
        pipe_sim::TransientParameters{
            100000.0/n, 15.0, 0.65, 1.0,
            area, 288.15, 0.9, 500.0,
            1.1e-5, 4.5e-5
        },
        pipe_sim::DemandRamp{
            8.0e6, 100.0, 105.0, 600.0
        },
        pipe_sim::NewtonOptions{
            1.0e-9, 1.0e-12, 30, false
        }
    };

    InitialState state{
        std::vector<double>(n+1),
        std::vector<double>(n, 100.0),
        100.0
    };

    const auto& par = config.parameters;

    const double reynolds =
        100.0*par.diameter /
        (par.area*par.viscosity);

    const double argument =
        par.roughness/(3.7*par.diameter) +
        5.74/std::pow(reynolds, 0.9);

    const double friction =
        0.25/std::pow(std::log10(argument), 2);

    const double coefficient =
        friction*par.z*par.gas_constant*
        par.temperature*10000.0 /
        (par.diameter*par.area*par.area);

    for (int i = 0; i <= n; ++i) {
        const double squared =
            8.0e6*8.0e6 -
            coefficient*i*par.dx;

        if (!(squared > 0.0))
            throw std::runtime_error(
                "Invalid reference pressure");

        state.pressure[i] = std::sqrt(squared);
    }

    const auto trajectory =
        record_trajectory(config, state);

    if (trajectory.info != 0 ||
        trajectory.completed_steps != steps)
    {
        throw std::runtime_error(
            "Reference trajectory failed");
    }

    Observations observations;

    observations.inlet_flow = trajectory.inlet_flow;
    observations.outlet_pressure = trajectory.outlet_pressure;
    observations.sigma_flow = 1.0;
    observations.sigma_pressure = 1.0e4;
    observations.include_initial = true;

    const InitialStateScales scales{1.0e5, 10.0};
    const std::vector<double> zero(2*n+1, 0.0);

    const QuadraticRegularizer regularizer(
        n, zero, 1.0e-4, 1.0e-1);

    constexpr double damping = 1.0e-3;

    GaussNewtonOperator A(
        config, trajectory, observations,
        regularizer, scales, damping);

    std::vector<double> u(2*n+1);
    std::vector<double> v(2*n+1);

    for (int j = 0; j < 2*n+1; ++j) {
        u[j] = std::sin(0.071*(j+1));
        v[j] = std::cos(0.053*(j+1));
    }

    const auto Au = A.apply(u);
    const auto Av = A.apply(v);

    const double lhs = cg_dot(u, Av);
    const double rhs = cg_dot(v, Au);

    const double symmetry_error =
        std::abs(lhs-rhs) /
        std::max({1.0, std::abs(lhs), std::abs(rhs)});

    const auto tangent =
        evaluate_tangent_linear(
            config, trajectory, v, scales);

    double observation_quadratic = 0.0;

    for (int k = 0; k <= steps; ++k) {
        const double dm =
            tangent.inlet_flow[k] /
            observations.sigma_flow;

        const double dp =
            tangent.outlet_pressure[k] /
            observations.sigma_pressure;

        observation_quadratic += dm*dm + dp*dp;
    }

    const auto penalty = regularizer.evaluate(v);

    const double expected =
        observation_quadratic +
        cg_dot(v, penalty.gradient) +
        damping*cg_dot(v, v);

    const double actual = cg_dot(v, Av);

    const double quadratic_error =
        std::abs(actual-expected) /
        std::max(1.0, std::abs(expected));

    std::cout
        << "symmetry_error=" << symmetry_error << '\n'
        << "quadratic_error=" << quadratic_error << '\n'
        << "quadratic_form=" << actual << '\n';

    if (symmetry_error > 1.0e-8)
        throw std::runtime_error(
            "Gauss-Newton symmetry failure");

    if (quadratic_error > 1.0e-8)
        throw std::runtime_error(
            "Gauss-Newton quadratic-form failure");

    if (!(actual > 0.0))
        throw std::runtime_error(
            "Gauss-Newton positivity failure");

    std::cout << "PASS Gauss-Newton operator\n";
}
