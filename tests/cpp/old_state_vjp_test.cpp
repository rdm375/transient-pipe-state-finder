#include "state_finder/old_state_vjp.hpp"

#include "pipe_sim/transient.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    double sum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        sum += a[i]*b[i];

    return sum;
}

void run_case(int n, double theta, int seed)
{
    const double diameter = 1.0;
    const double area = std::acos(-1.0)/4.0;

    const pipe_sim::TransientParameters par{
        1000.0,
        15.0,
        theta,
        diameter,
        area,
        288.15,
        0.9,
        500.0,
        1.1e-5,
        4.5e-5
    };

    const int width = 2*n + 2;

    std::vector<double> old_state(width);
    std::vector<double> new_state(width);
    std::vector<double> direction(width);
    std::vector<double> lambda(width);

    old_state[0] = 8.0e6;
    old_state[1] = 100.0 + seed;

    new_state[0] = 8.0e6;
    new_state[1] = 101.0 + seed;

    for (int i = 0; i < n; ++i) {
        old_state[2*i+2] =
            95.0 + 0.2*i + seed;

        old_state[2*i+3] =
            8.0e6 - 1500.0*(i+1);

        new_state[2*i+2] =
            old_state[2*i+2] + 0.4;

        new_state[2*i+3] =
            old_state[2*i+3] - 20.0;
    }

    for (int j = 0; j < width; ++j) {
        const bool pressure =
            j == 0 || (j >= 3 && j%2 == 1);

        direction[j] =
            (pressure ? 1.0e6 : 100.0) *
            std::sin(0.37*(j+1+seed));

        lambda[j] =
            std::cos(0.29*(j+2*seed));
    }

    // A small dimensionless step, with physically
    // scaled direction components.
    const double h = 1.0e-4;

    std::vector<double> plus = old_state;
    std::vector<double> minus = old_state;

    for (int j = 0; j < width; ++j) {
        plus[j] += h*direction[j];
        minus[j] -= h*direction[j];
    }

    auto evaluate = [&](const std::vector<double>& state) {
        std::vector<double> pressure(n+1);
        std::vector<double> flow(n);

        pressure[0] = state[0];

        for (int i = 0; i < n; ++i) {
            flow[i] = state[2*i+2];
            pressure[i+1] = state[2*i+3];
        }

        const pipe_sim::TransientBoundary bc{
            state[1],
            100.0,
            105.0,
            8.0e6
        };

        std::vector<double> residual(width);

        pipe_sim::assemble_residual(
            n,
            new_state,
            pressure,
            flow,
            bc,
            par,
            residual);

        return residual;
    };

    const auto rp = evaluate(plus);
    const auto rm = evaluate(minus);

    std::vector<double> fd(width);

    for (int j = 0; j < width; ++j)
        fd[j] = (rp[j]-rm[j])/(2.0*h);

    std::vector<double> vjp(width);

    state_finder::old_state_vjp(
        n, old_state, par, lambda, vjp);

    const double fd_directional = dot(lambda, fd);
    const double analytic_directional = dot(vjp, direction);

    const double scale = std::max(
        {1.0, std::abs(fd_directional),
         std::abs(analytic_directional)});

    const double relative_error =
        std::abs(fd_directional-analytic_directional)/scale;

    std::cout
        << "n=" << n
        << " theta=" << theta
        << " seed=" << seed
        << " FD=" << std::setprecision(16)
        << fd_directional
        << " analytic=" << analytic_directional
        << " relative_error=" << relative_error
        << '\n';

    // This tolerance applies ONLY to the finite-difference
    // diagnostic, not to the analytic derivative.
    if (relative_error > 2.0e-7)
        throw std::runtime_error(
            "Old-state VJP disagrees with residual finite differences");
}

} // namespace

int main()
{
    try {
        for (int n : {2, 10, 100}) {
            for (double theta : {0.0, 0.65, 1.0}) {
                for (int seed : {1, 2, 3})
                    run_case(n, theta, seed);
            }
        }

        std::cout << "PASS old-state VJP\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
