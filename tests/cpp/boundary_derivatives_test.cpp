#include "state_finder/boundary_derivatives.hpp"

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
    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i]*b[i];

    return result;
}

} // namespace

int main()
{
    try {
        constexpr int n = 10;
        constexpr double theta = 0.65;

        const double dp = 1234.5;
        const double dq_old = -0.37;
        const double dq_new = 0.82;

        std::vector<double> jvp(2*n+2);

        boundary_residual_jvp(
            n, theta,
            dp, dq_old, dq_new,
            jvp);

        require(jvp[0] == -dp,
                "Incorrect inlet-pressure derivative");

        const double expected_outlet =
            (1.0-theta)*dq_old + theta*dq_new;

        require(
            std::abs(jvp.back()-expected_outlet) < 1e-14,
            "Incorrect outlet-flow derivative");

        std::vector<double> lambda(2*n+2);

        for (std::size_t i = 0; i < lambda.size(); ++i)
            lambda[i] = std::sin(0.37*(i+1));

        double gp = 0.0;
        double gq_old = 0.0;
        double gq_new = 0.0;

        boundary_residual_vjp(
            n, theta, lambda,
            gp, gq_old, gq_new);

        const double lhs = dot(jvp, lambda);

        const double rhs =
            dp*gp +
            dq_old*gq_old +
            dq_new*gq_new;

        const double relative_error =
            std::abs(lhs-rhs) /
            std::max({1.0, std::abs(lhs), std::abs(rhs)});

        std::cout
            << "boundary_jvp_vjp_lhs=" << lhs << '\n'
            << "boundary_jvp_vjp_rhs=" << rhs << '\n'
            << "relative_error=" << relative_error << '\n';

        require(relative_error < 1e-14,
                "Boundary JVP/VJP identity failed");

        // Verify the exact theta endpoints.
        boundary_residual_jvp(
            n, 0.0, dp, dq_old, dq_new, jvp);

        require(jvp.back() == dq_old,
                "Theta=0 endpoint failed");

        boundary_residual_jvp(
            n, 1.0, dp, dq_old, dq_new, jvp);

        require(jvp.back() == dq_new,
                "Theta=1 endpoint failed");

        std::cout
            << "PASS boundary residual derivatives\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
