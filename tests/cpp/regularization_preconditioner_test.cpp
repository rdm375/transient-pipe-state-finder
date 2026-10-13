#include "state_finder/regularization_preconditioner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    constexpr std::size_t n = 100;
    constexpr double damping = 1.0e-3;

    const std::vector<double> reference(2*n+1, 0.0);

    const state_finder::QuadraticRegularizer regularizer(
        n, reference, 1.0e-4, 0.1);

    const state_finder::RegularizationPreconditioner M(
        regularizer, n, damping);

    std::vector<double> rhs(2*n+1);

    for (std::size_t i = 0; i < rhs.size(); ++i)
        rhs[i] =
            std::sin(0.17*(i+1)) +
            0.3*std::cos(0.031*(i+1));

    const auto x = M.apply(rhs);

    const auto g =
        regularizer.evaluate(x).gradient;

    const auto g0 =
        regularizer.evaluate(
            std::vector<double>(2*n+1, 0.0)).gradient;

    double error2 = 0.0;
    double rhs2 = 0.0;

    for (std::size_t i = 0; i < rhs.size(); ++i) {
        const double residual =
            g[i] - g0[i] + damping*x[i] - rhs[i];

        error2 += residual*residual;
        rhs2 += rhs[i]*rhs[i];
    }

    const double relative_error =
        std::sqrt(error2/rhs2);

    std::cout
        << "preconditioner_relative_residual="
        << relative_error << '\n';

    if (!(relative_error < 1.0e-11))
        throw std::runtime_error(
            "Preconditioner inverse verification failed");

    std::cout
        << "PASS regularization preconditioner\n";
}
