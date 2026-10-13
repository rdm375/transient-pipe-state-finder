#include "state_finder/joint_regularization_preconditioner.hpp"
#include "state_finder/joint_regularization_gradient.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace state_finder;

int main()
{
    constexpr int n = 6;
    constexpr int steps = 5;
    constexpr double damping = 1.0e-3;

    JointRegularization reg;
    reg.initial_weight = 1.0e-4;
    reg.boundary_weight = 1.0e-2;
    reg.boundary_smoothness = 1.0e-2;

    const auto dimension =
        joint_variable_dimension(n, steps);

    std::vector<double> rhs(dimension);

    for (std::size_t i = 0; i < dimension; ++i) {
        rhs[i] =
            std::sin(0.17*(i+1)) +
            0.3*std::cos(0.031*(i+1));
    }

    JointRegularizationPreconditioner M(
        n, steps, reg, damping);

    const auto x = M.apply(rhs);

    const auto Hx =
        joint_regularization_gradient(
            n, steps, x, reg);

    double error2 = 0.0;
    double rhs2 = 0.0;

    for (std::size_t i = 0; i < dimension; ++i) {
        const double residual =
            Hx[i] + damping*x[i] - rhs[i];

        error2 += residual*residual;
        rhs2 += rhs[i]*rhs[i];
    }

    const double relative_error =
        std::sqrt(error2/rhs2);

    std::cout
        << "joint_preconditioner_relative_residual="
        << relative_error << '\n';

    if (!(relative_error < 1.0e-11))
        throw std::runtime_error(
            "Joint preconditioner inverse mismatch");

    std::cout
        << "PASS joint regularization preconditioner\n";
}
