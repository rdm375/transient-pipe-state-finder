#include "state_finder/quadratic_regularizer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    constexpr std::size_t n = 100;
    constexpr std::size_t dim = 2 * n + 1;

    std::vector<double> reference(dim);
    std::vector<double> state(dim);

    for (std::size_t i = 0; i < dim; ++i) {
        reference[i] =
            0.02 * std::sin(0.07 * static_cast<double>(i));

        state[i] =
            reference[i]
            + 0.03 * std::cos(0.11 * static_cast<double>(i));
    }

    const state_finder::QuadraticRegularizer reg(
        n,
        reference,
        1e-2,
        1e-1
    );

    const auto result = reg.evaluate(state);

    constexpr double h = 1e-6;

    double max_error = 0.0;

    for (std::size_t i = 0; i < dim; ++i) {
        auto plus = state;
        auto minus = state;

        plus[i] += h;
        minus[i] -= h;

        const double fd =
            (reg.evaluate(plus).total()
             - reg.evaluate(minus).total())
            / (2.0 * h);

        max_error = std::max(
            max_error,
            std::abs(fd - result.gradient[i])
        );
    }

    std::cout.precision(16);

    std::cout
        << "Zeroth-order penalty: "
        << result.zeroth_order << '\n'
        << "Curvature penalty: "
        << result.curvature << '\n'
        << "Maximum gradient error: "
        << max_error << '\n';

    assert(max_error < 1e-8);

    return 0;
}
