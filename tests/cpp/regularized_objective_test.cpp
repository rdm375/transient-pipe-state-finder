#include "state_finder/regularized_objective.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main()
{
    constexpr std::size_t n = 10;
    constexpr std::size_t dim = 2*n + 1;

    std::vector<double> reference(dim, 0.0);
    std::vector<double> initial(dim, 0.0);

    for (std::size_t i = 0; i < dim; ++i) {
        reference[i] = 0.01 * std::sin(0.3 * i);
        initial[i] = 0.1 * std::cos(0.2 * i);
    }

    // A nonlinear observation objective with an analytic gradient.
    state_finder::ObjectiveFunction observation =
        [](const std::vector<double>& x, bool gradient) {
            state_finder::OptimizationEvaluation result;
            result.valid = true;

            if (gradient)
                result.gradient.resize(x.size());

            for (std::size_t i = 0; i < x.size(); ++i) {
                const double residual =
                    x[i] + 0.1*x[i]*x[i] - 0.02;

                result.objective += 0.5*residual*residual;

                if (gradient) {
                    result.gradient[i] =
                        residual * (1.0 + 0.2*x[i]);
                }
            }

            return result;
        };

    state_finder::QuadraticRegularizer regularizer(
        n, reference, 1e-2, 1e-1);

    auto objective =
        state_finder::make_regularized_objective(
            observation, regularizer);

    const auto initial_eval = objective(initial, true);

    constexpr double h = 1e-6;
    double max_gradient_error = 0.0;

    for (std::size_t i = 0; i < dim; ++i) {
        auto plus = initial;
        auto minus = initial;

        plus[i] += h;
        minus[i] -= h;

        const double fd =
            (objective(plus, false).objective -
             objective(minus, false).objective)
            / (2.0*h);

        max_gradient_error = std::max(
            max_gradient_error,
            std::abs(fd - initial_eval.gradient[i]));
    }

    state_finder::OptimizationOptions options;
    options.max_iterations = 100;
    options.gradient_tolerance = 1e-9;

    const auto optimized =
        state_finder::minimize_lbfgs(
            objective, initial, options);

    std::cout.precision(16);

    std::cout
        << "Maximum gradient error: "
        << max_gradient_error << '\n'
        << "Initial objective: "
        << optimized.initial_objective << '\n'
        << "Final objective: "
        << optimized.final_objective << '\n'
        << "Iterations: "
        << optimized.iterations << '\n'
        << "Reason: "
        << optimized.reason << '\n';

    assert(max_gradient_error < 1e-8);
    assert(optimized.final_objective <
           optimized.initial_objective);
    assert(optimized.gradient_norm < 1e-6);

    return 0;
}
