#include "state_finder/optimizer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main()
{
    try {
        using namespace state_finder;

        int objective_only_calls = 0;
        int gradient_calls = 0;

        const ObjectiveFunction quadratic =
            [&](const std::vector<double>& x,
                bool need_gradient)
            {
                OptimizationEvaluation result;

                if (x.size() != 2)
                    return result;

                const double a = x[0] - 2.0;
                const double b = x[1] + 3.0;

                result.objective =
                    0.5 * (4.0*a*a + b*b);

                if (need_gradient) {
                    ++gradient_calls;

                    result.gradient = {
                        4.0*a,
                        b
                    };
                }
                else {
                    ++objective_only_calls;
                }

                result.valid = true;
                return result;
            };

        OptimizationOptions options;
        options.max_iterations = 100;
        options.history_size = 5;
        options.gradient_tolerance = 1.0e-10;
        options.relative_objective_tolerance = 0.0;

        const auto result = minimize_lbfgs(
            quadratic,
            {10.0, -10.0},
            options);

        std::cout
            << "converged=" << result.converged
            << " iterations=" << result.iterations
            << " initial_objective="
            << result.initial_objective
            << " final_objective="
            << result.final_objective
            << " gradient_norm="
            << result.gradient_norm
            << " x0=" << result.variables[0]
            << " x1=" << result.variables[1]
            << " objective_only_calls="
            << objective_only_calls
            << " gradient_calls="
            << gradient_calls
            << '\n';

        require(
            result.converged,
            "Optimizer did not converge");

        require(
            std::abs(result.variables[0] - 2.0) < 1.0e-8,
            "Incorrect first minimizer component");

        require(
            std::abs(result.variables[1] + 3.0) < 1.0e-8,
            "Incorrect second minimizer component");

        require(
            result.final_objective < 1.0e-15,
            "Final objective is too large");

        require(
            result.final_objective <
                result.initial_objective,
            "Objective did not decrease");

        require(
            objective_only_calls > 0,
            "Objective-only line search was not used");

        require(
            gradient_calls > 0,
            "Gradient callback was not used");

        require(
            result.objective_evaluations ==
                objective_only_calls + gradient_calls,
            "Objective evaluation count mismatch");

        require(
            result.gradient_evaluations == gradient_calls,
            "Gradient evaluation count mismatch");

        std::cout
            << "PASS independent L-BFGS quadratic test\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
