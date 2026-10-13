#include "state_finder/matrix_free_cg.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    using namespace state_finder;

    // SPD tridiagonal system.
    //
    // A = tridiag(-1,4,-1)
    //
    // Construct b from a known solution.

    constexpr int n = 100;

    CGVector truth(n);

    for (int i = 0; i < n; ++i)
        truth[i] = std::sin(0.07 * (i + 1));

    const LinearOperator A = [](const CGVector& x) {
        CGVector y(x.size());

        for (std::size_t i = 0; i < x.size(); ++i) {
            y[i] = 4.0 * x[i];

            if (i > 0)
                y[i] -= x[i - 1];

            if (i + 1 < x.size())
                y[i] -= x[i + 1];
        }

        return y;
    };

    const CGVector b = A(truth);

    const LinearOperator preconditioner =
        [](const CGVector& r) {
            CGVector z = r;

            for (double& value : z)
                value /= 4.0;

            return z;
        };

    CGOptions options;
    options.max_iterations = 200;
    options.relative_tolerance = 1.0e-12;

    const auto result = solve_matrix_free_cg(
        A, b, options, preconditioner);

    CGVector error(n);

    for (int i = 0; i < n; ++i)
        error[i] = result.solution[i] - truth[i];

    const double relative_error =
        cg_norm(error) / cg_norm(truth);

    std::cout
        << "iterations=" << result.iterations << '\n'
        << "operator_calls=" << result.operator_calls << '\n'
        << "initial_residual=" << result.initial_residual << '\n'
        << "final_residual=" << result.final_residual << '\n'
        << "relative_solution_error=" << relative_error
        << '\n';

    if (!result.converged || relative_error > 1.0e-10)
        throw std::runtime_error(
            "Matrix-free CG regression failed");

    std::cout << "PASS matrix-free CG\n";
}
