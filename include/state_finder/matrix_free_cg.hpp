#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

namespace state_finder {

using CGVector = std::vector<double>;

using LinearOperator =
    std::function<CGVector(const CGVector&)>;

struct CGOptions {
    int max_iterations = 200;
    double relative_tolerance = 1.0e-8;
    double absolute_tolerance = 1.0e-12;
};

struct CGResult {
    CGVector solution;
    int iterations = 0;
    int operator_calls = 0;
    double initial_residual = 0.0;
    double final_residual = 0.0;
    bool converged = false;
};

inline double cg_dot(
    const CGVector& a,
    const CGVector& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("CG dimension mismatch");

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];

    return result;
}

inline double cg_norm(const CGVector& x)
{
    return std::sqrt(cg_dot(x, x));
}

// Solve A*x=b using matrix-free preconditioned CG.
//
// A must be symmetric positive definite.
// M_inverse applies an SPD preconditioner inverse.
// An empty M_inverse selects the identity preconditioner.
inline CGResult solve_matrix_free_cg(
    const LinearOperator& A,
    const CGVector& b,
    const CGOptions& options = {},
    const LinearOperator& M_inverse = {})
{
    if (!A || b.empty() ||
        options.max_iterations <= 0 ||
        !(options.relative_tolerance > 0.0) ||
        !(options.absolute_tolerance >= 0.0))
    {
        throw std::invalid_argument("Invalid CG arguments");
    }

    const std::size_t n = b.size();

    CGResult result;
    result.solution.assign(n, 0.0);

    CGVector r = b;

    result.initial_residual = cg_norm(r);
    result.final_residual = result.initial_residual;

    const double tolerance = std::max(
        options.absolute_tolerance,
        options.relative_tolerance *
            result.initial_residual);

    if (result.initial_residual <= tolerance) {
        result.converged = true;
        return result;
    }

    auto precondition = [&](const CGVector& v) {
        CGVector z = M_inverse ? M_inverse(v) : v;

        if (z.size() != n)
            throw std::runtime_error(
                "CG preconditioner dimension mismatch");

        return z;
    };

    CGVector z = precondition(r);
    CGVector direction = z;

    double rz = cg_dot(r, z);

    if (!(rz > 0.0) || !std::isfinite(rz))
        throw std::runtime_error(
            "CG preconditioner is not positive definite");

    for (int iteration = 0;
         iteration < options.max_iterations;
         ++iteration)
    {
        const CGVector Ad = A(direction);
        ++result.operator_calls;

        if (Ad.size() != n)
            throw std::runtime_error(
                "CG operator dimension mismatch");

        const double curvature = cg_dot(direction, Ad);

        if (!(curvature > 0.0) ||
            !std::isfinite(curvature))
        {
            throw std::runtime_error(
                "CG operator is not positive definite");
        }

        const double alpha = rz / curvature;

        if (!std::isfinite(alpha))
            throw std::runtime_error(
                "Nonfinite CG step");

        for (std::size_t i = 0; i < n; ++i) {
            result.solution[i] += alpha * direction[i];
            r[i] -= alpha * Ad[i];
        }

        result.iterations = iteration + 1;
        result.final_residual = cg_norm(r);

        if (result.final_residual <= tolerance) {
            result.converged = true;
            break;
        }

        z = precondition(r);

        const double rz_next = cg_dot(r, z);

        if (!(rz_next > 0.0) ||
            !std::isfinite(rz_next))
        {
            throw std::runtime_error(
                "Invalid CG preconditioned residual");
        }

        const double beta = rz_next / rz;

        for (std::size_t i = 0; i < n; ++i)
            direction[i] = z[i] + beta * direction[i];

        rz = rz_next;
    }

    return result;
}

} // namespace state_finder
