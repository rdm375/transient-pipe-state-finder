#pragma once

#include "state_finder/matrix_free_cg.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace state_finder {

struct DenseGNResult {
    std::vector<double> solution;
    int operator_calls = 0;
    double relative_residual = 0.0;
    double maximum_symmetry_defect = 0.0;
};

inline DenseGNResult solve_dense_gauss_newton(
    const LinearOperator& A,
    const std::vector<double>& rhs)
{
    if (!A || rhs.empty())
        throw std::invalid_argument("Invalid dense GN arguments");

    const std::size_t n = rhs.size();

    DenseGNResult result;
    std::vector<double> H(n*n, 0.0);

    // Assemble H column by column, without symmetrizing it.
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<double> e(n, 0.0);
        e[j] = 1.0;

        const auto column = A(e);
        ++result.operator_calls;

        if (column.size() != n)
            throw std::runtime_error(
                "Dense GN operator dimension mismatch");

        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(column[i]))
                throw std::runtime_error(
                    "Nonfinite dense GN matrix");

            H[i*n+j] = column[i];
        }
    }

    // Check symmetry before Cholesky.
    double maximum_entry = 0.0;
    double maximum_defect = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            maximum_entry =
                std::max(maximum_entry, std::abs(H[i*n+j]));

            maximum_defect =
                std::max(
                    maximum_defect,
                    std::abs(H[i*n+j]-H[j*n+i]));
        }
    }

    result.maximum_symmetry_defect =
        maximum_defect/std::max(1.0, maximum_entry);

    if (result.maximum_symmetry_defect > 1.0e-8)
        throw std::runtime_error(
            "Dense GN matrix fails symmetry check");

    // Cholesky H = L L^T.
    // Only the lower triangle is used; no matrix modification
    // or additional regularization is introduced.
    std::vector<double> L(n*n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = H[i*n+j];

            for (std::size_t k = 0; k < j; ++k)
                value -= L[i*n+k]*L[j*n+k];

            if (i == j) {
                if (!(value > 0.0) || !std::isfinite(value))
                    throw std::runtime_error(
                        "Dense GN Cholesky failed");

                L[i*n+i] = std::sqrt(value);
            }
            else {
                L[i*n+j] = value/L[j*n+j];
            }
        }
    }

    // Forward substitution.
    std::vector<double> y(n, 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        double value = rhs[i];

        for (std::size_t j = 0; j < i; ++j)
            value -= L[i*n+j]*y[j];

        y[i] = value/L[i*n+i];
    }

    // Backward substitution.
    result.solution.assign(n, 0.0);

    for (std::size_t i = n; i-- > 0;) {
        double value = y[i];

        for (std::size_t j = i+1; j < n; ++j)
            value -= L[j*n+i]*result.solution[j];

        result.solution[i] = value/L[i*n+i];
    }

    // Independent residual check against the original operator.
    const auto Ax = A(result.solution);
    ++result.operator_calls;

    if (Ax.size() != n)
        throw std::runtime_error(
            "Dense GN residual dimension mismatch");

    std::vector<double> residual(n);

    for (std::size_t i = 0; i < n; ++i)
        residual[i] = rhs[i]-Ax[i];

    result.relative_residual =
        cg_norm(residual)/
        std::max(
            cg_norm(rhs),
            std::numeric_limits<double>::min());

    if (!std::isfinite(result.relative_residual))
        throw std::runtime_error(
            "Nonfinite dense GN residual");

    return result;
}

} // namespace state_finder
