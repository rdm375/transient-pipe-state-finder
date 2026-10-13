#include "state_finder/banded_multi_rhs.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

extern "C" {

void dgbtrf_(
    const int* m,
    const int* n,
    const int* kl,
    const int* ku,
    double* ab,
    const int* ldab,
    int* ipiv,
    int* info);

void dgbtrs_(
    const char* trans,
    const int* n,
    const int* kl,
    const int* ku,
    const int* nrhs,
    const double* ab,
    const int* ldab,
    const int* ipiv,
    double* b,
    const int* ldb,
    int* info);

}

namespace state_finder {

MultiRhsSolveResult solve_banded_multi_rhs(
    int n,
    std::span<const double> ab,
    int ldab,
    int kl,
    int ku,
    int nrhs,
    std::span<const double> rhs)
{
    if (n < 1 || nrhs < 1 ||
        kl < 0 || ku < 0 ||
        ldab < 2*kl + ku + 1 ||
        ab.size() != static_cast<std::size_t>(ldab)*n ||
        rhs.size() != static_cast<std::size_t>(n)*nrhs)
        throw std::invalid_argument(
            "Invalid multiple-RHS banded dimensions");

    std::vector<double> lu(ab.begin(), ab.end());
    std::vector<int> pivots(n);

    int info = 0;

    dgbtrf_(
        &n, &n, &kl, &ku,
        lu.data(), &ldab,
        pivots.data(), &info);

    if (info != 0)
        throw std::runtime_error(
            "Multiple-RHS banded factorization failed: info=" +
            std::to_string(info));

    MultiRhsSolveResult result;
    result.solution.assign(rhs.begin(), rhs.end());

    const char trans = 'N';

    dgbtrs_(
        &trans, &n, &kl, &ku, &nrhs,
        lu.data(), &ldab,
        pivots.data(),
        result.solution.data(), &n,
        &info);

    if (info != 0)
        throw std::runtime_error(
            "Multiple-RHS banded solve failed: info=" +
            std::to_string(info));

    // Compute a normwise backward error for each RHS,
    // then report the maximum.
    //
    // eta_j = ||A*x_j-b_j||_inf /
    //         (||A||_inf*||x_j||_inf + ||b_j||_inf)

    std::vector<double> row_sums(n, 0.0);

    const int kv = kl + ku;

    for (int j = 0; j < n; ++j) {
        const int first = std::max(0, j-ku);
        const int last = std::min(n-1, j+kl);

        for (int i = first; i <= last; ++i) {
            const double value =
                ab[j*ldab + kv + i-j];

            row_sums[i] += std::abs(value);
        }
    }

    double matrix_norm = 0.0;

    for (double value : row_sums)
        matrix_norm = std::max(matrix_norm, value);

    for (int column = 0; column < nrhs; ++column) {
        const auto offset =
            static_cast<std::size_t>(column)*n;

        std::vector<double> residual(n);

        for (int i = 0; i < n; ++i)
            residual[i] = -rhs[offset+i];

        for (int j = 0; j < n; ++j) {
            const int first = std::max(0, j-ku);
            const int last = std::min(n-1, j+kl);

            for (int i = first; i <= last; ++i) {
                const double value =
                    ab[j*ldab + kv + i-j];

                residual[i] +=
                    value*result.solution[offset+j];
            }
        }

        double residual_norm = 0.0;
        double solution_norm = 0.0;
        double rhs_norm = 0.0;

        for (int i = 0; i < n; ++i) {
            residual_norm = std::max(
                residual_norm, std::abs(residual[i]));

            solution_norm = std::max(
                solution_norm,
                std::abs(result.solution[offset+i]));

            rhs_norm = std::max(
                rhs_norm, std::abs(rhs[offset+i]));
        }

        const double denominator =
            matrix_norm*solution_norm + rhs_norm;

        const double error =
            denominator > 0.0
                ? residual_norm/denominator
                : residual_norm;

        if (!std::isfinite(error))
            throw std::runtime_error(
                "Nonfinite multiple-RHS backward error");

        result.relative_backward_error = std::max(
            result.relative_backward_error, error);
    }

    return result;
}

} // namespace state_finder
