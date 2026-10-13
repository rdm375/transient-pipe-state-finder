#include "state_finder/transpose_solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

TransposeSolveResult solve_transposed_banded(
    int n,
    std::span<const double> ab,
    int ldab,
    int kl,
    int ku,
    std::span<const double> rhs)
{
    if (n < 1 || kl < 0 || ku < 0 ||
        ldab < 2*kl + ku + 1 ||
        ab.size() != static_cast<std::size_t>(ldab*n) ||
        rhs.size() != static_cast<std::size_t>(n)) {
        throw std::invalid_argument(
            "Invalid transpose-solver dimensions");
    }

    // LAPACK destroys the matrix during factorization.
    std::vector<double> lu(ab.begin(), ab.end());
    std::vector<int> pivots(n);

    int info = 0;

    dgbtrf_(
        &n, &n, &kl, &ku,
        lu.data(), &ldab,
        pivots.data(), &info);

    if (info != 0)
        throw std::runtime_error(
            "LAPACK dgbtrf factorization failed");

    TransposeSolveResult result;
    result.solution.assign(rhs.begin(), rhs.end());

    const char trans = 'T';
    const int nrhs = 1;

    dgbtrs_(
        &trans, &n, &kl, &ku, &nrhs,
        lu.data(), &ldab,
        pivots.data(),
        result.solution.data(), &n,
        &info);

    if (info != 0)
        throw std::runtime_error(
            "LAPACK dgbtrs transpose solve failed");

    // Compute a componentwise residual and normwise
    // relative backward error using the original matrix.
    //
    // eta = ||A^T*x-b||_inf /
    //       (||A^T||_inf*||x||_inf + ||b||_inf)

    std::vector<double> residual(n);
    std::vector<double> row_sums(n, 0.0);

    for (int i = 0; i < n; ++i)
        residual[i] = -rhs[i];

    const int kv = kl + ku;

    for (int j = 0; j < n; ++j) {
        const int first = std::max(0, j-ku);
        const int last = std::min(n-1, j+kl);

        for (int i = first; i <= last; ++i) {
            const double value =
                ab[j*ldab + kv + i-j];

            // A(i,j) contributes to (A^T*x)(j).
            residual[j] += value * result.solution[i];

            // Row j of A^T is column j of A.
            row_sums[j] += std::abs(value);
        }
    }

    double residual_norm = 0.0;
    double matrix_norm = 0.0;
    double solution_norm = 0.0;
    double rhs_norm = 0.0;

    for (int i = 0; i < n; ++i) {
        residual_norm =
            std::max(residual_norm, std::abs(residual[i]));

        matrix_norm =
            std::max(matrix_norm, row_sums[i]);

        solution_norm =
            std::max(solution_norm,
                     std::abs(result.solution[i]));

        rhs_norm =
            std::max(rhs_norm, std::abs(rhs[i]));
    }

    const double denominator =
        matrix_norm*solution_norm + rhs_norm;

    result.relative_backward_error =
        denominator > 0.0
            ? residual_norm/denominator
            : residual_norm;

    if (!std::isfinite(result.relative_backward_error))
        throw std::runtime_error(
            "Nonfinite transpose-solve backward error");

    return result;
}


BandedSolveResult solve_banded(
    int n,
    std::span<const double> ab,
    int ldab,
    int kl,
    int ku,
    std::span<const double> rhs)
{
    if (n < 1 || kl < 0 || ku < 0 ||
        ldab < 2*kl + ku + 1 ||
        ab.size() != static_cast<std::size_t>(ldab*n) ||
        rhs.size() != static_cast<std::size_t>(n))
        throw std::invalid_argument(
            "Invalid banded-solver dimensions");

    std::vector<double> lu(ab.begin(), ab.end());
    std::vector<int> pivots(n);

    int info = 0;

    dgbtrf_(
        &n, &n, &kl, &ku,
        lu.data(), &ldab,
        pivots.data(), &info);

    if (info != 0)
        throw std::runtime_error(
            "LAPACK banded factorization failed");

    BandedSolveResult result;
    result.solution.assign(rhs.begin(), rhs.end());

    const char trans = 'N';
    const int nrhs = 1;

    dgbtrs_(
        &trans, &n, &kl, &ku, &nrhs,
        lu.data(), &ldab,
        pivots.data(),
        result.solution.data(), &n,
        &info);

    if (info != 0)
        throw std::runtime_error(
            "LAPACK nontransposed solve failed");

    // Normwise relative backward error:
    //
    // ||A*x-b||_inf /
    // (||A||_inf*||x||_inf + ||b||_inf)

    std::vector<double> residual(n);
    std::vector<double> row_sums(n, 0.0);

    for (int i = 0; i < n; ++i)
        residual[i] = -rhs[i];

    const int kv = kl + ku;

    for (int j = 0; j < n; ++j) {
        const int first = std::max(0, j-ku);
        const int last = std::min(n-1, j+kl);

        for (int i = first; i <= last; ++i) {
            const double value =
                ab[j*ldab + kv + i-j];

            residual[i] +=
                value*result.solution[j];

            row_sums[i] += std::abs(value);
        }
    }

    double residual_norm = 0.0;
    double matrix_norm = 0.0;
    double solution_norm = 0.0;
    double rhs_norm = 0.0;

    for (int i = 0; i < n; ++i) {
        residual_norm =
            std::max(residual_norm, std::abs(residual[i]));

        matrix_norm =
            std::max(matrix_norm, row_sums[i]);

        solution_norm =
            std::max(solution_norm,
                     std::abs(result.solution[i]));

        rhs_norm =
            std::max(rhs_norm, std::abs(rhs[i]));
    }

    const double denominator =
        matrix_norm*solution_norm + rhs_norm;

    result.relative_backward_error =
        denominator > 0.0
            ? residual_norm/denominator
            : residual_norm;

    if (!std::isfinite(result.relative_backward_error))
        throw std::runtime_error(
            "Nonfinite banded-solve backward error");

    return result;
}




} // namespace state_finder
