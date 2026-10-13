#pragma once

#include <span>
#include <vector>

namespace state_finder {

struct MultiRhsSolveResult {
    // Column-major: rhs_column*n + row.
    std::vector<double> solution;
    double relative_backward_error = 0.0;
};

// Solve A*X = B for multiple right-hand sides.
//
// A uses LAPACK general band storage.
// B is column-major with leading dimension n.
//
// Neither input is modified.
MultiRhsSolveResult solve_banded_multi_rhs(
    int n,
    std::span<const double> ab,
    int ldab,
    int kl,
    int ku,
    int nrhs,
    std::span<const double> rhs);

} // namespace state_finder
