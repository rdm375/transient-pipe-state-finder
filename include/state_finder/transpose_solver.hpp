#pragma once

#include <span>
#include <vector>

namespace state_finder {

struct TransposeSolveResult {
    std::vector<double> solution;
    double relative_backward_error = 0.0;
};

// Solve A^T x = b.
//
// Input uses the forward simulator's LAPACK-compatible
// column-major band storage.
//
// The input matrix and RHS are not modified.
TransposeSolveResult solve_transposed_banded(
    int n,
    std::span<const double> ab,
    int ldab,
    int kl,
    int ku,
    std::span<const double> rhs);

} // namespace state_finder
