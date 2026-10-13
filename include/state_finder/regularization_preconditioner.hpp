#pragma once

#include "state_finder/quadratic_regularizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace state_finder {

// Exact inverse of the quadratic regularization Hessian
// plus positive Levenberg-Marquardt damping.
//
// Ordering:
//   pressure[0:n]
//   flow[0:n]
//   inlet_flow
//
// The regularization Hessian has half-bandwidth 2.
class RegularizationPreconditioner {
public:
    RegularizationPreconditioner(
        const QuadraticRegularizer& regularizer,
        std::size_t cells,
        double damping)
        : size_(2*cells + 1),
          diagonal_(size_, 0.0),
          sub1_(size_, 0.0),
          sub2_(size_, 0.0)
    {
        if (cells < 3 ||
            !(damping > 0.0) ||
            !std::isfinite(damping))
        {
            throw std::invalid_argument(
                "Invalid preconditioner parameters");
        }

        const std::vector<double> zero(size_, 0.0);

        const auto g0 =
            regularizer.evaluate(zero).gradient;

        if (g0.size() != size_)
            throw std::runtime_error(
                "Regularizer dimension mismatch");

        // Construct the three nonzero Hessian diagonals
        // directly from the verified regularizer gradient.
        //
        // A column probe costs only O(n), and the matrix
        // is assembled once per damping value.
        for (std::size_t j = 0; j < size_; ++j) {
            auto basis = zero;
            basis[j] = 1.0;

            const auto g =
                regularizer.evaluate(basis).gradient;

            diagonal_[j] = g[j] - g0[j] + damping;

            if (j + 1 < size_)
                sub1_[j+1] = g[j+1] - g0[j+1];

            if (j + 2 < size_)
                sub2_[j+2] = g[j+2] - g0[j+2];
        }

        // Banded Cholesky:
        //
        // A = L L^T
        //
        // L has two subdiagonals.
        for (std::size_t i = 0; i < size_; ++i) {
            if (i >= 2) {
                sub2_[i] /= diagonal_[i-2];
            }

            if (i >= 1) {
                if (i >= 2)
                    sub1_[i] -=
                        sub2_[i]*sub1_[i-1];

                sub1_[i] /= diagonal_[i-1];
            }

            double pivot = diagonal_[i];

            if (i >= 1)
                pivot -= sub1_[i]*sub1_[i];

            if (i >= 2)
                pivot -= sub2_[i]*sub2_[i];

            if (!(pivot > 0.0) ||
                !std::isfinite(pivot))
            {
                throw std::runtime_error(
                    "Preconditioner is not positive definite");
            }

            diagonal_[i] = std::sqrt(pivot);
        }
    }

    // Apply L^-1.
    std::vector<double> solve_lower(
        const std::vector<double>& rhs) const
    {
        if (rhs.size() != size_)
            throw std::invalid_argument(
                "Lower solve dimension mismatch");

        auto x = rhs;

        for (std::size_t i = 0; i < size_; ++i) {
            if (i >= 1)
                x[i] -= sub1_[i]*x[i-1];

            if (i >= 2)
                x[i] -= sub2_[i]*x[i-2];

            x[i] /= diagonal_[i];
        }

        return x;
    }

    // Apply L^-T.
    std::vector<double> solve_upper(
        const std::vector<double>& rhs) const
    {
        if (rhs.size() != size_)
            throw std::invalid_argument(
                "Upper solve dimension mismatch");

        auto x = rhs;

        for (std::size_t i = size_; i-- > 0;) {
            if (i+1 < size_)
                x[i] -= sub1_[i+1]*x[i+1];

            if (i+2 < size_)
                x[i] -= sub2_[i+2]*x[i+2];

            x[i] /= diagonal_[i];
        }

        return x;
    }

    std::vector<double> apply(
        const std::vector<double>& rhs) const
    {
        if (rhs.size() != size_)
            throw std::invalid_argument(
                "Preconditioner vector dimension mismatch");

        auto x = rhs;

        // Forward substitution: Ly = rhs.
        for (std::size_t i = 0; i < size_; ++i) {
            if (i >= 1)
                x[i] -= sub1_[i]*x[i-1];

            if (i >= 2)
                x[i] -= sub2_[i]*x[i-2];

            x[i] /= diagonal_[i];
        }

        // Backward substitution: L^T x = y.
        for (std::size_t k = size_; k-- > 0;) {
            if (k + 1 < size_)
                x[k] -= sub1_[k+1]*x[k+1];

            if (k + 2 < size_)
                x[k] -= sub2_[k+2]*x[k+2];

            x[k] /= diagonal_[k];
        }

        return x;
    }

private:
    std::size_t size_;
    std::vector<double> diagonal_;
    std::vector<double> sub1_;
    std::vector<double> sub2_;
};

} // namespace state_finder
