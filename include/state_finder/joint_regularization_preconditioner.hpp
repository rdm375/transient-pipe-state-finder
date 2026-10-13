#pragma once

#include "state_finder/gauss_newton_optimizer.hpp"
#include "state_finder/joint_scaled_problem.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace state_finder {

class JointRegularizationPreconditioner {
public:
    JointRegularizationPreconditioner(
        int n,
        int steps,
        JointRegularization regularization,
        double damping)
        : n_(n),
          steps_(steps),
          regularization_(regularization),
          damping_(damping)
    {
        if (n < 1 || steps < 1 ||
            !(damping > 0.0) ||
            !std::isfinite(damping) ||
            !valid_weight(regularization.initial_weight) ||
            !valid_weight(regularization.boundary_weight) ||
            !valid_weight(regularization.boundary_smoothness))
        {
            throw std::invalid_argument(
                "Invalid joint preconditioner parameters");
        }

        // Two identical boundary-history blocks.
        const std::size_t m =
            static_cast<std::size_t>(steps + 1);

        diagonal_.resize(m);
        subdiagonal_.assign(m, 0.0);

        const double base =
            regularization.boundary_weight + damping;

        const double smooth =
            regularization.boundary_smoothness;

        for (std::size_t i = 0; i < m; ++i) {
            const double degree =
                (i == 0 || i+1 == m) ? 1.0 : 2.0;

            const double aii = base + degree*smooth;

            if (i == 0) {
                diagonal_[i] = std::sqrt(aii);
            }
            else {
                subdiagonal_[i] =
                    -smooth / diagonal_[i-1];

                const double pivot =
                    aii -
                    subdiagonal_[i]*subdiagonal_[i];

                if (!(pivot > 0.0) ||
                    !std::isfinite(pivot))
                {
                    throw std::runtime_error(
                        "Joint preconditioner Cholesky failed");
                }

                diagonal_[i] = std::sqrt(pivot);
            }
        }
    }

    std::vector<double> apply(
        const std::vector<double>& rhs) const
    {
        const std::size_t dimension =
            joint_variable_dimension(n_, steps_);

        if (rhs.size() != dimension)
            throw std::invalid_argument(
                "Joint preconditioner dimension mismatch");

        auto x = rhs;

        const std::size_t initial =
            static_cast<std::size_t>(2*n_ + 1);

        const std::size_t history =
            static_cast<std::size_t>(steps_ + 1);

        const double initial_diagonal =
            regularization_.initial_weight + damping_;

        for (std::size_t i = 0; i < initial; ++i)
            x[i] /= initial_diagonal;

        solve_block(x, initial, history);
        solve_block(x, initial + history, history);

        return x;
    }

private:
    static bool valid_weight(double value)
    {
        return std::isfinite(value) && value >= 0.0;
    }

    void solve_block(
        std::vector<double>& x,
        std::size_t offset,
        std::size_t m) const
    {
        // Forward solve: L y = rhs.
        for (std::size_t i = 0; i < m; ++i) {
            if (i > 0) {
                x[offset+i] -=
                    subdiagonal_[i]*x[offset+i-1];
            }

            x[offset+i] /= diagonal_[i];
        }

        // Backward solve: L^T x = y.
        for (std::size_t i = m; i-- > 0;) {
            if (i+1 < m) {
                x[offset+i] -=
                    subdiagonal_[i+1]*x[offset+i+1];
            }

            x[offset+i] /= diagonal_[i];
        }
    }

    int n_;
    int steps_;
    JointRegularization regularization_;
    double damping_;

    std::vector<double> diagonal_;
    std::vector<double> subdiagonal_;
};

inline GaussNewtonPreconditionerFactory
make_joint_regularization_preconditioner_factory(
    int n,
    int steps,
    JointRegularization regularization)
{
    return [n, steps, regularization](
        const LinearOperator&,
        double damping) -> LinearOperator
    {
        auto preconditioner =
            std::make_shared<JointRegularizationPreconditioner>(
                n, steps, regularization, damping);

        return [preconditioner](
            const std::vector<double>& rhs)
        {
            return preconditioner->apply(rhs);
        };
    };
}

} // namespace state_finder
