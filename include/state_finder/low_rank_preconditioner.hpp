#pragma once

#include "state_finder/matrix_free_cg.hpp"
#include "state_finder/quadratic_regularizer.hpp"
#include "state_finder/regularization_preconditioner.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace state_finder {

// Symmetric low-rank correction in regularization-whitened coordinates.
//
// M0 = R + damping*I = L*L^T
// B  = L^-1 * (K^T K) * L^-T
//
// For orthonormal Q and T = Q^T B Q:
//
// M_LR^-1 = L^-T [I - Q(I-(I+T)^-1)Q^T] L^-1
//
// Construction requires approximately 2*rank GN products.
class LowRankPreconditioner {
public:
    LowRankPreconditioner(
        std::shared_ptr<const RegularizationPreconditioner> base,
        const QuadraticRegularizer& regularizer,
        const LinearOperator& A,
        double damping,
        std::size_t n,
        int rank,
        int& setup_calls)
        : base_(std::move(base))
    {
        if (!base_ || !A || rank < 1 ||
            !(damping > 0.0) || n == 0)
        {
            throw std::invalid_argument(
                "Invalid low-rank preconditioner arguments");
        }

        const CGVector zero(n, 0.0);
        const auto g0 = regularizer.evaluate(zero).gradient;

        if (g0.size() != n)
            throw std::runtime_error(
                "Regularizer dimension mismatch");

        // Apply the observation Hessian in whitened coordinates.
        //
        // A = K^T K + R + damping*I.
        auto B = [&](const CGVector& v) -> CGVector {
            const auto x = base_->solve_upper(v);

            auto y = A(x);
            ++setup_calls;

            const auto gx = regularizer.evaluate(x).gradient;

            if (y.size() != n || gx.size() != n)
                throw std::runtime_error(
                    "Low-rank operator dimension mismatch");

            for (std::size_t i = 0; i < n; ++i)
                y[i] -= gx[i] - g0[i] + damping*x[i];

            return base_->solve_lower(y);
        };

        std::mt19937_64 rng(20261013);
        std::normal_distribution<double> gaussian(0.0, 1.0);

        // Construct an orthonormal basis for the sampled range.
        for (int k = 0; k < rank; ++k) {
            CGVector probe(n);

            for (double& value : probe)
                value = gaussian(rng);

            auto q = B(probe);

            // Twice-reorthogonalized modified Gram-Schmidt.
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& u : Q_) {
                    const double alpha = cg_dot(u, q);

                    for (std::size_t i = 0; i < n; ++i)
                        q[i] -= alpha*u[i];
                }
            }

            const double qnorm = cg_norm(q);

            if (!(qnorm > 1.0e-12) ||
                !std::isfinite(qnorm))
                continue;

            for (double& value : q)
                value /= qnorm;

            Q_.push_back(std::move(q));
        }

        const std::size_t m = Q_.size();

        // Project the observation Hessian.
        std::vector<CGVector> BQ;
        BQ.reserve(m);

        for (const auto& q : Q_)
            BQ.push_back(B(q));

        std::vector<CGVector> T(
            m, CGVector(m, 0.0));

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j)
                T[i][j] = cg_dot(Q_[i], BQ[j]);
        }

        // Symmetrize against floating-point roundoff.
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                const double value =
                    0.5*(T[i][j] + T[j][i]);

                T[i][j] = value;
                T[j][i] = value;
            }
        }

        // Cholesky factorization of I + T.
        chol_.assign(m, CGVector(m, 0.0));

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                double s = T[i][j] + (i == j ? 1.0 : 0.0);

                for (std::size_t k = 0; k < j; ++k)
                    s -= chol_[i][k]*chol_[j][k];

                if (i == j) {
                    if (!(s > 0.0) || !std::isfinite(s))
                        throw std::runtime_error(
                            "Projected matrix is not SPD");

                    chol_[i][j] = std::sqrt(s);
                }
                else {
                    chol_[i][j] = s/chol_[j][j];
                }
            }
        }
    }

    CGVector apply(const CGVector& rhs) const
    {
        auto z = base_->solve_lower(rhs);

        const std::size_t m = Q_.size();

        CGVector t(m, 0.0);

        for (std::size_t j = 0; j < m; ++j)
            t[j] = cg_dot(Q_[j], z);

        auto u = t;

        // Solve (I+T)u=t.
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < i; ++j)
                u[i] -= chol_[i][j]*u[j];

            u[i] /= chol_[i][i];
        }

        for (std::size_t i = m; i-- > 0;) {
            for (std::size_t j = i+1; j < m; ++j)
                u[i] -= chol_[j][i]*u[j];

            u[i] /= chol_[i][i];
        }

        for (std::size_t j = 0; j < m; ++j) {
            const double alpha = u[j] - t[j];

            for (std::size_t i = 0; i < z.size(); ++i)
                z[i] += alpha*Q_[j][i];
        }

        return base_->solve_upper(z);
    }

private:
    std::shared_ptr<const RegularizationPreconditioner> base_;
    std::vector<CGVector> Q_;
    std::vector<CGVector> chol_;
};

} // namespace state_finder
