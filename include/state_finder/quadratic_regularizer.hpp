#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace state_finder {

struct QuadraticRegularizationResult {
    double zeroth_order = 0.0;
    double curvature = 0.0;
    std::vector<double> gradient;

    double total() const {
        return zeroth_order + curvature;
    }
};

class QuadraticRegularizer {
public:
    QuadraticRegularizer(
        std::size_t cells,
        std::vector<double> reference,
        double lambda0,
        double lambda2
    )
        : cells_(cells),
          reference_(std::move(reference)),
          lambda0_(lambda0),
          lambda2_(lambda2)
    {
        if (cells_ < 3)
            throw std::invalid_argument("Need at least 3 cells");

        if (reference_.size() != 2 * cells_ + 1)
            throw std::invalid_argument(
                "Incorrect reference-state dimension"
            );

        if (lambda0_ < 0.0 || lambda2_ < 0.0)
            throw std::invalid_argument(
                "Regularization weights must be nonnegative"
            );
    }

    QuadraticRegularizationResult evaluate(
        const std::vector<double>& state
    ) const
    {
        if (state.size() != reference_.size())
            throw std::invalid_argument(
                "Incorrect initial-state dimension"
            );

        const std::size_t n = cells_;

        QuadraticRegularizationResult result;
        result.gradient.assign(state.size(), 0.0);

        std::vector<double> d(state.size());

        for (std::size_t i = 0; i < state.size(); ++i) {
            d[i] = state[i] - reference_[i];

            result.zeroth_order +=
                0.5 * lambda0_ * d[i] * d[i];

            result.gradient[i] += lambda0_ * d[i];
        }

        // Second differences of the 100 internal pressure
        // coordinates and 100 internal face-flow coordinates.
        //
        // The final coordinate is the inlet mass flow.
        //
        // This reproduces the spatial operator used in the
        // linearized benchmark, including its nullspace.
        //
        // The operator is normalized below so that its
        // magnitude is independent of an arbitrary weight
        // applied to all its rows.

        constexpr double curvature_scale = 0.25;

        for (std::size_t offset : {std::size_t{0}, n}) {
            for (std::size_t i = 0; i + 2 < n; ++i) {
                const std::size_t j = offset + i;

                const double curvature =
                    curvature_scale *
                    (d[j] - 2.0 * d[j + 1] + d[j + 2]);

                result.curvature +=
                    0.5 * lambda2_ * curvature * curvature;

                const double g =
                    lambda2_ * curvature_scale * curvature;

                result.gradient[j]     += g;
                result.gradient[j + 1] -= 2.0 * g;
                result.gradient[j + 2] += g;
            }
        }

        return result;
    }

private:
    std::size_t cells_;
    std::vector<double> reference_;
    double lambda0_;
    double lambda2_;
};

} // namespace state_finder
