#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace state_finder {

// Time-major, four-channel measurement covariance.
//
// Ordering at each time:
//   inlet pressure, inlet flow, outlet pressure, outlet flow.
//
// C[(k,c),(j,d)] = delta(c,d) sigma[c]^2 rho^abs(k-j)
//
// The standard deviations are in engineering-normalized
// observation coordinates, NOT physical units.
class TemporalMeasurementCovariance {
public:
    TemporalMeasurementCovariance(
        std::size_t time_count,
        double dt,
        double correlation_time,
        std::array<double, 4> sigma = {
            0.5, 0.5, 0.5, 0.5
        })
        : count_(time_count),
          sigma_(sigma)
    {
        if (count_ == 0 ||
            !std::isfinite(dt) ||
            !(dt > 0.0) ||
            !std::isfinite(correlation_time) ||
            correlation_time < 0.0)
        {
            throw std::invalid_argument(
                "Invalid temporal covariance dimensions or times");
        }

        for (double value : sigma_) {
            if (!std::isfinite(value) || !(value > 0.0))
                throw std::invalid_argument(
                    "Invalid measurement standard deviation");
        }

        rho_ = correlation_time == 0.0
            ? 0.0
            : std::exp(-dt / correlation_time);

        // Stable even when rho is close to one.
        denominator_ = -std::expm1(
            -2.0 * dt / correlation_time_safe(
                correlation_time));

        if (correlation_time == 0.0)
            denominator_ = 1.0;

        if (!std::isfinite(denominator_) ||
            !(denominator_ > 0.0))
        {
            throw std::invalid_argument(
                "Degenerate temporal covariance");
        }
    }

    [[nodiscard]]
    std::size_t dimension() const noexcept
    {
        return 4 * count_;
    }

    [[nodiscard]]
    double correlation() const noexcept
    {
        return rho_;
    }

    [[nodiscard]]
    std::vector<double> apply_inverse(
        std::span<const double> input) const
    {
        if (input.size() != dimension())
            throw std::invalid_argument(
                "Covariance input dimension mismatch");

        std::vector<double> output(input.size());

        for (std::size_t c = 0; c < 4; ++c) {
            const double inv_variance =
                1.0 / (sigma_[c] * sigma_[c]);

            for (std::size_t k = 0; k < count_; ++k) {
                const double center = input[4*k+c];

                if (!std::isfinite(center))
                    throw std::invalid_argument(
                        "Nonfinite covariance input");

                double value;

                if (count_ == 1) {
                    value = center;
                }
                else if (k == 0) {
                    value = (
                        center - rho_*input[4*(k+1)+c]
                    ) / denominator_;
                }
                else if (k + 1 == count_) {
                    value = (
                        center - rho_*input[4*(k-1)+c]
                    ) / denominator_;
                }
                else {
                    value = (
                        (1.0 + rho_*rho_)*center
                        - rho_*input[4*(k-1)+c]
                        - rho_*input[4*(k+1)+c]
                    ) / denominator_;
                }

                output[4*k+c] = value*inv_variance;

                if (!std::isfinite(output[4*k+c]))
                    throw std::runtime_error(
                        "Nonfinite inverse covariance result");
            }
        }

        return output;
    }

    [[nodiscard]]
    double quadratic_form(
        std::span<const double> input) const
    {
        const auto weighted = apply_inverse(input);

        double result = 0.0;

        for (std::size_t i = 0; i < input.size(); ++i)
            result += input[i]*weighted[i];

        if (!std::isfinite(result))
            throw std::runtime_error(
                "Nonfinite covariance quadratic form");

        return result;
    }

private:
    static double correlation_time_safe(double value)
    {
        return value == 0.0 ? 1.0 : value;
    }

    std::size_t count_;
    std::array<double, 4> sigma_;
    double rho_ = 0.0;
    double denominator_ = 1.0;
};

} // namespace state_finder
