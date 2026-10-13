#include "state_finder/joint_regularization_gradient.hpp"

#include <cmath>
#include <stdexcept>

namespace state_finder {

std::vector<double> joint_regularization_gradient(
    int n,
    int steps,
    std::span<const double> x,
    JointRegularization regularization)
{
    const auto dimension =
        joint_variable_dimension(n, steps);

    if (x.size() != dimension)
        throw std::invalid_argument(
            "Joint regularization dimension mismatch");

    const double alpha = regularization.initial_weight;
    const double beta = regularization.boundary_weight;
    const double gamma = regularization.boundary_smoothness;

    if (!std::isfinite(alpha) || alpha < 0.0 ||
        !std::isfinite(beta) || beta < 0.0 ||
        !std::isfinite(gamma) || gamma < 0.0)
        throw std::invalid_argument(
            "Invalid joint regularization weight");

    for (double value : x) {
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Nonfinite joint optimization variable");
    }

    const std::size_t state_count =
        static_cast<std::size_t>(2*n+1);

    const std::size_t count =
        static_cast<std::size_t>(steps+1);

    std::vector<double> gradient(dimension, 0.0);

    // Initial-state quadratic regularization.
    for (std::size_t i = 0; i < state_count; ++i)
        gradient[i] = alpha*x[i];

    // Boundary-history quadratic regularization.
    for (std::size_t i = state_count; i < dimension; ++i)
        gradient[i] = beta*x[i];

    // First-difference smoothness regularization:
    //
    // gamma/2 * sum_k (x[k]-x[k-1])^2
    //
    // Each difference contributes equal and opposite
    // derivatives to its two endpoints.
    const auto add_smoothness = [&](std::size_t offset)
    {
        for (std::size_t k = 1; k < count; ++k) {
            const std::size_t left = offset+k-1;
            const std::size_t right = offset+k;

            const double difference =
                x[right]-x[left];

            gradient[left] -= gamma*difference;
            gradient[right] += gamma*difference;
        }
    };

    add_smoothness(state_count);
    add_smoothness(state_count+count);

    for (double value : gradient) {
        if (!std::isfinite(value))
            throw std::runtime_error(
                "Nonfinite regularization gradient");
    }

    return gradient;
}

} // namespace state_finder
