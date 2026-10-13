#pragma once

#include "state_finder/gauss_newton_optimizer.hpp"
#include "state_finder/joint_adjoint.hpp"
#include "state_finder/joint_regularization_gradient.hpp"
#include "state_finder/joint_tangent_linear.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace state_finder {

// Construct H = S^T S + H_reg + damping*I.
//
// S is the noise-normalized four-channel observation
// Jacobian in dimensionless joint optimization coordinates.
//
// The nonlinear trajectory is recorded once per factory
// invocation and shared by all subsequent operator products.
//
// No dense sensitivity matrix is constructed.
inline LinearOperator make_joint_gauss_newton_operator(
    const ForwardConfiguration& configuration,
    const BoundaryObservationRecord& observations,
    const JointInverseVariables& baseline,
    const std::vector<double>& x,
    double damping,
    JointVariableScales scales = {},
    JointRegularization regularization = {})
{
    if (!std::isfinite(damping) || damping < 0.0)
        throw std::invalid_argument(
            "Invalid joint Gauss-Newton damping");

    const int n = configuration.n;
    const int steps = configuration.steps;

    const auto dimension =
        joint_variable_dimension(n, steps);

    if (x.size() != dimension)
        throw std::invalid_argument(
            "Joint Gauss-Newton dimension mismatch");

    // Validate the regularization weights before constructing
    // the nonlinear trajectory.
    const std::vector<double> zero(dimension, 0.0);

    (void)joint_regularization_gradient(
        n, steps, zero, regularization);

    validate_boundary_observations(
        observations,
        static_cast<std::size_t>(steps + 1));

    const auto variables =
        unpack_joint_variables(x, baseline, scales);

    auto trajectory = std::make_shared<JointTrajectory>(
        record_joint_trajectory(configuration, variables));

    return [
        configuration,
        observations,
        scales,
        regularization,
        damping,
        dimension,
        trajectory = std::move(trajectory)
    ](const std::vector<double>& direction)
        -> std::vector<double>
    {
        if (direction.size() != dimension)
            throw std::invalid_argument(
                "Joint GN operator direction mismatch");

        // Forward sensitivity product: S*v.
        const auto tangent =
            evaluate_joint_tangent_linear(
                configuration,
                *trajectory,
                direction,
                scales);

        const auto response =
            normalized_joint_tangent_residuals(
                tangent, observations);

        // Transpose sensitivity product: S^T*(S*v).
        const auto adjoint =
            apply_joint_observation_adjoint(
                configuration,
                *trajectory,
                response,
                observations,
                scales);

        // Exact Hessian action of the quadratic
        // regularization in scaled coordinates.
        const auto reg =
            joint_regularization_gradient(
                configuration.n,
                configuration.steps,
                direction,
                regularization);

        auto result = adjoint.gradient;

        if (result.size() != dimension ||
            reg.size() != dimension)
            throw std::runtime_error(
                "Joint GN internal dimension mismatch");

        for (std::size_t i = 0; i < dimension; ++i) {
            result[i] += reg[i] + damping*direction[i];

            if (!std::isfinite(result[i]))
                throw std::runtime_error(
                    "Nonfinite joint GN operator product");
        }

        return result;
    };
}

// Adapter for the existing generic Gauss-Newton optimizer.
//
// The optimizer supplies the current scaled variables
// and damping at each linearization.
inline GaussNewtonOperatorFactory
make_joint_gauss_newton_factory(
    ForwardConfiguration configuration,
    BoundaryObservationRecord observations,
    JointInverseVariables baseline,
    JointVariableScales scales = {},
    JointRegularization regularization = {})
{
    return [
        configuration = std::move(configuration),
        observations = std::move(observations),
        baseline = std::move(baseline),
        scales,
        regularization
    ](
        const std::vector<double>& x,
        double damping
    ) {
        return make_joint_gauss_newton_operator(
            configuration,
            observations,
            baseline,
            x,
            damping,
            scales,
            regularization);
    };
}

} // namespace state_finder
