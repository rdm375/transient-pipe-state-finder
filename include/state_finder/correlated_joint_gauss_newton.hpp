#pragma once

#include "state_finder/gauss_newton_optimizer.hpp"
#include "state_finder/four_channel_objective.hpp"
#include "state_finder/joint_adjoint.hpp"
#include "state_finder/joint_tangent_linear.hpp"
#include "state_finder/joint_trajectory.hpp"
#include "state_finder/temporal_measurement_covariance.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace state_finder {

// Correlated measurement-only Gauss-Newton operator:
//
// H*v = S^T*C^-1*S*v + damping*v.
//
// No regularization is silently introduced.
//
// A physically justified regularization Hessian can be
// added explicitly in a subsequent implementation.
inline LinearOperator make_correlated_joint_gauss_newton_operator(
    const ForwardConfiguration& configuration,
    const BoundaryObservationRecord& observations,
    const JointInverseVariables& baseline,
    const std::vector<double>& x,
    double damping,
    const TemporalMeasurementCovariance& covariance,
    JointVariableScales scales = {})
{
    if (!std::isfinite(damping) || damping < 0.0)
        throw std::invalid_argument(
            "Invalid correlated Gauss-Newton damping");

    const auto dimension = joint_variable_dimension(
        configuration.n,
        configuration.steps);

    if (x.size() != dimension)
        throw std::invalid_argument(
            "Correlated Gauss-Newton dimension mismatch");

    const auto first =
        observations.include_initial ? 0 : 1;

    const auto time_count =
        static_cast<std::size_t>(
            configuration.steps + 1 - first);

    if (covariance.dimension() != 4*time_count)
        throw std::invalid_argument(
            "Correlated covariance observation mismatch");

    const auto variables =
        unpack_joint_variables(x, baseline, scales);

    auto trajectory = std::make_shared<JointTrajectory>(
        record_joint_trajectory(configuration, variables));

    return [
        configuration,
        observations,
        covariance,
        scales,
        damping,
        dimension,
        trajectory = std::move(trajectory)
    ](const std::vector<double>& direction)
        -> std::vector<double>
    {
        if (direction.size() != dimension)
            throw std::invalid_argument(
                "Correlated operator direction mismatch");

        const auto tangent =
            evaluate_joint_tangent_linear(
                configuration,
                *trajectory,
                direction,
                scales);

        const auto response =
            normalized_joint_tangent_residuals(
                tangent,
                observations);

        const auto weighted =
            covariance.apply_inverse(response);

        const auto adjoint =
            apply_joint_observation_adjoint(
                configuration,
                *trajectory,
                weighted,
                observations,
                scales);

        auto result = adjoint.gradient;

        if (result.size() != dimension)
            throw std::runtime_error(
                "Correlated operator output mismatch");

        for (std::size_t i = 0; i < dimension; ++i)
            result[i] += damping*direction[i];

        return result;
    };
}


// Nonlinear correlated measurement objective and gradient.
//
// The prediction is obtained from the recorded nonlinear
// trajectory. The existing four-channel objective supplies
// the engineering-normalized residual vector.
//
// IMPORTANT:
// We use its residual vector, not its independently
// weighted scalar objective.
inline OptimizationEvaluation
evaluate_correlated_joint_objective(
    const ForwardConfiguration& configuration,
    const BoundaryObservationRecord& observations,
    const JointInverseVariables& baseline,
    const std::vector<double>& x,
    const TemporalMeasurementCovariance& covariance,
    JointVariableScales scales,
    bool need_gradient)
{
    OptimizationEvaluation result;

    try {
        const auto variables =
            unpack_joint_variables(x, baseline, scales);

        const auto trajectory =
            record_joint_trajectory(configuration, variables);

        const FourChannelPrediction prediction{
            trajectory.predicted
        };

        const auto measurement =
            evaluate_four_channel_objective(
                observations,
                prediction);

        const auto& residual = measurement.residuals;

        if (residual.size() != covariance.dimension())
            throw std::invalid_argument(
                "Correlated objective covariance dimension mismatch");

        const auto weighted =
            covariance.apply_inverse(residual);

        result.objective =
            0.5*covariance.quadratic_form(residual);

        if (need_gradient) {
            const auto adjoint =
                apply_joint_observation_adjoint(
                    configuration,
                    trajectory,
                    weighted,
                    observations,
                    scales);

            result.gradient = adjoint.gradient;
        }

        result.valid = std::isfinite(result.objective);
    }
    catch (const std::exception&) {
        result.valid = false;
    }

    return result;
}


// Adapter for the existing matrix-free Gauss-Newton optimizer.
//
// The covariance is captured by value and reused across
// nonlinear iterations. Each operator construction records
// its own trajectory at the current optimization point.
inline GaussNewtonOperatorFactory
make_correlated_joint_gauss_newton_factory(
    ForwardConfiguration configuration,
    BoundaryObservationRecord observations,
    JointInverseVariables baseline,
    TemporalMeasurementCovariance covariance,
    JointVariableScales scales = {})
{
    return [
        configuration = std::move(configuration),
        observations = std::move(observations),
        baseline = std::move(baseline),
        covariance = std::move(covariance),
        scales
    ](
        const std::vector<double>& x,
        double damping
    ) {
        return make_correlated_joint_gauss_newton_operator(
            configuration,
            observations,
            baseline,
            x,
            damping,
            covariance,
            scales);
    };
}

} // namespace state_finder
