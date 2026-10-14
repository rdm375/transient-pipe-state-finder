#pragma once

#include "state_finder/correlated_joint_gauss_newton.hpp"
#include "state_finder/pressure_calibration.hpp"

#include <array>
#include <cmath>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

namespace state_finder {

// Calibration coordinates are in kPa.
// The covariance is expressed in kPa^2.
struct PressureCalibrationPrior {
    double sigma_kpa = 2.0;
    double correlation = 0.0;

    void validate() const
    {
        if (!std::isfinite(sigma_kpa) ||
            sigma_kpa <= 0.0 ||
            !std::isfinite(correlation) ||
            std::abs(correlation) >= 1.0)
            throw std::invalid_argument(
                "Invalid pressure calibration prior");
    }

    [[nodiscard]]
    std::array<double, 2> apply_precision(
        std::span<const double> b) const
    {
        validate();

        if (b.size() != 2)
            throw std::invalid_argument(
                "Calibration prior dimension mismatch");

        const double factor =
            1.0 /
            (sigma_kpa*sigma_kpa*
             (1.0-correlation*correlation));

        return {
            factor*(b[0]-correlation*b[1]),
            factor*(b[1]-correlation*b[0])
        };
    }

    [[nodiscard]]
    double quadratic_form(
        std::span<const double> b) const
    {
        const auto weighted = apply_precision(b);

        return b[0]*weighted[0]+b[1]*weighted[1];
    }
};

inline std::size_t calibrated_joint_dimension(
    const ForwardConfiguration& configuration)
{
    return joint_variable_dimension(
        configuration.n, configuration.steps) + 2;
}

// The observation adapter takes physical pascals.
// The augmented optimization coordinates use kPa.
inline PressureCalibrationBias calibration_from_kpa(
    std::span<const double> b)
{
    if (b.size() != 2)
        throw std::invalid_argument(
            "Calibration coordinate dimension mismatch");

    return {
        1000.0*b[0],
        1000.0*b[1]
    };
}

inline std::vector<double> calibration_response_kpa(
    std::span<const double> b,
    std::size_t measurement_dimension)
{
    return apply_pressure_calibration(
        std::vector<double>(measurement_dimension, 0.0),
        calibration_from_kpa(b));
}

inline OptimizationEvaluation
evaluate_calibrated_joint_objective(
    const ForwardConfiguration& configuration,
    const BoundaryObservationRecord& observations,
    const JointInverseVariables& baseline,
    const std::vector<double>& z,
    const TemporalMeasurementCovariance& covariance,
    const PressureCalibrationPrior& prior,
    JointVariableScales scales,
    bool need_gradient)
{
    OptimizationEvaluation result;

    try {
        prior.validate();

        const auto state_dimension =
            joint_variable_dimension(
                configuration.n,
                configuration.steps);

        if (z.size() != state_dimension+2)
            throw std::invalid_argument(
                "Augmented objective dimension mismatch");

        const std::vector<double> x(
            z.begin(),
            z.begin()+state_dimension);

        const std::span<const double> b(
            z.data()+state_dimension, 2);

        const auto variables =
            unpack_joint_variables(x, baseline, scales);

        const auto trajectory =
            record_joint_trajectory(configuration, variables);

        const FourChannelPrediction prediction{
            trajectory.predicted
        };

        const auto measurement =
            evaluate_four_channel_objective(
                observations, prediction);

        const auto residual =
            apply_pressure_calibration(
                measurement.residuals,
                calibration_from_kpa(b));

        if (residual.size() != covariance.dimension())
            throw std::invalid_argument(
                "Augmented covariance dimension mismatch");

        const auto weighted =
            covariance.apply_inverse(residual);

        result.objective =
            0.5*covariance.quadratic_form(residual)
            +0.5*prior.quadratic_form(b);

        if (need_gradient) {
            const auto adjoint =
                apply_joint_observation_adjoint(
                    configuration,
                    trajectory,
                    weighted,
                    observations,
                    scales);

            result.gradient = adjoint.gradient;

            if (result.gradient.size() != state_dimension)
                throw std::runtime_error(
                    "Augmented state gradient mismatch");

            const auto calibration_gradient_pa =
                pressure_calibration_transpose(weighted);

            const auto prior_gradient =
                prior.apply_precision(b);

            result.gradient.push_back(
                1000.0*calibration_gradient_pa[0]
                +prior_gradient[0]);

            result.gradient.push_back(
                1000.0*calibration_gradient_pa[1]
                +prior_gradient[1]);
        }

        result.valid = std::isfinite(result.objective);

        if (need_gradient) {
            for (double value : result.gradient)
                result.valid =
                    result.valid && std::isfinite(value);
        }
    }
    catch (const std::exception&) {
        result.valid = false;
    }

    return result;
}

inline LinearOperator
make_calibrated_joint_gauss_newton_operator(
    const ForwardConfiguration& configuration,
    const BoundaryObservationRecord& observations,
    const JointInverseVariables& baseline,
    const std::vector<double>& z,
    double damping,
    const TemporalMeasurementCovariance& covariance,
    const PressureCalibrationPrior& prior,
    JointVariableScales scales = {})
{
    prior.validate();

    if (!std::isfinite(damping) || damping < 0.0)
        throw std::invalid_argument(
            "Invalid augmented damping");

    const auto state_dimension =
        joint_variable_dimension(
            configuration.n, configuration.steps);

    if (z.size() != state_dimension+2)
        throw std::invalid_argument(
            "Augmented operator dimension mismatch");

    const auto first =
        observations.include_initial ? 0 : 1;

    const auto time_count =
        static_cast<std::size_t>(
            configuration.steps+1-first);

    if (covariance.dimension() != 4*time_count)
        throw std::invalid_argument(
            "Augmented operator covariance mismatch");

    const std::vector<double> x(
        z.begin(), z.begin()+state_dimension);

    const auto variables =
        unpack_joint_variables(x, baseline, scales);

    auto trajectory = std::make_shared<JointTrajectory>(
        record_joint_trajectory(configuration, variables));

    return [
        configuration,
        observations,
        covariance,
        prior,
        scales,
        damping,
        state_dimension,
        trajectory = std::move(trajectory)
    ](const std::vector<double>& direction)
        -> std::vector<double>
    {
        if (direction.size() != state_dimension+2)
            throw std::invalid_argument(
                "Augmented operator direction mismatch");

        const std::span<const double> state_direction(
            direction.data(), state_dimension);

        const std::span<const double> bias_direction(
            direction.data()+state_dimension, 2);

        const auto tangent =
            evaluate_joint_tangent_linear(
                configuration,
                *trajectory,
                state_direction,
                scales);

        auto response =
            normalized_joint_tangent_residuals(
                tangent, observations);

        const auto bias_response =
            calibration_response_kpa(
                bias_direction, response.size());

        for (std::size_t i = 0; i < response.size(); ++i)
            response[i] += bias_response[i];

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

        if (result.size() != state_dimension)
            throw std::runtime_error(
                "Augmented operator output mismatch");

        const auto calibration_gradient_pa =
            pressure_calibration_transpose(weighted);

        const auto prior_gradient =
            prior.apply_precision(bias_direction);

        result.push_back(
            1000.0*calibration_gradient_pa[0]
            +prior_gradient[0]);

        result.push_back(
            1000.0*calibration_gradient_pa[1]
            +prior_gradient[1]);

        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] += damping*direction[i];

        return result;
    };
}

inline GaussNewtonOperatorFactory
make_calibrated_joint_gauss_newton_factory(
    ForwardConfiguration configuration,
    BoundaryObservationRecord observations,
    JointInverseVariables baseline,
    TemporalMeasurementCovariance covariance,
    PressureCalibrationPrior prior,
    JointVariableScales scales = {})
{
    return [
        configuration,
        observations,
        baseline,
        covariance,
        prior,
        scales
    ](
        const std::vector<double>& z,
        double damping
    ) {
        return make_calibrated_joint_gauss_newton_operator(
            configuration,
            observations,
            baseline,
            z,
            damping,
            covariance,
            prior,
            scales);
    };
}

} // namespace state_finder
