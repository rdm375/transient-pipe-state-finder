#pragma once

#include "state_finder/forward_model.hpp"
#include "state_finder/observation_adjoint.hpp"
#include "state_finder/observations.hpp"
#include "state_finder/quadratic_regularizer.hpp"
#include "state_finder/scaled_initial_state.hpp"
#include "state_finder/tangent_linear.hpp"
#include "state_finder/trajectory.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace state_finder {

class GaussNewtonOperator {
public:
    GaussNewtonOperator(
        const ForwardConfiguration& config,
        const Trajectory& trajectory,
        const Observations& observations,
        const QuadraticRegularizer& regularizer,
        InitialStateScales scales = {},
        double damping = 0.0)
        : config_(config),
          trajectory_(trajectory),
          observations_(observations),
          regularizer_(regularizer),
          scales_(scales),
          damping_(damping),
          zero_(2*config.n+1, 0.0)
    {
        if (config.n < 3 ||
            trajectory.info != 0 ||
            trajectory.n != config.n ||
            trajectory.steps != config.steps ||
            trajectory.completed_steps != config.steps ||
            !(observations.sigma_flow > 0.0) ||
            !(observations.sigma_pressure > 0.0) ||
            !(scales.pressure > 0.0) ||
            !(scales.flow > 0.0) ||
            !std::isfinite(damping) ||
            damping < 0.0)
        {
            throw std::invalid_argument(
                "Invalid Gauss-Newton operator configuration");
        }

        const auto count =
            static_cast<std::size_t>(config.steps+1);

        if (observations.inlet_flow.size() != count ||
            observations.outlet_pressure.size() != count)
        {
            throw std::invalid_argument(
                "Gauss-Newton observation dimension mismatch");
        }

        regularizer_zero_gradient_ =
            regularizer_.evaluate(zero_).gradient;
    }

    std::vector<double> apply(
        const std::vector<double>& direction) const
    {
        if (direction.size() != zero_.size())
            throw std::invalid_argument(
                "Gauss-Newton direction dimension mismatch");

        const auto tangent = evaluate_tangent_linear(
            config_, trajectory_, direction, scales_);

        const auto count =
            static_cast<std::size_t>(config_.steps+1);

        if (tangent.inlet_flow.size() != count ||
            tangent.outlet_pressure.size() != count)
        {
            throw std::runtime_error(
                "Incomplete tangent-linear response");
        }

        std::vector<double> forcing(2*count, 0.0);

        const int start_step =
            observations_.include_initial ? 0 : 1;

        for (int k = start_step; k <= config_.steps; ++k) {
            forcing[2*k] =
                tangent.inlet_flow[k] /
                observations_.sigma_flow;

            forcing[2*k+1] =
                tangent.outlet_pressure[k] /
                observations_.sigma_pressure;
        }

        const auto adjoint = apply_observation_adjoint(
            config_,
            trajectory_,
            forcing,
            start_step,
            observations_.sigma_flow,
            observations_.sigma_pressure);

        auto result = scale_adjoint_gradient(
            adjoint.gradient,
            static_cast<std::size_t>(config_.n),
            scales_);

        const auto penalty = regularizer_.evaluate(direction);

        for (std::size_t j = 0; j < result.size(); ++j) {
            result[j] +=
                penalty.gradient[j] -
                regularizer_zero_gradient_[j] +
                damping_*direction[j];

            if (!std::isfinite(result[j]))
                throw std::runtime_error(
                    "Nonfinite Gauss-Newton operator result");
        }

        return result;
    }

private:
    const ForwardConfiguration& config_;
    const Trajectory& trajectory_;
    const Observations& observations_;
    const QuadraticRegularizer& regularizer_;

    InitialStateScales scales_;
    double damping_;

    std::vector<double> zero_;
    std::vector<double> regularizer_zero_gradient_;
};

} // namespace state_finder
