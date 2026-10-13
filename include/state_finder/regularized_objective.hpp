#pragma once

#include "state_finder/optimizer.hpp"
#include "state_finder/quadratic_regularizer.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace state_finder {

inline ObjectiveFunction make_regularized_objective(
    ObjectiveFunction observation_objective,
    QuadraticRegularizer regularizer)
{
    if (!observation_objective)
        throw std::invalid_argument(
            "Observation objective callback is empty");

    return [
        observation_objective = std::move(observation_objective),
        regularizer = std::move(regularizer)
    ](
        const std::vector<double>& variables,
        bool need_gradient
    ) -> OptimizationEvaluation {

        auto result = observation_objective(
            variables, need_gradient);

        if (!result.valid)
            return result;

        const auto penalty = regularizer.evaluate(variables);

        result.objective += penalty.total();

        if (!std::isfinite(result.objective)) {
            result.valid = false;
            return result;
        }

        if (need_gradient) {
            if (result.gradient.size() != variables.size())
                throw std::runtime_error(
                    "Observation gradient dimension mismatch");

            for (std::size_t i = 0; i < variables.size(); ++i)
                result.gradient[i] += penalty.gradient[i];
        }

        return result;
    };
}

} // namespace state_finder
