#include "state_finder/joint_scaled_problem.hpp"
#include "state_finder/joint_adjoint.hpp"
#include "state_finder/joint_regularization_gradient.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace state_finder {
namespace {

void require_scale(double value)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument("Invalid joint variable scale");
}

void validate_scales(const JointVariableScales& s)
{
    require_scale(s.initial.pressure);
    require_scale(s.initial.flow);
    require_scale(s.inlet_pressure);
    require_scale(s.outlet_flow);
}

void validate_weight(double value)
{
    if (!std::isfinite(value) || value < 0.0)
        throw std::invalid_argument("Invalid joint regularization weight");
}

void validate_variables(
    const JointInverseVariables& v,
    int n,
    int steps)
{
    const auto count = static_cast<std::size_t>(steps + 1);

    if (v.initial.pressure.size() !=
            static_cast<std::size_t>(n + 1) ||
        v.initial.flow.size() !=
            static_cast<std::size_t>(n) ||
        v.boundary.inlet_pressure.size() != count ||
        v.boundary.outlet_flow.size() != count)
        throw std::invalid_argument(
            "Invalid joint variable dimensions");
}

} // namespace

std::size_t joint_variable_dimension(int n, int steps)
{
    if (n < 2 || steps < 0)
        throw std::invalid_argument("Invalid joint dimensions");

    return static_cast<std::size_t>(2*n + 2*steps + 3);
}

std::vector<double> pack_joint_variables(
    const JointInverseVariables& variables,
    const JointInverseVariables& baseline,
    JointVariableScales scales)
{
    validate_scales(scales);

    const int n = static_cast<int>(baseline.initial.flow.size());
    const int steps =
        static_cast<int>(baseline.boundary.inlet_pressure.size()) - 1;

    joint_variable_dimension(n, steps);
    validate_variables(variables, n, steps);
    validate_variables(baseline, n, steps);

    auto x = pack_scaled_state(
        variables.initial,
        baseline.initial,
        scales.initial);

    const auto count = static_cast<std::size_t>(steps + 1);

    for (std::size_t k = 0; k < count; ++k)
        x.push_back(
            (variables.boundary.inlet_pressure[k] -
             baseline.boundary.inlet_pressure[k]) /
            scales.inlet_pressure);

    for (std::size_t k = 0; k < count; ++k)
        x.push_back(
            (variables.boundary.outlet_flow[k] -
             baseline.boundary.outlet_flow[k]) /
            scales.outlet_flow);

    return x;
}

JointInverseVariables unpack_joint_variables(
    const std::vector<double>& x,
    const JointInverseVariables& baseline,
    JointVariableScales scales)
{
    validate_scales(scales);

    const int n = static_cast<int>(baseline.initial.flow.size());
    const int steps =
        static_cast<int>(baseline.boundary.inlet_pressure.size()) - 1;

    if (x.size() != joint_variable_dimension(n, steps))
        throw std::invalid_argument(
            "Incorrect joint optimization dimension");

    validate_variables(baseline, n, steps);

    for (double value : x) {
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Nonfinite joint optimization variable");
    }

    const std::size_t state_count =
        static_cast<std::size_t>(2*n + 1);

    const std::size_t count =
        static_cast<std::size_t>(steps + 1);

    const std::vector<double> state_x(
        x.begin(), x.begin() + state_count);

    JointInverseVariables result = baseline;

    result.initial = unpack_scaled_state(
        state_x, baseline.initial, scales.initial);

    for (std::size_t k = 0; k < count; ++k) {
        result.boundary.inlet_pressure[k] =
            baseline.boundary.inlet_pressure[k] +
            scales.inlet_pressure*x[state_count + k];

        result.boundary.outlet_flow[k] =
            baseline.boundary.outlet_flow[k] +
            scales.outlet_flow*x[state_count + count + k];
    }

    // One physical inlet pressure at t=0.
    result.initial.pressure[0] =
        result.boundary.inlet_pressure[0];

    return result;
}

JointScaledProblem::JointScaledProblem(
    ForwardConfiguration configuration,
    BoundaryObservationRecord observations,
    JointInverseVariables baseline,
    JointVariableScales scales,
    JointRegularization regularization)
    : configuration_(std::move(configuration)),
      observations_(std::move(observations)),
      baseline_(std::move(baseline)),
      scales_(scales),
      regularization_(regularization)
{
    validate_scales(scales_);

    validate_weight(regularization_.initial_weight);
    validate_weight(regularization_.boundary_weight);
    validate_weight(regularization_.boundary_smoothness);

    joint_variable_dimension(
        configuration_.n, configuration_.steps);

    validate_variables(
        baseline_, configuration_.n, configuration_.steps);

    validate_boundary_observations(
        observations_,
        static_cast<std::size_t>(configuration_.steps + 1));
}

JointObjectiveBreakdown JointScaledProblem::evaluate(
    const std::vector<double>& x) const
{
    const auto variables =
        unpack_joint_variables(x, baseline_, scales_);

    const auto objective =
        evaluate_joint_inverse_objective(
            configuration_, variables, observations_);

    JointObjectiveBreakdown result;

    result.measurement = objective.measurement.total;

    const std::size_t state_count =
        static_cast<std::size_t>(2*configuration_.n + 1);

    const std::size_t count =
        static_cast<std::size_t>(configuration_.steps + 1);

    double state_squared = 0.0;
    double boundary_squared = 0.0;
    double smoothness_squared = 0.0;

    for (std::size_t i = 0; i < state_count; ++i)
        state_squared += x[i]*x[i];

    for (std::size_t i = state_count; i < x.size(); ++i)
        boundary_squared += x[i]*x[i];

    // First differences penalize rapid changes in
    // boundary corrections, not the measured trajectories.
    for (std::size_t k = 1; k < count; ++k) {
        const double dp =
            x[state_count+k] - x[state_count+k-1];

        const double dm =
            x[state_count+count+k] -
            x[state_count+count+k-1];

        smoothness_squared += dp*dp + dm*dm;
    }

    result.initial_regularization =
        0.5 * regularization_.initial_weight *
        state_squared;

    result.boundary_regularization =
        0.5 * regularization_.boundary_weight *
        boundary_squared;

    result.boundary_smoothness =
        0.5 * regularization_.boundary_smoothness *
        smoothness_squared;

    result.total =
        result.measurement +
        result.initial_regularization +
        result.boundary_regularization +
        result.boundary_smoothness;

    if (!std::isfinite(result.total))
        throw std::runtime_error(
            "Nonfinite joint regularized objective");

    return result;
}

std::vector<double> JointScaledProblem::finite_difference_gradient(
    const std::vector<double>& x,
    double relative_step) const
{
    if (!std::isfinite(relative_step) || relative_step <= 0.0)
        throw std::invalid_argument(
            "Invalid finite-difference step");

    const double base = evaluate(x).total;

    std::vector<double> gradient(x.size());

    auto trial = x;

    for (std::size_t i = 0; i < x.size(); ++i) {
        const double h =
            relative_step * std::max(1.0, std::abs(x[i]));

        trial[i] = x[i] + h;

        bool plus_valid = false;
        double plus = 0.0;

        try {
            plus = evaluate(trial).total;
            plus_valid = true;
        }
        catch (const std::exception&) {
        }

        trial[i] = x[i] - h;

        bool minus_valid = false;
        double minus = 0.0;

        try {
            minus = evaluate(trial).total;
            minus_valid = true;
        }
        catch (const std::exception&) {
        }

        trial[i] = x[i];

        if (plus_valid && minus_valid)
            gradient[i] = (plus-minus)/(2.0*h);
        else if (plus_valid)
            gradient[i] = (plus-base)/h;
        else if (minus_valid)
            gradient[i] = (base-minus)/h;
        else
            throw std::runtime_error(
                "Both finite-difference perturbations failed");
    }

    return gradient;
}

OptimizationEvaluation JointScaledProblem::evaluate_for_optimizer(
    const std::vector<double>& x,
    bool need_gradient) const
{
    OptimizationEvaluation result;

    try {
        // Preserve the established objective evaluation.
        result.objective = evaluate(x).total;

        if (need_gradient) {
            const auto variables =
                unpack_joint_variables(x, baseline_, scales_);

            const auto adjoint =
                evaluate_joint_adjoint(
                    configuration_,
                    variables,
                    observations_,
                    scales_);

            const auto regularization_gradient =
                joint_regularization_gradient(
                    configuration_.n,
                    configuration_.steps,
                    x,
                    regularization_);

            result.gradient = adjoint.gradient;

            if (result.gradient.size() !=
                regularization_gradient.size())
                throw std::runtime_error(
                    "Joint gradient dimension mismatch");

            for (std::size_t i = 0;
                 i < result.gradient.size(); ++i) {
                result.gradient[i] +=
                    regularization_gradient[i];

                if (!std::isfinite(result.gradient[i]))
                    throw std::runtime_error(
                        "Nonfinite joint objective gradient");
            }
        }

        result.valid = true;
    }
    catch (const std::exception&) {
        result.valid = false;
    }

    return result;
}

} // namespace state_finder
