#pragma once

#include "state_finder/matrix_free_cg.hpp"
#include "state_finder/dense_gauss_newton.hpp"
#include "state_finder/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace state_finder {

// Construct a matrix-free, damped GN operator at the
// current nonlinear iterate.
//
// The returned operator must retain ownership of any
// trajectory or linearization data it requires.
using GaussNewtonOperatorFactory =
    std::function<LinearOperator(
        const std::vector<double>& variables,
        double damping)>;

using GaussNewtonPreconditionerFactory =
    std::function<LinearOperator(
        const LinearOperator& A,
        double damping)>;

struct GaussNewtonDiagnostics {
    int linearizations = 0;
    int pcg_solves = 0;
    int pcg_iterations = 0;
    int operator_calls = 0;
    int damping_retries = 0;
    int gradient_fallbacks = 0;
    int dense_solves = 0;
    int dense_operator_calls = 0;

    double maximum_true_residual_ratio = 0.0;
    double final_damping = 0.0;
    double final_forcing_tolerance = 0.0;
};

struct GaussNewtonOptions {
    int max_iterations = 30;
    bool dense_linear_solver = false;
    int max_cg_iterations = 200;
    int max_line_search_steps = 40;
    int max_damping_attempts = 6;

    double gradient_tolerance = 1.0e-8;
    double relative_objective_tolerance = 1.0e-14;

    double cg_relative_tolerance = 1.0e-6;

    // Enable a looser solve far from stationarity.
    bool adaptive_forcing = true;
    double maximum_forcing_tolerance = 1.0e-2;
    double forcing_coefficient = 1.0e-3;
    double cg_absolute_tolerance = 1.0e-12;

    double initial_damping = 1.0e-3;
    double damping_growth = 10.0;
    double damping_reduction = 0.5;

    double armijo_c1 = 1.0e-4;
    double backtracking_factor = 0.5;
    double minimum_step = 1.0e-14;
};

inline OptimizationResult minimize_gauss_newton(
    const ObjectiveFunction& objective,
    const std::vector<double>& initial_variables,
    const GaussNewtonOperatorFactory& make_operator,
    const GaussNewtonOptions& options = {},
    GaussNewtonDiagnostics* diagnostics = nullptr,
    const GaussNewtonPreconditionerFactory&
        make_preconditioner = {})
{
    if (!objective || !make_operator)
        throw std::invalid_argument(
            "Missing Gauss-Newton callback");

    if (initial_variables.empty() ||
        options.max_iterations < 1 ||
        options.max_cg_iterations < 1 ||
        options.max_line_search_steps < 1 ||
        options.max_damping_attempts < 1 ||
        !(options.initial_damping > 0.0) ||
        !(options.damping_growth > 1.0) ||
        !(options.damping_reduction > 0.0) ||
        !(options.damping_reduction <= 1.0) ||
        !(options.backtracking_factor > 0.0) ||
        !(options.backtracking_factor < 1.0) ||
        !(options.armijo_c1 > 0.0) ||
        !(options.armijo_c1 < 1.0))
    {
        throw std::invalid_argument(
            "Invalid Gauss-Newton options");
    }

    if (diagnostics)
        *diagnostics = {};

    OptimizationResult result;
    result.variables = initial_variables;

    const auto norm = [](const std::vector<double>& x) {
        return cg_norm(x);
    };

    auto current = objective(result.variables, true);

    ++result.objective_evaluations;
    ++result.gradient_evaluations;

    if (!current.valid ||
        current.gradient.size() != result.variables.size() ||
        !std::isfinite(current.objective))
    {
        throw std::runtime_error(
            "Invalid initial Gauss-Newton evaluation");
    }

    result.initial_objective = current.objective;

    double damping = options.initial_damping;

    for (int iteration = 0;
         iteration < options.max_iterations;
         ++iteration)
    {
        const double gradient_norm =
            norm(current.gradient);

        if (!std::isfinite(gradient_norm))
            throw std::runtime_error(
                "Nonfinite Gauss-Newton gradient");

        if (gradient_norm <= options.gradient_tolerance) {
            result.converged = true;
            result.reason = "gradient tolerance";
            break;
        }

        std::vector<double> rhs = current.gradient;

        for (double& value : rhs)
            value = -value;

        std::vector<double> direction;
        bool found_direction = false;

        double forcing_tolerance =
            options.cg_relative_tolerance;

        if (options.adaptive_forcing) {
            forcing_tolerance = std::clamp(
                options.forcing_coefficient *
                    std::sqrt(gradient_norm),
                options.cg_relative_tolerance,
                options.maximum_forcing_tolerance);
        }

        if (diagnostics)
            diagnostics->final_forcing_tolerance =
                forcing_tolerance;

        // A new trajectory is constructed by the factory
        // at each damping attempt. This can be optimized
        // later by sharing one trajectory per outer step.
        for (int attempt = 0;
             attempt < options.max_damping_attempts;
             ++attempt)
        {
            const LinearOperator A =
                make_operator(result.variables, damping);

            if (diagnostics) {
                ++diagnostics->pcg_solves;

                if (attempt > 0)
                    ++diagnostics->damping_retries;
            }

            CGOptions cg_options;
            cg_options.max_iterations =
                options.max_cg_iterations;
            cg_options.relative_tolerance =
                forcing_tolerance;
            cg_options.absolute_tolerance =
                options.cg_absolute_tolerance;

            try {
                LinearOperator M_inverse;

                if (make_preconditioner)
                    M_inverse = make_preconditioner(A, damping);

                CGResult cg;

                if (options.dense_linear_solver) {
                    if (diagnostics)
                        ++diagnostics->dense_solves;

                    const auto dense =
                        solve_dense_gauss_newton(A, rhs);

                    cg.solution = dense.solution;
                    cg.converged = true;
                    cg.operator_calls = dense.operator_calls;

                    if (diagnostics)
                        diagnostics->dense_operator_calls +=
                            dense.operator_calls;
                }
                else {
                    cg = solve_matrix_free_cg(
                        A, rhs, cg_options, M_inverse);
                }

                if (diagnostics) {
                    diagnostics->pcg_iterations +=
                        cg.iterations;

                    diagnostics->operator_calls +=
                        cg.operator_calls;
                }

                if (cg.solution.size() != rhs.size()) {
                    throw std::runtime_error(
                        "PCG solution dimension mismatch");
                }

                // Independently recompute the true residual.
                const auto Ax = A(cg.solution);

                if (diagnostics)
                    ++diagnostics->operator_calls;

                if (Ax.size() != rhs.size()) {
                    throw std::runtime_error(
                        "GN operator dimension mismatch");
                }

                std::vector<double> residual(rhs.size());

                for (std::size_t j = 0;
                     j < rhs.size();
                     ++j)
                {
                    residual[j] = rhs[j] - Ax[j];
                }

                const double tolerance =
                    std::max(
                        options.cg_absolute_tolerance,
                        forcing_tolerance * norm(rhs));

                if (diagnostics) {
                    const double ratio =
                        norm(residual) /
                        std::max(
                            norm(rhs),
                            std::numeric_limits<double>::min());

                    if (std::isfinite(ratio)) {
                        diagnostics->maximum_true_residual_ratio =
                            std::max(
                                diagnostics->maximum_true_residual_ratio,
                                ratio);
                    }
                }

                const double directional_derivative =
                    cg_dot(
                        current.gradient,
                        cg.solution);

                if (cg.converged &&
                    norm(residual) <= tolerance &&
                    std::isfinite(directional_derivative) &&
                    directional_derivative < 0.0)
                {
                    direction = cg.solution;
                    found_direction = true;
                    break;
                }
            }
            catch (const std::exception&) {
                // Retry with stronger damping.
            }

            damping *= options.damping_growth;

            if (!std::isfinite(damping))
                break;
        }

        // Guaranteed descent fallback in exact arithmetic.
        if (!found_direction) {
            direction = rhs;

            if (diagnostics)
                ++diagnostics->gradient_fallbacks;
        }

        const double slope =
            cg_dot(current.gradient, direction);

        if (!(slope < 0.0) || !std::isfinite(slope)) {
            result.reason = "no descent direction";
            break;
        }

        const double previous_objective =
            current.objective;

        double step = 1.0;
        bool accepted = false;
        std::vector<double> trial_variables;

        for (int ls = 0;
             ls < options.max_line_search_steps;
             ++ls)
        {
            trial_variables = result.variables;

            for (std::size_t j = 0;
                 j < trial_variables.size();
                 ++j)
            {
                trial_variables[j] +=
                    step*direction[j];
            }

            const auto trial =
                objective(trial_variables, false);

            ++result.objective_evaluations;

            if (trial.valid &&
                std::isfinite(trial.objective) &&
                trial.objective <=
                    previous_objective +
                    options.armijo_c1*step*slope)
            {
                accepted = true;
                break;
            }

            step *= options.backtracking_factor;

            if (step < options.minimum_step)
                break;
        }

        if (!accepted) {
            result.reason = "line search failed";
            break;
        }

        // Evaluate the gradient before committing the
        // accepted trial, preserving a valid current state.
        auto next = objective(trial_variables, true);

        ++result.objective_evaluations;
        ++result.gradient_evaluations;

        if (!next.valid ||
            next.gradient.size() != trial_variables.size() ||
            !std::isfinite(next.objective))
        {
            result.reason = "accepted gradient evaluation failed";
            break;
        }

        result.variables = std::move(trial_variables);
        current = std::move(next);

        ++result.iterations;

        const double new_gradient_norm =
            norm(current.gradient);

        result.history.push_back({
            result.iterations,
            current.objective,
            new_gradient_norm,
            step
        });

        // Successful full steps permit less damping.
        if (found_direction && step == 1.0) {
            damping = std::max(
                options.initial_damping,
                damping*options.damping_reduction);
        }
        else {
            damping *= options.damping_growth;
        }

        const double reduction =
            previous_objective-current.objective;

        const double relative_reduction =
            reduction /
            std::max(1.0, std::abs(previous_objective));

        if (new_gradient_norm <= options.gradient_tolerance) {
            result.converged = true;
            result.reason = "gradient tolerance";
            break;
        }

        if (relative_reduction <=
            options.relative_objective_tolerance)
        {
            result.converged = true;
            result.reason = "relative objective tolerance";
            break;
        }
    }

    if (diagnostics)
        diagnostics->final_damping = damping;

    result.final_objective = current.objective;
    result.gradient = current.gradient;
    result.gradient_norm = norm(current.gradient);

    if (result.reason.empty())
        result.reason = "maximum iterations";

    return result;
}

} // namespace state_finder
