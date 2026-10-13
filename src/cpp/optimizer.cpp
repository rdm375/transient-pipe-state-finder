#include "state_finder/optimizer.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace state_finder {
namespace {

using Vector = std::vector<double>;

struct Correction {
    Vector s;
    Vector y;
    double rho = 0.0;
};

double dot(const Vector& a, const Vector& b)
{
    if (a.size() != b.size())
        throw std::invalid_argument("L-BFGS vector size mismatch");

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i] * b[i];

    return result;
}

double norm(const Vector& x)
{
    return std::sqrt(dot(x, x));
}

bool finite(const Vector& x)
{
    for (double value : x) {
        if (!std::isfinite(value))
            return false;
    }

    return true;
}

Vector add_scaled(
    const Vector& a,
    const Vector& b,
    double scale)
{
    if (a.size() != b.size())
        throw std::invalid_argument("L-BFGS vector size mismatch");

    Vector result(a.size());

    for (std::size_t i = 0; i < a.size(); ++i)
        result[i] = a[i] + scale * b[i];

    return result;
}

Vector subtract(const Vector& a, const Vector& b)
{
    return add_scaled(a, b, -1.0);
}

Vector lbfgs_direction(
    const Vector& gradient,
    const std::deque<Correction>& corrections)
{
    Vector q = gradient;

    const std::size_t m = corrections.size();
    Vector alpha(m, 0.0);

    for (std::size_t j = m; j-- > 0;) {
        const auto& c = corrections[j];

        alpha[j] = c.rho * dot(c.s, q);
        q = add_scaled(q, c.y, -alpha[j]);
    }

    double gamma = 1.0;

    if (!corrections.empty()) {
        const auto& c = corrections.back();

        const double yy = dot(c.y, c.y);
        const double sy = dot(c.s, c.y);

        if (yy > 0.0 && sy > 0.0)
            gamma = sy / yy;
    }

    Vector r = q;

    for (double& value : r)
        value *= gamma;

    for (std::size_t j = 0; j < m; ++j) {
        const auto& c = corrections[j];

        const double beta = c.rho * dot(c.y, r);

        r = add_scaled(
            r,
            c.s,
            alpha[j] - beta);
    }

    for (double& value : r)
        value = -value;

    return r;
}

void validate_options(const OptimizationOptions& o)
{
    if (o.max_iterations < 0 ||
        o.history_size < 1 ||
        o.max_line_search_steps < 1 ||
        !(o.gradient_tolerance > 0.0) ||
        !(o.relative_objective_tolerance >= 0.0) ||
        !(o.armijo_c1 > 0.0 && o.armijo_c1 < 1.0) ||
        !(o.backtracking_factor > 0.0 &&
          o.backtracking_factor < 1.0) ||
        !(o.minimum_step > 0.0) ||
        !(o.curvature_tolerance >= 0.0))
    {
        throw std::invalid_argument(
            "Invalid L-BFGS optimization options");
    }
}

} // namespace

OptimizationResult minimize_lbfgs(
    const ObjectiveFunction& objective,
    const Vector& initial_variables,
    const OptimizationOptions& options)
{
    validate_options(options);

    if (initial_variables.empty() ||
        !finite(initial_variables))
    {
        throw std::invalid_argument(
            "Invalid initial optimization variables");
    }

    OptimizationResult result;
    result.variables = initial_variables;

    auto evaluate = [&](const Vector& x, bool gradient) {
        auto evaluation = objective(x, gradient);

        ++result.objective_evaluations;

        if (gradient)
            ++result.gradient_evaluations;

        if (!evaluation.valid ||
            !std::isfinite(evaluation.objective))
        {
            evaluation.valid = false;
            return evaluation;
        }

        if (gradient &&
            (evaluation.gradient.size() != x.size() ||
             !finite(evaluation.gradient)))
        {
            evaluation.valid = false;
        }

        return evaluation;
    };

    auto current = evaluate(result.variables, true);

    if (!current.valid)
        throw std::runtime_error(
            "Initial L-BFGS evaluation failed");

    result.initial_objective = current.objective;
    result.final_objective = current.objective;
    result.gradient = current.gradient;
    result.gradient_norm = norm(current.gradient);

    result.history.push_back({
        0,
        current.objective,
        result.gradient_norm,
        0.0
    });

    std::deque<Correction> corrections;

    for (int iteration = 0;
         iteration < options.max_iterations;
         ++iteration)
    {
        if (result.gradient_norm <=
            options.gradient_tolerance)
        {
            result.converged = true;
            result.reason = "Gradient tolerance satisfied";
            return result;
        }

        Vector direction =
            lbfgs_direction(current.gradient, corrections);

        double slope = dot(current.gradient, direction);

        // Guard against a non-descent direction.
        if (!finite(direction) ||
            !std::isfinite(slope) ||
            !(slope < 0.0))
        {
            corrections.clear();

            direction = current.gradient;

            for (double& value : direction)
                value = -value;

            slope = dot(current.gradient, direction);
        }

        if (!(slope < 0.0) || !std::isfinite(slope)) {
            result.reason = "No valid descent direction";
            return result;
        }

        double step = 1.0;
        bool accepted = false;

        Vector trial_variables;
        OptimizationEvaluation trial;

        for (int ls = 0;
             ls < options.max_line_search_steps;
             ++ls)
        {
            if (step < options.minimum_step)
                break;

            trial_variables = add_scaled(
                result.variables,
                direction,
                step);

            if (finite(trial_variables)) {
                trial = evaluate(trial_variables, false);

                if (trial.valid &&
                    trial.objective <=
                        current.objective +
                        options.armijo_c1 * step * slope)
                {
                    accepted = true;
                    break;
                }
            }

            step *= options.backtracking_factor;
        }

        if (!accepted) {
            result.reason = "Armijo line search failed";
            return result;
        }

        // The accepted point must also support an adjoint
        // gradient evaluation. Otherwise it cannot enter
        // the quasi-Newton history.
        auto next = evaluate(trial_variables, true);

        if (!next.valid) {
            result.reason =
                "Gradient evaluation failed at accepted step";
            return result;
        }

        const Vector s = subtract(
            trial_variables,
            result.variables);

        const Vector y = subtract(
            next.gradient,
            current.gradient);

        const double sy = dot(s, y);
        const double ss = dot(s, s);
        const double yy = dot(y, y);

        // Skip unreliable curvature updates.
        if (std::isfinite(sy) &&
            std::isfinite(ss) &&
            std::isfinite(yy) &&
            ss > 0.0 &&
            yy > 0.0 &&
            sy > options.curvature_tolerance *
                     std::sqrt(ss * yy))
        {
            corrections.push_back({
                s,
                y,
                1.0 / sy
            });

            if (corrections.size() >
                static_cast<std::size_t>(
                    options.history_size))
            {
                corrections.pop_front();
            }
        }

        const double previous_objective =
            current.objective;

        result.variables = std::move(trial_variables);
        current = std::move(next);

        result.iterations = iteration + 1;
        result.final_objective = current.objective;
        result.gradient = current.gradient;
        result.gradient_norm = norm(current.gradient);

        result.history.push_back({
            result.iterations,
            result.final_objective,
            result.gradient_norm,
            step
        });

        if (result.gradient_norm <=
            options.gradient_tolerance)
        {
            result.converged = true;
            result.reason = "Gradient tolerance satisfied";
            return result;
        }

        const double reduction =
            previous_objective - current.objective;

        const double scale = std::max(
            {1.0,
             std::abs(previous_objective),
             std::abs(current.objective)});

        if (reduction >= 0.0 &&
            reduction <=
                options.relative_objective_tolerance * scale)
        {
            result.converged = true;
            result.reason =
                "Relative objective tolerance satisfied";
            return result;
        }
    }

    result.reason = "Maximum iterations reached";
    return result;
}

} // namespace state_finder
