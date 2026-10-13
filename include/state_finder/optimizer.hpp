#pragma once

#include <functional>
#include <string>
#include <vector>

namespace state_finder {

struct OptimizationOptions {
    int max_iterations = 100;
    int history_size = 10;
    int max_line_search_steps = 30;

    double gradient_tolerance = 1.0e-7;
    double relative_objective_tolerance = 1.0e-12;

    double armijo_c1 = 1.0e-4;
    double backtracking_factor = 0.5;

    double minimum_step = 1.0e-14;
    double curvature_tolerance = 1.0e-12;
};

struct OptimizationEvaluation {
    double objective = 0.0;
    std::vector<double> gradient;

    // False means the trial point is inadmissible or its
    // underlying numerical evaluation failed.
    bool valid = false;
};

using ObjectiveFunction =
    std::function<OptimizationEvaluation(
        const std::vector<double>&,
        bool need_gradient)>;

struct OptimizationIteration {
    int iteration = 0;
    double objective = 0.0;
    double gradient_norm = 0.0;
    double step = 0.0;
};

struct OptimizationResult {
    std::vector<double> variables;
    std::vector<double> gradient;

    double initial_objective = 0.0;
    double final_objective = 0.0;
    double gradient_norm = 0.0;

    int iterations = 0;
    int objective_evaluations = 0;
    int gradient_evaluations = 0;

    bool converged = false;
    std::string reason;

    std::vector<OptimizationIteration> history;
};

// Variables and gradients are expressed in the same
// dimensionless optimization coordinates.
//
// During line search, need_gradient=false. The callback
// may then omit the gradient.
//
// After accepting a step, the callback is invoked with
// need_gradient=true.
OptimizationResult minimize_lbfgs(
    const ObjectiveFunction& objective,
    const std::vector<double>& initial_variables,
    const OptimizationOptions& options = {});

} // namespace state_finder
