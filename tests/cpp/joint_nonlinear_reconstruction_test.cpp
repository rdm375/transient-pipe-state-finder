
#include "state_finder/joint_gauss_newton_operator.hpp"
#include "state_finder/joint_regularization_preconditioner.hpp"
#include "state_finder/joint_scaled_problem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace state_finder;

namespace {

void require(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}

double norm(const std::vector<double>& x)
{
    return cg_norm(x);
}

double distance(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    require(a.size() == b.size(), "Vector size mismatch");

    auto difference = a;

    for (std::size_t i = 0; i < a.size(); ++i)
        difference[i] -= b[i];

    return norm(difference);
}

struct Run {
    OptimizationResult result;
    GaussNewtonDiagnostics diagnostics;
    double seconds = 0.0;
    double truth_error = 0.0;
};

void report(
    const std::string& label,
    const Run& run)
{
    const auto& r = run.result;

    std::cout
        << label
        << " initial=" << r.initial_objective
        << " final=" << r.final_objective
        << " iterations=" << r.iterations
        << " gradient_norm=" << r.gradient_norm
        << " truth_error=" << run.truth_error
        << " seconds=" << run.seconds
        << " converged=" << r.converged
        << " reason=\"" << r.reason << "\""
        << " objective_evaluations="
        << r.objective_evaluations
        << " gradient_evaluations="
        << r.gradient_evaluations
        << " pcg_iterations="
        << run.diagnostics.pcg_iterations
        << " operator_calls="
        << run.diagnostics.operator_calls
        << " damping_retries="
        << run.diagnostics.damping_retries
        << " gradient_fallbacks="
        << run.diagnostics.gradient_fallbacks
        << '\n';
}

}

int main()
{
    try {
        constexpr int n = 6;
        constexpr int steps = 5;

        const double pi = std::acos(-1.0);

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                1000.0, 15.0, 0.65,
                1.0, pi/4.0,
                288.15, 0.9, 500.0,
                1.1e-5, 4.5e-5
            },
            pipe_sim::DemandRamp{
                8.0e6, 100.0, 100.0, 1.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9, 1.0e-12, 30, false
            }
        };

        JointInverseVariables truth;

        truth.initial.pressure.assign(n+1, 8.0e6);
        truth.initial.flow.assign(n, 100.0);
        truth.initial.inlet_flow = 100.0;

        truth.boundary.inlet_pressure.assign(
            steps+1, 8.0e6);

        truth.boundary.outlet_flow.assign(
            steps+1, 100.0);

        // Generate a nonconstant, physically consistent
        // synthetic trajectory.
        truth.initial.pressure[2] += 1500.0;
        truth.initial.pressure[4] -= 800.0;

        truth.initial.flow[1] += 0.25;
        truth.initial.flow[4] -= 0.15;
        truth.initial.inlet_flow += 0.3;

        truth.boundary.inlet_pressure[0] += 500.0;
        truth.boundary.inlet_pressure[2] -= 700.0;
        truth.boundary.inlet_pressure[4] += 400.0;

        truth.boundary.outlet_flow[0] += 0.15;
        truth.boundary.outlet_flow[3] -= 0.25;

        truth.initial.pressure[0] =
            truth.boundary.inlet_pressure[0];

        const auto reference =
            record_joint_trajectory(config, truth);

        BoundaryObservationRecord observations;
        observations.include_initial = true;
        observations.measured = reference.predicted;

        // Baseline differs from truth. Both optimizers
        // start at x=0, representing this baseline.
        JointInverseVariables baseline = truth;

        baseline.initial.pressure[2] += 1000.0;
        baseline.initial.pressure[4] -= 600.0;

        baseline.initial.flow[1] += 0.20;
        baseline.initial.flow[4] -= 0.10;
        baseline.initial.inlet_flow += 0.25;

        baseline.boundary.inlet_pressure[0] += 300.0;
        baseline.boundary.inlet_pressure[2] -= 500.0;
        baseline.boundary.inlet_pressure[4] += 300.0;

        baseline.boundary.outlet_flow[0] += 0.10;
        baseline.boundary.outlet_flow[3] -= 0.20;

        baseline.initial.pressure[0] =
            baseline.boundary.inlet_pressure[0];

        JointVariableScales scales;

        const auto truth_x =
            pack_joint_variables(truth, baseline, scales);

        const std::vector<double> initial(
            joint_variable_dimension(n, steps), 0.0);

        const auto run_case =
            [&](const std::string& name,
                JointRegularization regularization)
        {
            JointScaledProblem problem(
                config,
                observations,
                baseline,
                scales,
                regularization);

            const ObjectiveFunction objective =
                [&problem](
                    const std::vector<double>& x,
                    bool need_gradient)
                {
                    return problem.evaluate_for_optimizer(
                        x, need_gradient);
                };

            const auto factory =
                make_joint_gauss_newton_factory(
                    config,
                    observations,
                    baseline,
                    scales,
                    regularization);

            GaussNewtonOptions gn_options;
            gn_options.max_iterations = 50;
            gn_options.max_cg_iterations = 300;
            gn_options.gradient_tolerance = 1.0e-7;
            gn_options.initial_damping = 1.0e-3;

            OptimizationOptions lbfgs_options;
            lbfgs_options.max_iterations = 150;
            lbfgs_options.gradient_tolerance = 1.0e-7;

            Run gn;
            Run pcgn;
            Run lbfgs;

            const auto clock = [] {
                return std::chrono::steady_clock::now();
            };

            const auto t0 = clock();

            gn.result = minimize_gauss_newton(
                objective,
                initial,
                factory,
                gn_options,
                &gn.diagnostics);

            const auto t1 = clock();

            const auto preconditioner_factory =
                make_joint_regularization_preconditioner_factory(
                    n, steps, regularization);

            pcgn.result = minimize_gauss_newton(
                objective,
                initial,
                factory,
                gn_options,
                &pcgn.diagnostics,
                preconditioner_factory);

            const auto t1b = clock();

            lbfgs.result = minimize_lbfgs(
                objective,
                initial,
                lbfgs_options);

            const auto t2 = clock();

            gn.seconds =
                std::chrono::duration<double>(
                    t1-t0).count();

            pcgn.seconds =
                std::chrono::duration<double>(
                    t1b-t1).count();

            lbfgs.seconds =
                std::chrono::duration<double>(
                    t2-t1b).count();

            gn.truth_error =
                distance(gn.result.variables, truth_x);

            pcgn.truth_error =
                distance(pcgn.result.variables, truth_x);

            lbfgs.truth_error =
                distance(lbfgs.result.variables, truth_x);

            std::cout << "\nCASE " << name << '\n';

            report("GN", gn);
            report("PCGN", pcgn);
            report("LBFGS", lbfgs);

            std::cout
                << "pcg_iteration_ratio="
                << static_cast<double>(
                    pcgn.diagnostics.pcg_iterations) /
                   std::max(
                       1, gn.diagnostics.pcg_iterations)
                << '\n';

            const auto initial_objective =
                problem.evaluate(initial).total;

            require(
                gn.result.final_objective <=
                    initial_objective + 1.0e-10,
                "Gauss-Newton increased objective");

            require(
                pcgn.result.final_objective <=
                    initial_objective + 1.0e-10,
                "Preconditioned GN increased objective");

            require(
                lbfgs.result.final_objective <=
                    initial_objective + 1.0e-10,
                "L-BFGS increased objective");

            require(
                std::isfinite(gn.truth_error) &&
                std::isfinite(pcgn.truth_error) &&
                std::isfinite(lbfgs.truth_error),
                "Nonfinite reconstruction error");

            require(
                gn.result.variables.size() ==
                    initial.size(),
                "GN reconstruction dimension mismatch");

            require(
                pcgn.result.variables.size() ==
                    initial.size(),
                "PCGN reconstruction dimension mismatch");

            require(
                lbfgs.result.variables.size() ==
                    initial.size(),
                "L-BFGS reconstruction dimension mismatch");

            return gn;
        };

        JointRegularization regularized;
        regularized.initial_weight = 1.0e-4;
        regularized.boundary_weight = 1.0e-2;
        regularized.boundary_smoothness = 1.0e-2;

        const auto regularized_run =
            run_case("regularized", regularized);

        JointRegularization unregularized;
        unregularized.initial_weight = 0.0;
        unregularized.boundary_weight = 0.0;
        unregularized.boundary_smoothness = 0.0;

        const auto unregularized_run =
            run_case("unregularized", unregularized);

        std::cout
            << "\ninitial_truth_error="
            << norm(truth_x)
            << '\n'
            << "regularized_GN_truth_error="
            << regularized_run.truth_error
            << '\n'
            << "unregularized_GN_truth_error="
            << unregularized_run.truth_error
            << '\n'
            << "PASS joint nonlinear reconstruction\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';

        return 1;
    }
}
