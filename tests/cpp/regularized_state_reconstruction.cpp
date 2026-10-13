#include "state_finder/gauss_newton_operator.hpp"
#include "state_finder/gauss_newton_optimizer.hpp"
#include "state_finder/regularization_preconditioner.hpp"
#include "state_finder/low_rank_preconditioner.hpp"
#include "state_finder/trajectory.hpp"
#include "state_finder/adjoint.hpp"
#include "state_finder/objective.hpp"
#include "state_finder/optimizer.hpp"
#include "state_finder/quadratic_regularizer.hpp"
#include "state_finder/regularized_objective.hpp"
#include "state_finder/scaled_initial_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace state_finder;

InitialState make_reference(
    const ForwardConfiguration& config)
{
    const int n = config.n;
    const auto& par = config.parameters;

    InitialState state{
        std::vector<double>(n+1),
        std::vector<double>(n, 100.0),
        100.0
    };

    const double mdot = 100.0;

    const double reynolds =
        mdot*par.diameter /
        (par.area*par.viscosity);

    const double argument =
        par.roughness/(3.7*par.diameter) +
        5.74/std::pow(reynolds, 0.9);

    const double friction =
        0.25/std::pow(std::log10(argument), 2);

    const double coefficient =
        friction*par.z*par.gas_constant*
        par.temperature*mdot*mdot /
        (par.diameter*par.area*par.area);

    for (int i = 0; i <= n; ++i) {
        const double squared =
            config.demand.inlet_pressure *
            config.demand.inlet_pressure -
            coefficient*i*par.dx;

        if (!(squared > 0.0))
            throw std::runtime_error(
                "Invalid reference pressure profile");

        state.pressure[i] = std::sqrt(squared);
    }

    return state;
}

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error(
            "Dot-product dimension mismatch");

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i]*b[i];

    return result;
}

double norm(const std::vector<double>& a)
{
    return std::sqrt(dot(a, a));
}

struct Errors {
    double pressure_rmse = 0.0;
    double flow_rmse = 0.0;
    double inlet_flow_error = 0.0;
};

Errors reconstruction_errors(
    const InitialState& state,
    const InitialState& truth)
{
    const std::size_t n = truth.flow.size();

    double pressure_squared = 0.0;
    double flow_squared = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const double dp =
            state.pressure[i+1] -
            truth.pressure[i+1];

        const double dm =
            state.flow[i] -
            truth.flow[i];

        pressure_squared += dp*dp;
        flow_squared += dm*dm;
    }

    return {
        std::sqrt(pressure_squared/n),
        std::sqrt(flow_squared/n),
        state.inlet_flow-truth.inlet_flow
    };
}

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        int max_iterations = 30;

        if (argc < 2 || argc > 4)
            throw std::runtime_error(
                "Usage: regularized_state_reconstruction "
                "METHOD [MAX_ITERATIONS] [lbfgs|gn]");

        const std::string method = argv[1];

        if (argc >= 3)
            max_iterations = std::stoi(argv[2]);

        const std::string optimizer_name =
            argc == 4 ? argv[3] : "lbfgs";

        require(
            optimizer_name == "lbfgs" ||
            optimizer_name == "gn",
            "Unknown optimizer");

        require(
            max_iterations > 0,
            "MAX_ITERATIONS must be positive");

        require(
            method == "unregularized" ||
            method == "tikhonov" ||
            method == "curvature" ||
            method == "hybrid",
            "Unknown regularization method");

        double lambda0 = 0.0;
        double lambda2 = 0.0;

        if (method == "tikhonov") {
            lambda0 = 1.0e-4;
        }
        else if (method == "curvature") {
            lambda2 = 1.0e-1;
        }
        else if (method == "hybrid") {
            lambda0 = 1.0e-4;
            lambda2 = 1.0e-1;
        }

        constexpr int n = 100;
        constexpr int steps = 80;

        const double area = std::acos(-1.0)/4.0;

        const ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                100000.0/n,
                15.0,
                0.65,
                1.0,
                area,
                288.15,
                0.9,
                500.0,
                1.1e-5,
                4.5e-5
            },
            pipe_sim::DemandRamp{
                8.0e6,
                100.0,
                105.0,
                600.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9,
                1.0e-12,
                30,
                false
            }
        };

        const auto truth = make_reference(config);

        const auto reference =
            evaluate_forward(config, truth);

        require(
            reference.info == 0 &&
            reference.completed_steps == steps,
            "Reference forward simulation failed");

        // Construct the complete four-channel boundary
        // measurement record.
        //
        // The present forward formulation prescribes
        // inlet pressure and outlet mass flow.
        //
        // Inlet mass flow and outlet pressure are
        // independent response observations.

        BoundaryObservationRecord boundary_record;

        const auto sample_count =
            static_cast<std::size_t>(steps + 1);

        boundary_record.measured.inlet_pressure.assign(
            sample_count,
            config.demand.inlet_pressure);

        boundary_record.measured.inlet_flow =
            reference.inlet_flow;

        boundary_record.measured.outlet_pressure =
            reference.outlet_pressure;

        // Populate outlet mass flow from the actual
        // reference trajectory, including the initial
        // sample and the demand-ramp history.
        boundary_record.measured.outlet_flow =
            reference.outlet_flow;

        boundary_record.scales.inlet_pressure = 1.0e4;
        boundary_record.scales.inlet_flow = 1.0;
        boundary_record.scales.outlet_pressure = 1.0e4;
        boundary_record.scales.outlet_flow = 1.0;

        boundary_record.roles.inlet_pressure =
            MeasurementRole::PrescribedInput;

        boundary_record.roles.inlet_flow =
            MeasurementRole::ObjectiveResidual;

        boundary_record.roles.outlet_pressure =
            MeasurementRole::ObjectiveResidual;

        boundary_record.roles.outlet_flow =
            MeasurementRole::PrescribedInput;

        boundary_record.include_initial = true;

        validate_boundary_observations(
            boundary_record,
            sample_count);

        validate_current_boundary_roles(
            boundary_record.roles);

        // Compatibility conversion preserves the
        // verified two-residual objective and adjoint.
        const Observations observations =
            make_current_observations(
                boundary_record,
                sample_count);

        require(
            observations.inlet_flow ==
                reference.inlet_flow,
            "Inlet-flow observation migration failed");

        require(
            observations.outlet_pressure ==
                reference.outlet_pressure,
            "Outlet-pressure observation migration failed");

        require(
            observations.include_initial,
            "Initial observation flag was lost");

        std::cout
            << "PASS four-channel reconstruction input\\n";

        auto candidate = truth;

        // Reproduce the M1.5b perturbation exactly.
        for (int i = 1; i <= n; ++i)
            candidate.pressure[i] +=
                1000.0*std::sin(0.031*i);

        for (int i = 0; i < n; ++i)
            candidate.flow[i] +=
                0.5*std::cos(0.047*(i+1));

        candidate.inlet_flow += 0.7;

        // The baseline is fixed at the perturbed guess.
        // xi = 0 corresponds to the original candidate.
        const InitialState baseline = candidate;

        const InitialStateScales scales{
            1.0e5,
            10.0
        };

        const std::vector<double> x0(2*n+1, 0.0);

        // The baseline is the perturbed initial guess.
        // The regularization reference is therefore xi=0.
        //
        // The synthetic truth is NOT used as a prior.

        const std::vector<double> regularization_reference(
            2*n+1, 0.0);

        const QuadraticRegularizer regularizer(
            n,
            regularization_reference,
            lambda0,
            lambda2);

        const auto initial =
            evaluate_adjoint(
                config,
                candidate,
                observations);

        const double expected =
            5.3415934241208767;

        std::cout << std::setprecision(17);

        std::cout
            << "initial_objective="
            << initial.objective << '\n';

        require(
            std::abs(initial.objective-expected) <
                1.0e-8,
            "M1.5b initial-objective regression failed");

        // Check that the scaled transformation and
        // chain-rule gradient agree with finite differences.
        const auto scaled_gradient =
            scale_adjoint_gradient(
                initial.gradient,
                n,
                scales);

        std::vector<double> direction(2*n+1);

        for (int i = 0; i < 2*n+1; ++i)
            direction[i] =
                std::sin(0.071*(i+1));

        const double direction_norm = norm(direction);

        for (double& value : direction)
            value /= direction_norm;

        const double analytical =
            dot(scaled_gradient, direction);

        const double h = 1.0e-5;

        auto plus = x0;
        auto minus = x0;

        for (std::size_t i = 0; i < x0.size(); ++i) {
            plus[i] += h*direction[i];
            minus[i] -= h*direction[i];
        }

        const double j_plus =
            evaluate_objective(
                config,
                unpack_scaled_state(
                    plus, baseline, scales),
                observations).value;

        const double j_minus =
            evaluate_objective(
                config,
                unpack_scaled_state(
                    minus, baseline, scales),
                observations).value;

        const double numerical =
            (j_plus-j_minus)/(2.0*h);

        const double relative_error =
            std::abs(numerical-analytical) /
            std::max(1.0, std::abs(analytical));

        std::cout
            << "scaled_directional_adjoint="
            << analytical << '\n'
            << "scaled_directional_fd="
            << numerical << '\n'
            << "scaled_directional_relative_error="
            << relative_error << '\n';

        require(
            relative_error < 1.0e-5,
            "Scaled gradient verification failed");

        int rejected_evaluations = 0;

        const ObjectiveFunction objective =
            [&](const std::vector<double>& x,
                bool need_gradient)
            {
                OptimizationEvaluation evaluation;

                try {
                    const auto state =
                        unpack_scaled_state(
                            x, baseline, scales);

                    // Reject physically invalid pressure
                    // states before calling the solver.
                    for (double pressure : state.pressure) {
                        if (!std::isfinite(pressure) ||
                            pressure <= 0.0)
                        {
                            ++rejected_evaluations;
                            return evaluation;
                        }
                    }

                    if (need_gradient) {
                        const auto result =
                            evaluate_adjoint(
                                config,
                                state,
                                observations);

                        evaluation.objective =
                            result.objective;

                        evaluation.gradient =
                            scale_adjoint_gradient(
                                result.gradient,
                                n,
                                scales);
                    }
                    else {
                        const auto result =
                            evaluate_objective(
                                config,
                                state,
                                observations);

                        evaluation.objective =
                            result.value;
                    }

                    evaluation.valid =
                        std::isfinite(evaluation.objective);

                    if (!evaluation.valid)
                        ++rejected_evaluations;
                }
                catch (const std::exception&) {
                    // Failed trial states are rejected.
                    // The line search may try a smaller step.
                    ++rejected_evaluations;
                }

                return evaluation;
            };

        OptimizationOptions options;
        options.max_iterations = max_iterations;
        options.history_size = 10;
        options.gradient_tolerance = 1.0e-8;
        options.relative_objective_tolerance = 1.0e-14;
        options.max_line_search_steps = 40;

        const auto regularized_objective =
            make_regularized_objective(
                objective,
                regularizer);

        OptimizationResult result;

        if (optimizer_name == "lbfgs") {
            result = minimize_lbfgs(
                regularized_objective,
                x0,
                options);
        }
        else {
            GaussNewtonOptions gn_options;

            if (const char* value =
                    std::getenv("STATE_FINDER_GN_FIXED_FORCING"))
            {
                gn_options.adaptive_forcing =
                    std::string(value) != "1";
            }

            gn_options.max_iterations =
                max_iterations;

            gn_options.gradient_tolerance =
                options.gradient_tolerance;

            gn_options.relative_objective_tolerance =
                options.relative_objective_tolerance;

            gn_options.max_line_search_steps =
                options.max_line_search_steps;

            // Build an independent linearization at the
            // current nonlinear state. Its lifetime must
            // extend through the complete PCG solve.
            GaussNewtonDiagnostics gn_diagnostics;

            const char* rank_env =
                std::getenv("STATE_FINDER_GN_LOWRANK");

            const int lowrank =
                rank_env ? std::stoi(rank_env) : 0;

            if (lowrank < 0 || lowrank > 32)
                throw std::invalid_argument(
                    "Low-rank must be between 0 and 32");

            int lowrank_setup_calls = 0;
            int lowrank_builds = 0;

            const bool use_preconditioner =
                lowrank > 0 ||
                std::getenv("STATE_FINDER_GN_PRECONDITIONED") !=
                    nullptr;

            const GaussNewtonPreconditionerFactory
                preconditioner_factory =
                    [&](const LinearOperator& A,
                        double damping) -> LinearOperator
                    {
                        auto base = std::make_shared<
                            RegularizationPreconditioner>(
                                regularizer, n, damping);

                        if (lowrank == 0) {
                            return [base](
                                const std::vector<double>& rhs)
                            {
                                return base->apply(rhs);
                            };
                        }

                        auto lr = std::make_shared<
                            LowRankPreconditioner>(
                                base,
                                regularizer,
                                A,
                                damping,
                                2*n+1,
                                lowrank,
                                lowrank_setup_calls);

                        ++lowrank_builds;

                        return [lr](
                            const std::vector<double>& rhs)
                        {
                            return lr->apply(rhs);
                        };
                    };

            std::cout
                << "gn_preconditioned="
                << (use_preconditioner ? 1 : 0)
                << '\n';

            std::cout
                << "gn_lowrank="
                << lowrank << '\n';

            std::vector<double> cached_x;
            LinearOperator cached_undamped_operator;
            bool cache_valid = false;

            const GaussNewtonOperatorFactory factory =
                [&](const std::vector<double>& x,
                    double damping) -> LinearOperator
                {
                    if (!cache_valid || cached_x != x) {
                        const auto state =
                            unpack_scaled_state(
                                x, baseline, scales);

                        auto trajectory =
                            std::make_shared<Trajectory>(
                                record_trajectory(config, state));

                        if (trajectory->info != 0 ||
                            trajectory->completed_steps !=
                                config.steps)
                        {
                            throw std::runtime_error(
                                "Gauss-Newton trajectory failed");
                        }

                        auto operator_ptr =
                            std::make_shared<GaussNewtonOperator>(
                                config,
                                *trajectory,
                                observations,
                                regularizer,
                                scales,
                                0.0);

                        cached_undamped_operator =
                            [trajectory, operator_ptr](
                                const std::vector<double>& v)
                            {
                                return operator_ptr->apply(v);
                            };

                        cached_x = x;
                        cache_valid = true;
                        ++gn_diagnostics.linearizations;
                    }

                    // Capture the undamped operator by value.
                    // Retains trajectory ownership and avoids
                    // dangling references when the cache changes.
                    const auto H = cached_undamped_operator;

                    return [H, damping](
                        const std::vector<double>& v)
                    {
                        auto result = H(v);

                        for (std::size_t j = 0;
                             j < v.size(); ++j)
                        {
                            result[j] += damping*v[j];
                        }

                        return result;
                    };
                };

            result = minimize_gauss_newton(
                regularized_objective,
                x0,
                factory,
                gn_options,
                &gn_diagnostics,
                use_preconditioner
                    ? preconditioner_factory
                    : GaussNewtonPreconditionerFactory{});

            std::cout
                << "gn_lowrank_setup_calls="
                << lowrank_setup_calls << '\n';

            std::cout
                << "gn_lowrank_builds="
                << lowrank_builds << '\n';

            std::cout
                << "gn_total_operator_calls="
                << gn_diagnostics.operator_calls +
                   lowrank_setup_calls
                << '\n';

            std::cout
                << "gn_linearizations="
                << gn_diagnostics.linearizations << '\n'
                << "gn_pcg_solves="
                << gn_diagnostics.pcg_solves << '\n'
                << "gn_pcg_iterations="
                << gn_diagnostics.pcg_iterations << '\n'
                << "gn_operator_calls="
                << gn_diagnostics.operator_calls << '\n'
                << "gn_damping_retries="
                << gn_diagnostics.damping_retries << '\n'
                << "gn_gradient_fallbacks="
                << gn_diagnostics.gradient_fallbacks << '\n'
                << "gn_max_true_residual_ratio="
                << gn_diagnostics.maximum_true_residual_ratio << '\n'
                << "gn_final_damping="
                << gn_diagnostics.final_damping << '\n'
                << "gn_final_forcing_tolerance="
                << gn_diagnostics.final_forcing_tolerance << '\n';
        }

        const auto recovered =
            unpack_scaled_state(
                result.variables,
                baseline,
                scales);

        const auto initial_errors =
            reconstruction_errors(candidate, truth);

        const auto final_errors =
            reconstruction_errors(recovered, truth);

        const auto final_penalty =
            regularizer.evaluate(result.variables);

        const auto final_observation =
            evaluate_objective(
                config,
                recovered,
                observations);

        const double final_total_check =
            final_observation.value +
            final_penalty.total();

        std::cout
            << "\noptimizer=" << optimizer_name
            << "\nmethod=" << method
            << "\nlambda0=" << lambda0
            << "\nlambda2=" << lambda2
            << "\nfinal_observation_objective="
            << final_observation.value
            << "\nfinal_zeroth_order_penalty="
            << final_penalty.zeroth_order
            << "\nfinal_curvature_penalty="
            << final_penalty.curvature
            << "\nfinal_total_objective_check="
            << final_total_check
            << '\n';

        std::cout
            << "\niteration,objective,"
            << "scaled_gradient_norm,step\n";

        for (const auto& row : result.history) {
            std::cout
                << row.iteration << ','
                << row.objective << ','
                << row.gradient_norm << ','
                << row.step << '\n';
        }

        std::cout
            << "\nconverged=" << result.converged
            << "\nreason=" << result.reason
            << "\niterations=" << result.iterations
            << "\ninitial_objective="
            << result.initial_objective
            << "\nfinal_objective="
            << result.final_objective
            << "\nfinal_scaled_gradient_norm="
            << result.gradient_norm
            << "\nobjective_evaluations="
            << result.objective_evaluations
            << "\ngradient_evaluations="
            << result.gradient_evaluations
            << "\nrejected_evaluations="
            << rejected_evaluations
            << "\ninitial_pressure_rmse_pa="
            << initial_errors.pressure_rmse
            << "\nfinal_pressure_rmse_pa="
            << final_errors.pressure_rmse
            << "\ninitial_flow_rmse_kg_s="
            << initial_errors.flow_rmse
            << "\nfinal_flow_rmse_kg_s="
            << final_errors.flow_rmse
            << "\ninitial_inlet_flow_error_kg_s="
            << initial_errors.inlet_flow_error
            << "\nfinal_inlet_flow_error_kg_s="
            << final_errors.inlet_flow_error
            << '\n';

        require(
            std::isfinite(result.final_objective),
            "Final objective is nonfinite");

        require(
            result.final_objective <
                result.initial_objective,
            "Optimizer failed to reduce objective");

        require(
            result.objective_evaluations >
                result.gradient_evaluations,
            "Objective-only line search was not used");

        const auto final_objective_check =
            evaluate_objective(
                config,
                recovered,
                observations);

        const double parity_error =
            std::abs(
                final_total_check -
                result.final_objective);

        require(
            parity_error <
                1.0e-10 *
                std::max(
                    1.0,
                    std::abs(result.final_objective)),
            "Final objective parity failure");


        // M1.6d predictive validation.
        //
        // The inverse problem above uses exactly 80 steps.
        // Extend the forward horizon without changing the
        // reconstructed initial state or optimization data.

        constexpr int prediction_steps = 160;
        constexpr int assimilation_steps = steps;

        auto extended_config = config;
        extended_config.steps = prediction_steps;

        const auto prediction_truth =
            evaluate_forward(extended_config, truth);

        const auto prediction_recovered =
            evaluate_forward(extended_config, recovered);

        require(
            prediction_truth.info == 0 &&
            prediction_truth.completed_steps == prediction_steps,
            "Extended truth simulation failed");

        require(
            prediction_recovered.info == 0 &&
            prediction_recovered.completed_steps == prediction_steps,
            "Extended reconstructed simulation failed");


        // --------------------------------------------------
        // M2.1a: Initial-state reconstruction validation.
        //
        // Compare the actual recovered state with the known
        // synthetic truth, not merely the observation fit.
        //
        // The inlet pressure is prescribed and therefore
        // excluded from the optimized pressure coordinates.
        // --------------------------------------------------

        const auto candidate_forward =
            evaluate_forward(config, candidate);

        require(
            candidate_forward.info == 0 &&
            candidate_forward.completed_steps == steps,
            "M2.1a candidate forward simulation failed");

        require(
            !reference.linepack.empty() &&
            !candidate_forward.linepack.empty() &&
            !prediction_recovered.linepack.empty(),
            "M2.1a missing linepack samples");

        const double truth_initial_linepack =
            reference.linepack.front();

        const double guess_initial_linepack =
            candidate_forward.linepack.front();

        const double recovered_initial_linepack =
            prediction_recovered.linepack.front();

        const double guess_linepack_error =
            guess_initial_linepack - truth_initial_linepack;

        const double recovered_linepack_error =
            recovered_initial_linepack -
            truth_initial_linepack;

        double guess_pressure_max_error = 0.0;
        double recovered_pressure_max_error = 0.0;

        double guess_flow_max_error = 0.0;
        double recovered_flow_max_error = 0.0;

        for (int i = 1; i <= n; ++i) {
            guess_pressure_max_error = std::max(
                guess_pressure_max_error,
                std::abs(
                    candidate.pressure[i] -
                    truth.pressure[i]));

            recovered_pressure_max_error = std::max(
                recovered_pressure_max_error,
                std::abs(
                    recovered.pressure[i] -
                    truth.pressure[i]));
        }

        for (int i = 0; i < n; ++i) {
            guess_flow_max_error = std::max(
                guess_flow_max_error,
                std::abs(
                    candidate.flow[i] -
                    truth.flow[i]));

            recovered_flow_max_error = std::max(
                recovered_flow_max_error,
                std::abs(
                    recovered.flow[i] -
                    truth.flow[i]));
        }

        const auto improvement_factor =
            [](double before, double after) {
                if (after == 0.0)
                    return before == 0.0 ? 1.0 : 0.0;

                return before/after;
            };

        std::filesystem::create_directories(
            "benchmarks/results-m21");

        const std::string m21_prefix =
            std::string("benchmarks/results-m21/") +
            optimizer_name + "_" + method;

        // Spatial profile. Pressure node zero is included
        // for context, but is a prescribed boundary value.
        {
            std::ofstream csv(
                m21_prefix + "_initial_state.csv");

            require(
                static_cast<bool>(csv),
                "Cannot open M2.1a state CSV");

            csv << std::setprecision(17);

            csv
                << "field,index,x_m,"
                << "truth,initial_guess,recovered,"
                << "guess_error,recovered_error\n";

            for (int i = 0; i <= n; ++i) {
                const double x =
                    i*config.parameters.dx;

                csv
                    << "pressure_pa,"
                    << i << ','
                    << x << ','
                    << truth.pressure[i] << ','
                    << candidate.pressure[i] << ','
                    << recovered.pressure[i] << ','
                    << candidate.pressure[i] -
                       truth.pressure[i] << ','
                    << recovered.pressure[i] -
                       truth.pressure[i] << '\n';
            }

            for (int i = 0; i < n; ++i) {
                const double x =
                    (i+0.5)*config.parameters.dx;

                csv
                    << "flow_kg_s,"
                    << i << ','
                    << x << ','
                    << truth.flow[i] << ','
                    << candidate.flow[i] << ','
                    << recovered.flow[i] << ','
                    << candidate.flow[i] -
                       truth.flow[i] << ','
                    << recovered.flow[i] -
                       truth.flow[i] << '\n';
            }

            csv
                << "inlet_flow_kg_s,0,0,"
                << truth.inlet_flow << ','
                << candidate.inlet_flow << ','
                << recovered.inlet_flow << ','
                << candidate.inlet_flow -
                   truth.inlet_flow << ','
                << recovered.inlet_flow -
                   truth.inlet_flow << '\n';

            csv.close();

            require(
                static_cast<bool>(csv),
                "Failed writing M2.1a state CSV");
        }

        const double pressure_improvement =
            improvement_factor(
                initial_errors.pressure_rmse,
                final_errors.pressure_rmse);

        const double flow_improvement =
            improvement_factor(
                initial_errors.flow_rmse,
                final_errors.flow_rmse);

        const double inlet_flow_improvement =
            improvement_factor(
                std::abs(initial_errors.inlet_flow_error),
                std::abs(final_errors.inlet_flow_error));

        const double linepack_improvement =
            improvement_factor(
                std::abs(guess_linepack_error),
                std::abs(recovered_linepack_error));

        const double relative_linepack_error =
            std::abs(recovered_linepack_error) /
            std::max(
                std::abs(truth_initial_linepack),
                1.0);

        std::cout
            << "\nM2.1a initial-state recovery\n"
            << "m21_state_dimension=" << 2*n+1 << '\n'
            << "m21_pressure_dofs=" << n << '\n'
            << "m21_flow_dofs=" << n << '\n'
            << "m21_inlet_flow_dofs=1\n"

            << "m21_guess_pressure_rmse_pa="
            << initial_errors.pressure_rmse << '\n'

            << "m21_recovered_pressure_rmse_pa="
            << final_errors.pressure_rmse << '\n'

            << "m21_pressure_improvement="
            << pressure_improvement << '\n'

            << "m21_guess_pressure_max_error_pa="
            << guess_pressure_max_error << '\n'

            << "m21_recovered_pressure_max_error_pa="
            << recovered_pressure_max_error << '\n'

            << "m21_guess_flow_rmse_kg_s="
            << initial_errors.flow_rmse << '\n'

            << "m21_recovered_flow_rmse_kg_s="
            << final_errors.flow_rmse << '\n'

            << "m21_flow_improvement="
            << flow_improvement << '\n'

            << "m21_guess_flow_max_error_kg_s="
            << guess_flow_max_error << '\n'

            << "m21_recovered_flow_max_error_kg_s="
            << recovered_flow_max_error << '\n'

            << "m21_inlet_flow_improvement="
            << inlet_flow_improvement << '\n'

            << "m21_truth_initial_linepack="
            << truth_initial_linepack << '\n'

            << "m21_guess_initial_linepack="
            << guess_initial_linepack << '\n'

            << "m21_recovered_initial_linepack="
            << recovered_initial_linepack << '\n'

            << "m21_guess_linepack_error="
            << guess_linepack_error << '\n'

            << "m21_recovered_linepack_error="
            << recovered_linepack_error << '\n'

            << "m21_linepack_improvement="
            << linepack_improvement << '\n'

            << "m21_initial_linepack_relative_error="
            << relative_linepack_error << '\n'

            << "m21_state_csv="
            << m21_prefix + "_initial_state.csv"
            << '\n';

        {
            std::ofstream csv(
                m21_prefix + "_summary.csv");

            require(
                static_cast<bool>(csv),
                "Cannot open M2.1a summary CSV");

            csv << std::setprecision(17);

            csv << "metric,value\n";

            const auto write =
                [&](const char* name, double value) {
                    csv << name << ',' << value << '\n';
                };

            write(
                "guess_pressure_rmse_pa",
                initial_errors.pressure_rmse);

            write(
                "recovered_pressure_rmse_pa",
                final_errors.pressure_rmse);

            write(
                "pressure_improvement",
                pressure_improvement);

            write(
                "guess_pressure_max_error_pa",
                guess_pressure_max_error);

            write(
                "recovered_pressure_max_error_pa",
                recovered_pressure_max_error);

            write(
                "guess_flow_rmse_kg_s",
                initial_errors.flow_rmse);

            write(
                "recovered_flow_rmse_kg_s",
                final_errors.flow_rmse);

            write(
                "flow_improvement",
                flow_improvement);

            write(
                "guess_flow_max_error_kg_s",
                guess_flow_max_error);

            write(
                "recovered_flow_max_error_kg_s",
                recovered_flow_max_error);

            write(
                "inlet_flow_improvement",
                inlet_flow_improvement);

            write(
                "truth_initial_linepack",
                truth_initial_linepack);

            write(
                "guess_initial_linepack",
                guess_initial_linepack);

            write(
                "recovered_initial_linepack",
                recovered_initial_linepack);

            write(
                "guess_linepack_error",
                guess_linepack_error);

            write(
                "recovered_linepack_error",
                recovered_linepack_error);

            write(
                "linepack_improvement",
                linepack_improvement);

            write(
                "initial_linepack_relative_error",
                relative_linepack_error);

            write(
                "final_objective",
                result.final_objective);

            write(
                "final_gradient_norm",
                result.gradient_norm);

            write(
                "iterations",
                result.iterations);

            csv.close();

            require(
                static_cast<bool>(csv),
                "Failed writing M2.1a summary CSV");
        }

        require(
            std::isfinite(pressure_improvement) &&
            std::isfinite(flow_improvement) &&
            std::isfinite(relative_linepack_error),
            "Nonfinite M2.1a reconstruction diagnostics");

        const std::size_t expected_samples =
            static_cast<std::size_t>(prediction_steps + 1);

        require(
            prediction_truth.inlet_flow.size() == expected_samples &&
            prediction_recovered.inlet_flow.size() == expected_samples &&
            prediction_truth.outlet_pressure.size() == expected_samples &&
            prediction_recovered.outlet_pressure.size() == expected_samples &&
            prediction_truth.linepack.size() == expected_samples &&
            prediction_recovered.linepack.size() == expected_samples &&
            prediction_truth.time.size() == expected_samples,
            "Extended trajectory has unexpected sample count");

        struct WindowErrors {
            double inlet_flow_rmse = 0.0;
            double outlet_pressure_rmse = 0.0;
            double linepack_rmse = 0.0;
            double linepack_relative_rmse = 0.0;
        };

        const auto calculate_window_errors =
            [&](int first, int last) -> WindowErrors
        {
            double inlet_squared = 0.0;
            double outlet_squared = 0.0;
            double linepack_squared = 0.0;
            double linepack_reference_squared = 0.0;

            for (int k = first; k <= last; ++k) {
                const std::size_t i =
                    static_cast<std::size_t>(k);

                const double inlet_error =
                    prediction_recovered.inlet_flow[i] -
                    prediction_truth.inlet_flow[i];

                const double outlet_error =
                    prediction_recovered.outlet_pressure[i] -
                    prediction_truth.outlet_pressure[i];

                const double linepack_error =
                    prediction_recovered.linepack[i] -
                    prediction_truth.linepack[i];

                inlet_squared += inlet_error*inlet_error;
                outlet_squared += outlet_error*outlet_error;
                linepack_squared += linepack_error*linepack_error;

                const double inventory =
                    prediction_truth.linepack[i];

                linepack_reference_squared +=
                    inventory*inventory;
            }

            const double count =
                static_cast<double>(last-first+1);

            WindowErrors errors;

            errors.inlet_flow_rmse =
                std::sqrt(inlet_squared/count);

            errors.outlet_pressure_rmse =
                std::sqrt(outlet_squared/count);

            errors.linepack_rmse =
                std::sqrt(linepack_squared/count);

            errors.linepack_relative_rmse =
                linepack_reference_squared > 0.0
                ? std::sqrt(
                    linepack_squared /
                    linepack_reference_squared)
                : 0.0;

            return errors;
        };

        // Assimilation samples include t=0 through t=1200 s.
        // Prediction samples start at t=1215 s.
        // The windows are disjoint.
        const auto assimilation_errors =
            calculate_window_errors(0, assimilation_steps);

        const auto prediction_errors =
            calculate_window_errors(
                assimilation_steps+1,
                prediction_steps);

        require(
            prediction_truth.final_pressure.size() ==
                prediction_recovered.final_pressure.size() &&
            prediction_truth.final_flow.size() ==
                prediction_recovered.final_flow.size(),
            "Final internal state dimension mismatch");

        double final_pressure_squared = 0.0;
        double final_flow_squared = 0.0;

        for (std::size_t i = 0;
             i < prediction_truth.final_pressure.size(); ++i)
        {
            const double difference =
                prediction_recovered.final_pressure[i] -
                prediction_truth.final_pressure[i];

            final_pressure_squared += difference*difference;
        }

        for (std::size_t i = 0;
             i < prediction_truth.final_flow.size(); ++i)
        {
            const double difference =
                prediction_recovered.final_flow[i] -
                prediction_truth.final_flow[i];

            final_flow_squared += difference*difference;
        }

        const double final_internal_pressure_rmse =
            std::sqrt(
                final_pressure_squared /
                prediction_truth.final_pressure.size());

        const double final_internal_flow_rmse =
            std::sqrt(
                final_flow_squared /
                prediction_truth.final_flow.size());

        std::filesystem::create_directories(
            "benchmarks/results-m16d");

        const std::string csv_path =
            std::string("benchmarks/results-m16d/") +
            (optimizer_name == "gn" ? "gn_" : "") + method + "_trajectory.csv";

        std::ofstream csv(csv_path);

        require(
            static_cast<bool>(csv),
            "Cannot open predictive trajectory CSV");

        csv << std::setprecision(17);

        csv
            << "step,time_s,window,"
            << "truth_inlet_flow_kg_s,"
            << "recovered_inlet_flow_kg_s,"
            << "truth_outlet_pressure_pa,"
            << "recovered_outlet_pressure_pa,"
            << "truth_linepack,"
            << "recovered_linepack\n";

        for (int k = 0; k <= prediction_steps; ++k) {
            const std::size_t i =
                static_cast<std::size_t>(k);

            csv
                << k << ','
                << prediction_truth.time[i] << ','
                << (k <= assimilation_steps
                    ? "assimilation" : "prediction") << ','
                << prediction_truth.inlet_flow[i] << ','
                << prediction_recovered.inlet_flow[i] << ','
                << prediction_truth.outlet_pressure[i] << ','
                << prediction_recovered.outlet_pressure[i] << ','
                << prediction_truth.linepack[i] << ','
                << prediction_recovered.linepack[i] << '\n';
        }

        csv.close();

        require(
            static_cast<bool>(csv),
            "Failed writing predictive trajectory CSV");

        std::cout
            << "\nM1.6d predictive validation\n"
            << "prediction_method=" << method << '\n'
            << "assimilation_steps=" << assimilation_steps << '\n'
            << "prediction_steps=" << prediction_steps << '\n'
            << "assimilation_inlet_flow_rmse_kg_s="
            << assimilation_errors.inlet_flow_rmse << '\n'
            << "assimilation_outlet_pressure_rmse_pa="
            << assimilation_errors.outlet_pressure_rmse << '\n'
            << "prediction_inlet_flow_rmse_kg_s="
            << prediction_errors.inlet_flow_rmse << '\n'
            << "prediction_outlet_pressure_rmse_pa="
            << prediction_errors.outlet_pressure_rmse << '\n'
            << "prediction_linepack_rmse="
            << prediction_errors.linepack_rmse << '\n'
            << "prediction_linepack_relative_rmse="
            << prediction_errors.linepack_relative_rmse << '\n'
            << "prediction_final_internal_pressure_rmse_pa="
            << final_internal_pressure_rmse << '\n'
            << "prediction_final_internal_flow_rmse_kg_s="
            << final_internal_flow_rmse << '\n'
            << "prediction_csv=" << csv_path << '\n';

        require(
            std::isfinite(prediction_errors.inlet_flow_rmse) &&
            std::isfinite(prediction_errors.outlet_pressure_rmse) &&
            std::isfinite(prediction_errors.linepack_rmse) &&
            std::isfinite(final_internal_pressure_rmse) &&
            std::isfinite(final_internal_flow_rmse),
            "Nonfinite predictive error");


        // M1.6e changed-demand validation.
        //
        // Restart at the end of the assimilation window.
        // Both trajectories receive identical future inputs.
        // The optimization itself is unchanged.

        const auto assimilation_truth =
            evaluate_forward(config, truth);

        const auto assimilation_recovered =
            evaluate_forward(config, recovered);

        require(
            assimilation_truth.info == 0 &&
            assimilation_truth.completed_steps == steps &&
            assimilation_recovered.info == 0 &&
            assimilation_recovered.completed_steps == steps,
            "M1.6e assimilation propagation failed");

        const auto restart_state =
            [](const ForwardResult& result) -> InitialState
        {
            return {
                result.final_pressure,
                result.final_flow,
                result.final_inlet_flow
            };
        };

        const auto truth_restart =
            restart_state(assimilation_truth);

        const auto recovered_restart =
            restart_state(assimilation_recovered);

        struct DemandScenario {
            const char* name;
            double target_flow;
        };

        const DemandScenario scenarios[] = {
            {"baseline", 105.0},
            {"increased", 110.0},
            {"decreased", 100.0}
        };

        std::filesystem::create_directories(
            "benchmarks/results-m16e");

        for (const auto& scenario : scenarios) {
            auto future_config = config;

            future_config.steps = steps;

            // Local time begins at the restart.
            // Demand changes over one 15-second time step.
            future_config.demand =
                pipe_sim::DemandRamp{
                    config.demand.inlet_pressure,
                    105.0,
                    scenario.target_flow,
                    config.parameters.dt
                };

            const auto future_truth =
                evaluate_forward(
                    future_config,
                    truth_restart);

            const auto future_recovered =
                evaluate_forward(
                    future_config,
                    recovered_restart);

            require(
                future_truth.info == 0 &&
                future_truth.completed_steps == steps &&
                future_recovered.info == 0 &&
                future_recovered.completed_steps == steps,
                "M1.6e future propagation failed");

            double inlet_error2 = 0.0;
            double outlet_error2 = 0.0;
            double inventory_error2 = 0.0;
            double inventory_reference2 = 0.0;

            const double dt = config.parameters.dt;

            const std::string csv_path =
                std::string("benchmarks/results-m16e/") +
                (optimizer_name == "gn" ? "gn_" : "") + method + "_" +
                scenario.name + ".csv";

            std::ofstream csv(csv_path);

            require(
                static_cast<bool>(csv),
                "Cannot open M1.6e CSV");

            csv << std::setprecision(17);

            csv
                << "time_s,"
                << "outlet_demand_kg_s,"
                << "truth_inlet_flow_kg_s,"
                << "recovered_inlet_flow_kg_s,"
                << "truth_outlet_pressure_pa,"
                << "recovered_outlet_pressure_pa,"
                << "truth_linepack,"
                << "recovered_linepack\n";

            for (int k = 0; k <= steps; ++k) {
                const auto i =
                    static_cast<std::size_t>(k);

                const double time =
                    steps*dt + k*dt;

                const double inlet_error =
                    future_recovered.inlet_flow[i] -
                    future_truth.inlet_flow[i];

                const double outlet_error =
                    future_recovered.outlet_pressure[i] -
                    future_truth.outlet_pressure[i];

                const double inventory_error =
                    future_recovered.linepack[i] -
                    future_truth.linepack[i];

                inlet_error2 += inlet_error*inlet_error;
                outlet_error2 += outlet_error*outlet_error;
                inventory_error2 +=
                    inventory_error*inventory_error;

                inventory_reference2 +=
                    future_truth.linepack[i] *
                    future_truth.linepack[i];

                csv
                    << time << ','
                    << future_truth.outlet_flow[i] << ','
                    << future_truth.inlet_flow[i] << ','
                    << future_recovered.inlet_flow[i] << ','
                    << future_truth.outlet_pressure[i] << ','
                    << future_recovered.outlet_pressure[i] << ','
                    << future_truth.linepack[i] << ','
                    << future_recovered.linepack[i] << '\n';
            }

            csv.close();

            require(
                static_cast<bool>(csv),
                "Failed writing M1.6e CSV");

            const double count =
                static_cast<double>(steps + 1);

            const double inlet_rmse =
                std::sqrt(inlet_error2/count);

            const double outlet_rmse =
                std::sqrt(outlet_error2/count);

            const double inventory_rmse =
                std::sqrt(inventory_error2/count);

            const double inventory_relative_rmse =
                std::sqrt(
                    inventory_error2 /
                    inventory_reference2);

            double pressure_error2 = 0.0;
            double flow_error2 = 0.0;

            require(
                future_truth.final_pressure.size() ==
                    future_recovered.final_pressure.size() &&
                future_truth.final_flow.size() ==
                    future_recovered.final_flow.size(),
                "M1.6e final state size mismatch");

            for (std::size_t i = 0;
                 i < future_truth.final_pressure.size(); ++i)
            {
                const double error =
                    future_recovered.final_pressure[i] -
                    future_truth.final_pressure[i];

                pressure_error2 += error*error;
            }

            for (std::size_t i = 0;
                 i < future_truth.final_flow.size(); ++i)
            {
                const double error =
                    future_recovered.final_flow[i] -
                    future_truth.final_flow[i];

                flow_error2 += error*error;
            }

            const double pressure_rmse =
                std::sqrt(
                    pressure_error2 /
                    future_truth.final_pressure.size());

            const double flow_rmse =
                std::sqrt(
                    flow_error2 /
                    future_truth.final_flow.size());

            // The baseline restart must reproduce the
            // existing continuous 160-step simulation.
            if (std::string(scenario.name) == "baseline") {
                const double pressure_difference =
                    std::abs(
                        future_truth.outlet_pressure.back() -
                        prediction_truth.outlet_pressure.back());

                const double flow_difference =
                    std::abs(
                        future_truth.inlet_flow.back() -
                        prediction_truth.inlet_flow.back());

                require(
                    pressure_difference < 1.0e-5 &&
                    flow_difference < 1.0e-8,
                    "Restart disagrees with continuous baseline");
            }


            // M1.6e full restart parity and conservation checks.
            //
            // Compare all boundary and inventory samples,
            // including the restart state at local step zero.

            const double conservation_tolerance = 1.0e-8;

            require(
                std::isfinite(future_truth.max_mass_defect) &&
                std::isfinite(future_recovered.max_mass_defect) &&
                future_truth.max_mass_defect <=
                    conservation_tolerance &&
                future_recovered.max_mass_defect <=
                    conservation_tolerance,
                "M1.6e mass conservation defect exceeded tolerance");

            if (std::string(scenario.name) == "baseline") {
                const auto check_series =
                    [&](const std::vector<double>& restarted,
                        const std::vector<double>& continuous,
                        double tolerance,
                        const char* label)
                {
                    require(
                        restarted.size() ==
                            static_cast<std::size_t>(steps + 1) &&
                        continuous.size() ==
                            static_cast<std::size_t>(2*steps + 1),
                        "M1.6e restart series size mismatch");

                    for (int k = 0; k <= steps; ++k) {
                        const double difference =
                            std::abs(
                                restarted[
                                    static_cast<std::size_t>(k)] -
                                continuous[
                                    static_cast<std::size_t>(
                                        steps + k)]);

                        if (!std::isfinite(difference) ||
                            difference > tolerance) {
                            throw std::runtime_error(
                                std::string(
                                    "M1.6e full restart parity: ") +
                                label);
                        }
                    }
                };

                check_series(
                    future_truth.inlet_flow,
                    prediction_truth.inlet_flow,
                    1.0e-8,
                    "truth inlet flow");

                check_series(
                    future_truth.outlet_pressure,
                    prediction_truth.outlet_pressure,
                    1.0e-5,
                    "truth outlet pressure");

                check_series(
                    future_truth.linepack,
                    prediction_truth.linepack,
                    1.0e-5,
                    "truth linepack");

                check_series(
                    future_recovered.inlet_flow,
                    prediction_recovered.inlet_flow,
                    1.0e-8,
                    "recovered inlet flow");

                check_series(
                    future_recovered.outlet_pressure,
                    prediction_recovered.outlet_pressure,
                    1.0e-5,
                    "recovered outlet pressure");

                check_series(
                    future_recovered.linepack,
                    prediction_recovered.linepack,
                    1.0e-5,
                    "recovered linepack");

                std::cout
                    << "PASS M1.6e full restart parity\n";
            }

            std::cout
                << "m16e_truth_max_mass_defect="
                << future_truth.max_mass_defect << '\n'
                << "m16e_recovered_max_mass_defect="
                << future_recovered.max_mass_defect << '\n';

            require(
                std::isfinite(inlet_rmse) &&
                std::isfinite(outlet_rmse) &&
                std::isfinite(inventory_rmse) &&
                std::isfinite(inventory_relative_rmse) &&
                std::isfinite(pressure_rmse) &&
                std::isfinite(flow_rmse),
                "Nonfinite M1.6e prediction error");

            std::cout
                << "\nM1.6e changed-demand validation\n"
                << "m16e_method=" << method << '\n'
                << "m16e_scenario=" << scenario.name << '\n'
                << "m16e_target_flow_kg_s="
                << scenario.target_flow << '\n'
                << "m16e_inlet_flow_rmse_kg_s="
                << inlet_rmse << '\n'
                << "m16e_outlet_pressure_rmse_pa="
                << outlet_rmse << '\n'
                << "m16e_linepack_rmse="
                << inventory_rmse << '\n'
                << "m16e_linepack_relative_rmse="
                << inventory_relative_rmse << '\n'
                << "m16e_final_pressure_rmse_pa="
                << pressure_rmse << '\n'
                << "m16e_final_flow_rmse_kg_s="
                << flow_rmse << '\n'
                << "m16e_csv=" << csv_path << '\n';
        }

        std::cout
            << "PASS M1.6e changed-demand validation\n";

        std::cout
            << "PASS M1.6d predictive validation\n";

        std::cout
            << "PASS M1.6c regularized pipeline reconstruction\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
