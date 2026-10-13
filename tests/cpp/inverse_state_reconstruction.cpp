#include "state_finder/adjoint.hpp"
#include "state_finder/objective.hpp"
#include "state_finder/optimizer.hpp"
#include "state_finder/scaled_initial_state.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
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

        if (argc > 2)
            throw std::runtime_error(
                "Usage: inverse_state_reconstruction "
                "[MAX_ITERATIONS]");

        if (argc == 2)
            max_iterations = std::stoi(argv[1]);

        require(
            max_iterations > 0,
            "MAX_ITERATIONS must be positive");

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

        const auto result =
            minimize_lbfgs(objective, x0, options);

        const auto recovered =
            unpack_scaled_state(
                result.variables,
                baseline,
                scales);

        const auto initial_errors =
            reconstruction_errors(candidate, truth);

        const auto final_errors =
            reconstruction_errors(recovered, truth);

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
                final_objective_check.value -
                result.final_objective);

        require(
            parity_error <
                1.0e-10 *
                std::max(
                    1.0,
                    std::abs(result.final_objective)),
            "Final objective parity failure");

        std::cout
            << "PASS M1.6a pipeline reconstruction\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
