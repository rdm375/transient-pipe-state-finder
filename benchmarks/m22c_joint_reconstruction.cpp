#include "state_finder/joint_scaled_problem.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using namespace state_finder;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double rmse(
    const std::vector<double>& a,
    const std::vector<double>& b,
    std::size_t first = 0)
{
    require(a.size() == b.size() && first < a.size(),
            "RMSE dimension mismatch");

    double sum = 0.0;

    for (std::size_t i = first; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        sum += d*d;
    }

    return std::sqrt(sum / (a.size() - first));
}

struct Metrics {
    double pressure_rmse = 0.0;
    double flow_rmse = 0.0;
    double inlet_flow_error = 0.0;
    double inlet_pressure_rmse = 0.0;
    double outlet_flow_rmse = 0.0;
    double measurement_objective = 0.0;
    double regularization = 0.0;
    double total_objective = 0.0;
    double mass_defect = 0.0;
};

Metrics measure(
    const JointInverseVariables& candidate,
    const JointInverseVariables& truth,
    const JointScaledProblem& problem,
    const std::vector<double>& x,
    const ForwardConfiguration& config)
{
    Metrics m;

    m.pressure_rmse = rmse(
        candidate.initial.pressure,
        truth.initial.pressure,
        1);

    m.flow_rmse = rmse(
        candidate.initial.flow,
        truth.initial.flow);

    m.inlet_flow_error = std::abs(
        candidate.initial.inlet_flow -
        truth.initial.inlet_flow);

    m.inlet_pressure_rmse = rmse(
        candidate.boundary.inlet_pressure,
        truth.boundary.inlet_pressure);

    m.outlet_flow_rmse = rmse(
        candidate.boundary.outlet_flow,
        truth.boundary.outlet_flow);

    const auto objective = problem.evaluate(x);

    m.measurement_objective = objective.measurement;

    m.regularization =
        objective.initial_regularization +
        objective.boundary_regularization +
        objective.boundary_smoothness;

    m.total_objective = objective.total;

    auto pressure = candidate.initial.pressure;
    auto flow = candidate.initial.flow;
    double inlet = candidate.initial.inlet_flow;

    pressure[0] = candidate.boundary.inlet_pressure[0];

    const auto trajectory = integrate_boundary_history(
        config.n,
        pressure,
        flow,
        inlet,
        candidate.boundary,
        config.steps,
        config.parameters,
        config.newton);

    require(trajectory.integration.info == 0,
            "Metric trajectory integration failed");

    m.mass_defect =
        trajectory.integration.max_mass_defect;

    return m;
}

void write_metrics(
    std::ofstream& output,
    const std::string& method,
    const std::string& stage,
    const Metrics& m)
{
    output
        << method << ','
        << stage << ','
        << m.pressure_rmse << ','
        << m.flow_rmse << ','
        << m.inlet_flow_error << ','
        << m.inlet_pressure_rmse << ','
        << m.outlet_flow_rmse << ','
        << m.measurement_objective << ','
        << m.regularization << ','
        << m.total_objective << ','
        << m.mass_defect << '\n';
}

void print_metrics(
    const std::string& label,
    const Metrics& m)
{
    std::cout
        << label << '\n'
        << "  pressure_rmse_pa="
        << m.pressure_rmse << '\n'
        << "  flow_rmse_kg_s="
        << m.flow_rmse << '\n'
        << "  inlet_flow_error_kg_s="
        << m.inlet_flow_error << '\n'
        << "  inlet_pressure_history_rmse_pa="
        << m.inlet_pressure_rmse << '\n'
        << "  outlet_flow_history_rmse_kg_s="
        << m.outlet_flow_rmse << '\n'
        << "  measurement_objective="
        << m.measurement_objective << '\n'
        << "  total_objective="
        << m.total_objective << '\n'
        << "  mass_defect_kg_s="
        << m.mass_defect << '\n';
}

} // namespace

int main()
{
    try {
        constexpr int n = 10;
        constexpr int steps = 12;

        const double pi = std::acos(-1.0);

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                1000.0,
                15.0,
                0.65,
                1.0,
                pi/4.0,
                288.15,
                0.9,
                500.0,
                1.1e-5,
                4.5e-5
            },
            pipe_sim::DemandRamp{
                8.0e6, 100.0, 100.0, 1.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9, 1.0e-12, 30, false
            }
        };

        JointInverseVariables truth;

        truth.initial.pressure.resize(n+1);
        truth.initial.flow.resize(n);

        truth.initial.inlet_flow = 100.0;

        truth.boundary.inlet_pressure.resize(steps+1);
        truth.boundary.outlet_flow.resize(steps+1);

        // Smooth, nonconstant synthetic initial state.
        for (int i = 0; i <= n; ++i) {
            const double s = static_cast<double>(i)/n;

            truth.initial.pressure[i] =
                8.0e6 - 12000.0*s
                + 1500.0*std::sin(pi*s);
        }

        for (int i = 0; i < n; ++i) {
            const double s =
                (static_cast<double>(i)+0.5)/n;

            truth.initial.flow[i] =
                100.0 + 0.5*std::sin(2.0*pi*s);
        }

        for (int k = 0; k <= steps; ++k) {
            const double s =
                static_cast<double>(k)/steps;

            truth.boundary.inlet_pressure[k] =
                8.0e6
                + 1000.0*std::sin(2.0*pi*s);

            truth.boundary.outlet_flow[k] =
                100.0
                + 1.5*s
                + 0.2*std::sin(2.0*pi*s);
        }

        // Ensure the initial state and boundary input
        // describe the same inlet pressure at t=0.
        truth.initial.pressure[0] =
            truth.boundary.inlet_pressure[0];

        // Generate noise-free synthetic observations.
        auto pressure = truth.initial.pressure;
        auto flow = truth.initial.flow;
        double inlet = truth.initial.inlet_flow;

        const auto reference = integrate_boundary_history(
            n,
            pressure,
            flow,
            inlet,
            truth.boundary,
            steps,
            config.parameters,
            config.newton);

        require(reference.integration.info == 0,
                "Truth integration failed");

        BoundaryObservationRecord observations;

        observations.include_initial = true;
        observations.measured = reference.predicted;

        observations.scales.inlet_pressure = 1.0e4;
        observations.scales.inlet_flow = 1.0;
        observations.scales.outlet_pressure = 1.0e4;
        observations.scales.outlet_flow = 1.0;

        // Deliberately imperfect baseline.
        auto baseline = truth;

        for (int i = 1; i <= n; ++i) {
            const double s = static_cast<double>(i)/n;

            baseline.initial.pressure[i] +=
                3000.0*std::sin(pi*s) + 1000.0*s;
        }

        for (int i = 0; i < n; ++i) {
            const double s =
                (static_cast<double>(i)+0.5)/n;

            baseline.initial.flow[i] +=
                0.4*std::cos(pi*s);
        }

        baseline.initial.inlet_flow += 0.4;

        for (int k = 0; k <= steps; ++k) {
            const double s =
                static_cast<double>(k)/steps;

            baseline.boundary.inlet_pressure[k] +=
                700.0*std::cos(2.0*pi*s);

            baseline.boundary.outlet_flow[k] +=
                0.3*std::sin(pi*s) + 0.1;
        }

        baseline.initial.pressure[0] =
            baseline.boundary.inlet_pressure[0];

        JointVariableScales scales;

        JointRegularization regularization;
        regularization.initial_weight = 1.0e-4;
        regularization.boundary_weight = 1.0e-2;
        regularization.boundary_smoothness = 1.0e-2;

        JointScaledProblem problem(
            config,
            observations,
            baseline,
            scales,
            regularization);

        const std::size_t dimension =
            joint_variable_dimension(n, steps);

        const std::size_t state_dimension =
            static_cast<std::size_t>(2*n+1);

        std::vector<double> x0(dimension, 0.0);

        const auto before = measure(
            baseline, truth, problem, x0, config);

        // Full joint optimization.
        OptimizationOptions options;
        options.max_iterations = 35;
        options.history_size = 10;

        const auto joint_callback =
            [&](const std::vector<double>& x,
                bool need_gradient)
            {
                return problem.evaluate_for_optimizer(
                    x, need_gradient);
            };

        const auto joint_result = minimize_lbfgs(
            joint_callback,
            x0,
            options);

        require(std::isfinite(joint_result.final_objective),
                "Nonfinite joint objective");

        require(joint_result.final_objective <
                    joint_result.initial_objective,
                "Joint optimizer did not reduce objective");

        const auto joint_variables = unpack_joint_variables(
            joint_result.variables,
            baseline,
            scales);

        const auto after_joint = measure(
            joint_variables,
            truth,
            problem,
            joint_result.variables,
            config);

        // Control: optimize initial state only.
        //
        // Boundary histories remain fixed at the
        // imperfect baseline values.
        const auto fixed_callback =
            [&](const std::vector<double>& state_x,
                bool need_gradient)
            {
                OptimizationEvaluation result;

                if (state_x.size() != state_dimension)
                    return result;

                std::vector<double> full_x(
                    dimension, 0.0);

                std::copy(
                    state_x.begin(),
                    state_x.end(),
                    full_x.begin());

                const auto evaluation =
                    problem.evaluate_for_optimizer(
                        full_x, need_gradient);

                if (!evaluation.valid)
                    return result;

                result.objective = evaluation.objective;
                result.valid = true;

                if (need_gradient) {
                    result.gradient.assign(
                        evaluation.gradient.begin(),
                        evaluation.gradient.begin() +
                            state_dimension);
                }

                return result;
            };

        const std::vector<double> fixed_x0(
            state_dimension, 0.0);

        const auto fixed_result = minimize_lbfgs(
            fixed_callback,
            fixed_x0,
            options);

        require(std::isfinite(fixed_result.final_objective),
                "Nonfinite fixed-boundary objective");

        std::vector<double> fixed_full_x(
            dimension, 0.0);

        std::copy(
            fixed_result.variables.begin(),
            fixed_result.variables.end(),
            fixed_full_x.begin());

        const auto fixed_variables = unpack_joint_variables(
            fixed_full_x,
            baseline,
            scales);

        const auto after_fixed = measure(
            fixed_variables,
            truth,
            problem,
            fixed_full_x,
            config);

        std::filesystem::create_directories(
            "benchmarks/results-m22c");

        std::ofstream csv(
            "benchmarks/results-m22c/"
            "joint_reconstruction_summary.csv");

        require(csv.good(), "Cannot create benchmark CSV");

        csv << std::setprecision(17);

        csv
            << "method,stage,"
            << "pressure_rmse_pa,"
            << "flow_rmse_kg_s,"
            << "inlet_flow_error_kg_s,"
            << "inlet_pressure_history_rmse_pa,"
            << "outlet_flow_history_rmse_kg_s,"
            << "measurement_objective,"
            << "regularization,"
            << "total_objective,"
            << "max_mass_defect_kg_s\n";

        write_metrics(csv, "baseline", "before", before);
        write_metrics(csv, "joint", "after", after_joint);
        write_metrics(csv, "fixed_boundary", "after", after_fixed);

        csv.close();

        print_metrics("BASELINE", before);
        print_metrics("JOINT", after_joint);
        print_metrics("FIXED BOUNDARY", after_fixed);

        std::cout
            << "joint_variables=" << dimension << '\n'
            << "joint_iterations="
            << joint_result.iterations << '\n'
            << "joint_converged="
            << joint_result.converged << '\n'
            << "joint_reason="
            << joint_result.reason << '\n'
            << "fixed_iterations="
            << fixed_result.iterations << '\n'
            << "fixed_converged="
            << fixed_result.converged << '\n'
            << "fixed_reason="
            << fixed_result.reason << '\n'
            << "reference_mass_defect_kg_s="
            << reference.integration.max_mass_defect << '\n';

        std::cout
            << "PASS M2.2c joint reconstruction benchmark\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';

        return 1;
    }
}
