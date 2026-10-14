#include "state_finder/calibrated_joint_gauss_newton.hpp"
#include "state_finder/steady_state_initializer.hpp"

#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace state_finder;

namespace {

constexpr int n = 20;
constexpr int steps = 60;
constexpr double dt = 10.0;
constexpr double tau = 120.0;

std::vector<double> load_noise(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error(
            "Cannot open measurement file: " + filename);

    std::string line;
    std::getline(file, line);

    if (line !=
        "measurement_index,noise_normalized,whitened_noise")
        throw std::runtime_error(
            "Unexpected measurement CSV header");

    std::vector<double> noise;
    noise.reserve(4*(steps+1));

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::istringstream input(line);
        std::string index_field;
        std::string noise_field;
        std::string whitened_field;
        std::string extra;

        if (!std::getline(input, index_field, ',') ||
            !std::getline(input, noise_field, ',') ||
            !std::getline(input, whitened_field, ',') ||
            std::getline(input, extra, ','))
            throw std::runtime_error(
                "Malformed measurement CSV row");

        const auto index = std::stoul(index_field);

        if (index != noise.size())
            throw std::runtime_error(
                "Unexpected measurement CSV ordering");

        const double value = std::stod(noise_field);

        if (!std::isfinite(value))
            throw std::runtime_error(
                "Nonfinite measurement noise");

        noise.push_back(value);
    }

    if (noise.size() != 4*(steps+1))
        throw std::runtime_error(
            "Expected exactly 244 measurements");

    return noise;
}

std::vector<double> pressures(
    std::span<const double> state)
{
    std::vector<double> result(n+1);
    result[0] = state[0];

    for (int i = 1; i <= n; ++i)
        result[i] = state[2*i+1];

    return result;
}

double inventory(
    std::span<const double> state,
    const ForwardConfiguration& config)
{
    return pipe_sim::linepack_cz(
        n, pressures(state), config.parameters);
}

double norm(const std::vector<double>& values)
{
    return std::sqrt(
        std::inner_product(
            values.begin(),
            values.end(),
            values.begin(),
            0.0));
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 6)
            throw std::runtime_error(
                "Usage: state_finder_m27_calibrated_reconstruction "
                "<corrected-noise.csv> "
                "<max-iterations> "
                "<sigma-cal-kpa> "
                "<rho-cal> "
                "<solution-output.csv>");

        const std::string measurement_file = argv[1];

        const std::string iteration_argument = argv[2];

        std::size_t parsed_characters = 0;

        const int max_iterations = std::stoi(
            iteration_argument,
            &parsed_characters);

        if (parsed_characters != iteration_argument.size()
            || max_iterations < 1
            || max_iterations > 1000)
            throw std::runtime_error(
                "Invalid maximum iteration count");

        const double sigma_cal_kpa = std::stod(argv[3]);
        const double rho_cal = std::stod(argv[4]);

        const PressureCalibrationPrior prior{
            sigma_cal_kpa, rho_cal
        };

        prior.validate();

        const std::filesystem::path solution_file = argv[5];

        if (std::filesystem::exists(solution_file))
            throw std::runtime_error(
                "Refusing to overwrite existing solution: "
                + solution_file.string());

        constexpr double diameter = 0.6;
        constexpr double flow = 100.0;
        constexpr double pin = 8.0e6;

        const double pi = std::acos(-1.0);
        const double area = pi*diameter*diameter/4.0;

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                50000.0/n,
                dt,
                0.65,
                diameter,
                area,
                288.15,
                0.9,
                500.0,
                1.1e-5,
                4.5e-5
            },
            pipe_sim::DemandRamp{
                pin, flow, flow, 1.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9, 1.0e-12, 30, false
            }
        };

        JointInverseVariables truth;

        truth.initial =
            initialize_steady_state(config, flow);

        truth.boundary.inlet_pressure.assign(
            steps+1, pin);

        truth.boundary.outlet_flow.assign(
            steps+1, flow);

        for (int k = 0; k <= steps; ++k) {
            const double t = k*dt;

            const double fraction = std::clamp(
                (t-120.0)/120.0,
                0.0,
                1.0);

            truth.boundary.outlet_flow[k] =
                flow + 10.0*fraction*fraction*
                (3.0-2.0*fraction);
        }

        const auto reference =
            record_joint_trajectory(config, truth);

        BoundaryObservationRecord observations;
        observations.include_initial = true;
        observations.measured = reference.predicted;

        const auto noise = load_noise(measurement_file);

        // Engineering-normalized measurement convention:
        // [pin/1e4, min/1, pout/1e4, mout/1].
        for (int k = 0; k <= steps; ++k) {
            observations.measured.inlet_pressure[k] +=
                1.0e4*noise[4*k];

            observations.measured.inlet_flow[k] +=
                noise[4*k+1];

            observations.measured.outlet_pressure[k] +=
                1.0e4*noise[4*k+2];

            observations.measured.outlet_flow[k] +=
                noise[4*k+3];
        }

        const JointVariableScales scales{};

        TemporalMeasurementCovariance covariance(
            steps+1,
            dt,
            tau);

        const std::size_t dimension =
            joint_variable_dimension(n, steps);

        if (dimension != 163 ||
            covariance.dimension() != 244)
            throw std::runtime_error(
                "Unexpected M27 problem dimensions");

        // The baseline equals the synthetic truth.
        // This is an initial guess, not a constraint.
        const JointInverseVariables baseline = truth;

        const std::vector<double> initial(
            dimension+2, 0.0);

        const ObjectiveFunction objective =
            [&](const std::vector<double>& x,
                bool need_gradient)
            {
                return evaluate_calibrated_joint_objective(
                    config,
                    observations,
                    baseline,
                    x,
                    covariance,
                    prior,
                    scales,
                    need_gradient);
            };

        const auto factory =
            make_calibrated_joint_gauss_newton_factory(
                config,
                observations,
                baseline,
                covariance,
                prior,
                scales);

        GaussNewtonOptions options;
        options.max_iterations = max_iterations;
        options.max_cg_iterations = 200;
        options.gradient_tolerance = 1.0e-7;
        options.initial_damping = 1.0e-3;

        const auto initial_evaluation =
            objective(initial, true);

        if (!initial_evaluation.valid)
            throw std::runtime_error(
                "Initial objective evaluation failed");

        const auto start =
            std::chrono::steady_clock::now();

        GaussNewtonDiagnostics diagnostics;

        const auto result =
            minimize_gauss_newton(
                objective,
                initial,
                factory,
                options,
                &diagnostics);

        const auto stop =
            std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(
                stop-start).count();

        const auto recovered_variables =
            unpack_joint_variables(
                std::vector<double>(
                    result.variables.begin(),
                    result.variables.begin()+dimension),
                baseline,
                scales);

        const auto recovered =
            record_joint_trajectory(
                config,
                recovered_variables);

        const double terminal_inventory_true =
            inventory(reference.state(steps), config);

        const double terminal_inventory_estimated =
            inventory(recovered.state(steps), config);

        const double terminal_inventory_error =
            terminal_inventory_estimated
            - terminal_inventory_true;

        if (!std::isfinite(terminal_inventory_true)
            || terminal_inventory_true <= 0.0)
            throw std::runtime_error(
                "Invalid reference terminal inventory");

        const double terminal_inventory_relative_error =
            terminal_inventory_error
            / terminal_inventory_true;

        const double terminal_inventory_absolute_error =
            std::abs(terminal_inventory_error);

        const double terminal_inventory_absolute_relative_error =
            std::abs(terminal_inventory_relative_error);

        double minimum_pressure =
            std::numeric_limits<double>::infinity();

        double maximum_mass_defect = 0.0;

        for (int k = 0; k <= steps; ++k) {
            const auto state = recovered.state(k);

            for (double pressure : pressures(state))
                minimum_pressure =
                    std::min(minimum_pressure, pressure);

            if (k == 0)
                continue;

            const double previous_inventory =
                inventory(recovered.state(k-1), config);

            const double current_inventory =
                inventory(state, config);

            const double theta = config.parameters.theta;

            const double inlet_weighted =
                (1.0-theta)*recovered.state(k-1)[1]
                +theta*state[1];

            const double outlet_weighted =
                (1.0-theta)*
                    recovered_variables.boundary.outlet_flow[k-1]
                +theta*
                    recovered_variables.boundary.outlet_flow[k];

            const double defect =
                (current_inventory-previous_inventory)
                -dt*(inlet_weighted-outlet_weighted);

            maximum_mass_defect = std::max(
                maximum_mass_defect,
                std::abs(defect));
        }

        const auto final_evaluation =
            objective(result.variables, true);

        if (!final_evaluation.valid)
            throw std::runtime_error(
                "Final objective evaluation failed");

        const auto weighted_noise =
            covariance.apply_inverse(noise);

        const double truth_objective =
            0.5*std::inner_product(
                noise.begin(),
                noise.end(),
                weighted_noise.begin(),
                0.0);

        std::cout << std::setprecision(17)
            << "M27_NATIVE_CALIBRATED_RECONSTRUCTION\n"
            << "dimensions=" << dimension+2 << '\n'
            << "max_iterations=" << max_iterations << '\n'
            << "sigma_cal_kpa=" << sigma_cal_kpa << '\n'
            << "rho_cal=" << rho_cal << '\n'
            << "calibration_inlet_kpa="
            << result.variables[dimension] << '\n'
            << "calibration_outlet_kpa="
            << result.variables[dimension+1] << '\n'
            << "calibration_inlet_pa="
            << 1000.0*result.variables[dimension] << '\n'
            << "calibration_outlet_pa="
            << 1000.0*result.variables[dimension+1] << '\n'
            << "measurements=" << noise.size() << '\n'
            << "tau_s=" << tau << '\n'
            << "normalized_sensor_sigma=0.5\n"
            << "truth_objective=" << truth_objective << '\n'
            << "initial_objective="
            << initial_evaluation.objective << '\n'
            << "final_objective="
            << result.final_objective << '\n'
            << "verified_final_objective="
            << final_evaluation.objective << '\n'
            << "calibration_prior_penalty="
            << 0.5*prior.quadratic_form(
                std::span<const double>(
                    result.variables.data()+dimension, 2))
            << '\n'
            << "measurement_objective="
            << final_evaluation.objective
               -0.5*prior.quadratic_form(
                   std::span<const double>(
                       result.variables.data()+dimension, 2))
            << '\n'
            << "iterations=" << result.iterations << '\n'
            << "converged=" << result.converged << '\n'
            << "reason=" << result.reason << '\n'
            << "gradient_norm=" << result.gradient_norm << '\n'
            << "objective_evaluations="
            << result.objective_evaluations << '\n'
            << "gradient_evaluations="
            << result.gradient_evaluations << '\n'
            << "pcg_solves="
            << diagnostics.pcg_solves << '\n'
            << "maximum_true_residual_ratio="
            << diagnostics.maximum_true_residual_ratio << '\n'
            << "final_damping="
            << diagnostics.final_damping << '\n'
            << "final_forcing_tolerance="
            << diagnostics.final_forcing_tolerance << '\n'
            << "pcg_iterations="
            << diagnostics.pcg_iterations << '\n'
            << "operator_calls="
            << diagnostics.operator_calls << '\n'
            << "damping_retries="
            << diagnostics.damping_retries << '\n'
            << "gradient_fallbacks="
            << diagnostics.gradient_fallbacks << '\n'
            << "scaled_solution_norm="
            << norm(result.variables) << '\n'
            << "terminal_inventory_true_kg="
            << terminal_inventory_true << '\n'
            << "terminal_inventory_estimated_kg="
            << terminal_inventory_estimated << '\n'
            << "terminal_inventory_error_kg="
            << terminal_inventory_error << '\n'
            << "terminal_inventory_absolute_error_kg="
            << terminal_inventory_absolute_error << '\n'
            << "terminal_inventory_relative_error="
            << terminal_inventory_relative_error << '\n'
            << "terminal_inventory_relative_error_pct="
            << 100.0*terminal_inventory_relative_error << '\n'
            << "terminal_inventory_absolute_relative_error_pct="
            << 100.0*terminal_inventory_absolute_relative_error
            << '\n'
            << "minimum_pressure_pa="
            << minimum_pressure << '\n'
            << "maximum_mass_defect_kg="
            << maximum_mass_defect << '\n'
            << "elapsed_seconds=" << seconds << '\n';

        if (result.final_objective >
            initial_evaluation.objective + 1.0e-8)
            throw std::runtime_error(
                "Native reconstruction increased objective");

        {
            std::ofstream output(solution_file);

            if (!output)
                throw std::runtime_error(
                    "Cannot create solution file: "
                    + solution_file.string());

            output << "coordinate,reconstructed_scaled_value\n";

            output << std::setprecision(17);

            for (std::size_t j = 0;
                 j < result.variables.size();
                 ++j)
            {
                output
                    << j << ','
                    << result.variables[j]
                    << '\n';
            }

            output.flush();

            if (!output)
                throw std::runtime_error(
                    "Failed writing solution file");
        }

        std::cout
            << "solution_file="
            << solution_file.string()
            << '\n';

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
