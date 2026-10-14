#include "state_finder/forecast_continuation.hpp"
#include "state_finder/joint_scaled_problem.hpp"
#include "state_finder/steady_state_initializer.hpp"

#include "pipe_sim/transient.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace state_finder;

constexpr int n = 20;
constexpr int reconstruction_steps = 60;
constexpr double dt = 10.0;

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

ForwardConfiguration make_configuration(int steps)
{
    constexpr double diameter = 0.6;
    constexpr double flow = 100.0;
    constexpr double pin = 8.0e6;

    const double area =
        std::acos(-1.0)*diameter*diameter/4.0;

    return {
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
            1.0e-9,
            1.0e-12,
            30,
            false
        }
    };
}

JointInverseVariables make_reference_variables(
    const ForwardConfiguration& config)
{
    constexpr double flow = 100.0;
    constexpr double pin = 8.0e6;

    JointInverseVariables result;

    result.initial =
        initialize_steady_state(config, flow);

    result.boundary.inlet_pressure.assign(
        config.steps+1, pin);

    result.boundary.outlet_flow.resize(
        config.steps+1);

    for (int k = 0; k <= config.steps; ++k) {
        const double t = k*dt;

        const double fraction = std::clamp(
            (t-120.0)/120.0, 0.0, 1.0);

        result.boundary.outlet_flow[k] =
            flow + 10.0*fraction*fraction*
            (3.0-2.0*fraction);
    }

    return result;
}

std::vector<double> load_solution(
    const std::filesystem::path& filename)
{
    std::ifstream input(filename);

    require(
        static_cast<bool>(input),
        "Cannot open reconstruction file");

    std::string line;

    require(
        static_cast<bool>(std::getline(input, line)),
        "Empty reconstruction file");

    require(
        line == "coordinate,reconstructed_scaled_value",
        "Unexpected reconstruction CSV header");

    std::vector<double> x;

    while (std::getline(input, line)) {
        if (line.empty())
            continue;

        std::istringstream row(line);

        std::string index_field;
        std::string value_field;
        std::string extra;

        require(
            static_cast<bool>(
                std::getline(row, index_field, ',')) &&
            static_cast<bool>(
                std::getline(row, value_field, ',')) &&
            !std::getline(row, extra, ','),
            "Malformed reconstruction CSV row");

        std::size_t index_chars = 0;
        std::size_t value_chars = 0;

        const auto index =
            std::stoul(index_field, &index_chars);

        const double value =
            std::stod(value_field, &value_chars);

        require(
            index_chars == index_field.size() &&
            value_chars == value_field.size() &&
            index == x.size() &&
            std::isfinite(value),
            "Invalid reconstruction coordinate");

        x.push_back(value);
    }

    require(
        x.size() == joint_variable_dimension(
            n, reconstruction_steps),
        "Expected 163 original reconstruction coordinates");

    return x;
}

std::vector<double> pressures(
    std::span<const double> state)
{
    require(
        state.size() == static_cast<std::size_t>(2*n+2),
        "Unexpected pipeline state dimension");

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

double minimum_pressure(
    std::span<const double> state)
{
    const auto p = pressures(state);

    return *std::min_element(p.begin(), p.end());
}

double maximum_pressure_error(
    std::span<const double> reference,
    std::span<const double> estimate)
{
    const auto p_ref = pressures(reference);
    const auto p_est = pressures(estimate);

    double result = 0.0;

    for (int i = 0; i <= n; ++i)
        result = std::max(
            result,
            std::abs(p_est[i]-p_ref[i]));

    return result;
}

AdjustableBoundaryHistory make_future_boundary(
    int steps,
    double initial_inlet_pressure)
{
    AdjustableBoundaryHistory result;

    result.inlet_pressure.assign(
        steps+1, 8.0e6);

    result.outlet_flow.assign(
        steps+1, 110.0);

    // Preserve the terminal state at the handoff.
    // Subsequent samples use the prescribed future pressure.
    result.inlet_pressure[0] =
        initial_inlet_pressure;

    return result;
}

void write_forecast(
    const std::filesystem::path& output_path,
    const ForwardConfiguration& config,
    const JointTrajectory& reference,
    const JointTrajectory& estimate,
    double initial_inventory_error)
{
    require(
        reference.steps == estimate.steps,
        "Forecast trajectory length mismatch");

    require(
        reference.steps == config.steps,
        "Forecast configuration mismatch");

    require(
        !std::filesystem::exists(output_path),
        "Refusing to overwrite existing output file");

    std::ofstream output(output_path);

    require(
        static_cast<bool>(output),
        "Cannot create forecast CSV");

    output << std::setprecision(17);

    output
        << "forecast_time_s,"
        << "reference_inventory_kg,"
        << "estimated_inventory_kg,"
        << "inventory_error_kg,"
        << "inventory_error_growth_kg,"
        << "reference_min_pressure_pa,"
        << "estimated_min_pressure_pa,"
        << "maximum_pressure_error_pa,"
        << "reference_flow_imbalance_kg_s,"
        << "estimated_flow_imbalance_kg_s,"
        << "reference_mass_defect_kg,"
        << "estimated_mass_defect_kg\n";

    double maximum_absolute_inventory_error = 0.0;
    double maximum_pressure_difference = 0.0;
    double maximum_reference_mass_defect = 0.0;
    double maximum_estimated_mass_defect = 0.0;

    double previous_reference_inventory =
        inventory(reference.state(0), config);

    double previous_estimated_inventory =
        inventory(estimate.state(0), config);

    const double theta = config.parameters.theta;

    for (int k = 0; k <= config.steps; ++k) {
        const auto reference_state =
            reference.state(k);

        const auto estimated_state =
            estimate.state(k);

        const double reference_inventory =
            inventory(reference_state, config);

        const double estimated_inventory =
            inventory(estimated_state, config);

        const double inventory_error =
            estimated_inventory-reference_inventory;

        const double pressure_error =
            maximum_pressure_error(
                reference_state,
                estimated_state);

        const double reference_imbalance =
            reference.predicted.inlet_flow[k] -
            reference.predicted.outlet_flow[k];

        const double estimated_imbalance =
            estimate.predicted.inlet_flow[k] -
            estimate.predicted.outlet_flow[k];

        double reference_mass_defect = 0.0;
        double estimated_mass_defect = 0.0;

        if (k > 0) {
            const double reference_previous_imbalance =
                reference.predicted.inlet_flow[k-1] -
                reference.predicted.outlet_flow[k-1];

            const double estimated_previous_imbalance =
                estimate.predicted.inlet_flow[k-1] -
                estimate.predicted.outlet_flow[k-1];

            const double reference_theta_imbalance =
                theta*reference_imbalance +
                (1.0-theta)*reference_previous_imbalance;

            const double estimated_theta_imbalance =
                theta*estimated_imbalance +
                (1.0-theta)*estimated_previous_imbalance;

            reference_mass_defect =
                reference_inventory -
                previous_reference_inventory -
                dt*reference_theta_imbalance;

            estimated_mass_defect =
                estimated_inventory -
                previous_estimated_inventory -
                dt*estimated_theta_imbalance;
        }

        maximum_absolute_inventory_error =
            std::max(
                maximum_absolute_inventory_error,
                std::abs(inventory_error));

        maximum_pressure_difference =
            std::max(
                maximum_pressure_difference,
                pressure_error);

        maximum_reference_mass_defect =
            std::max(
                maximum_reference_mass_defect,
                std::abs(reference_mass_defect));

        maximum_estimated_mass_defect =
            std::max(
                maximum_estimated_mass_defect,
                std::abs(estimated_mass_defect));

        output
            << k*dt << ','
            << reference_inventory << ','
            << estimated_inventory << ','
            << inventory_error << ','
            << inventory_error-initial_inventory_error << ','
            << minimum_pressure(reference_state) << ','
            << minimum_pressure(estimated_state) << ','
            << pressure_error << ','
            << reference_imbalance << ','
            << estimated_imbalance << ','
            << reference_mass_defect << ','
            << estimated_mass_defect << '\n';

        previous_reference_inventory =
            reference_inventory;

        previous_estimated_inventory =
            estimated_inventory;
    }

    output.flush();

    require(
        static_cast<bool>(output),
        "Failed writing forecast CSV");

    const double final_error =
        inventory(estimate.state(config.steps), config) -
        inventory(reference.state(config.steps), config);

    std::cout << std::setprecision(17)
        << "M28_A2_FORECAST\n"
        << "forecast_horizon_s="
        << config.steps*dt << '\n'
        << "initial_inventory_error_kg="
        << initial_inventory_error << '\n'
        << "final_inventory_error_kg="
        << final_error << '\n'
        << "inventory_error_growth_kg="
        << final_error-initial_inventory_error << '\n'
        << "maximum_absolute_inventory_error_kg="
        << maximum_absolute_inventory_error << '\n'
        << "maximum_pressure_error_pa="
        << maximum_pressure_difference << '\n'
        << "maximum_reference_mass_defect_kg="
        << maximum_reference_mass_defect << '\n'
        << "maximum_estimated_mass_defect_kg="
        << maximum_estimated_mass_defect << '\n'
        << "output_file="
        << output_path.string() << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try {
        require(
            argc == 3,
            "Usage: state_finder_m28_forecast "
            "<reconstruction.csv> <output.csv>");

        const auto reconstruction_config =
            make_configuration(reconstruction_steps);

        const auto truth =
            make_reference_variables(
                reconstruction_config);

        // M27's baseline is the synthetic truth.
        const auto baseline = truth;

        const auto x =
            load_solution(argv[1]);

        const auto reconstructed_variables =
            unpack_joint_variables(
                x, baseline, JointVariableScales{});

        const auto reference_history =
            record_joint_trajectory(
                reconstruction_config, truth);

        const auto reconstructed_history =
            record_joint_trajectory(
                reconstruction_config,
                reconstructed_variables);

        const auto reference_terminal =
            reference_history.state(reconstruction_steps);

        const auto reconstructed_terminal =
            reconstructed_history.state(reconstruction_steps);

        const double initial_inventory_error =
            inventory(
                reconstructed_terminal,
                reconstruction_config) -
            inventory(
                reference_terminal,
                reconstruction_config);

        const std::filesystem::path output_base(argv[2]);

        require(
            output_base.extension() == ".csv",
            "Output filename must have .csv extension");

        for (const int horizon : {600, 1200, 1800}) {
            const auto forecast_config =
                make_configuration(
                    static_cast<int>(horizon/dt));

            const auto reference_boundary =
                make_future_boundary(
                    forecast_config.steps,
                    reference_terminal[0]);

            const auto reconstructed_boundary =
                make_future_boundary(
                    forecast_config.steps,
                    reconstructed_terminal[0]);

            const auto reference_forecast =
                continue_joint_trajectory(
                    forecast_config,
                    reference_terminal,
                    reference_boundary);

            const auto reconstructed_forecast =
                continue_joint_trajectory(
                    forecast_config,
                    reconstructed_terminal,
                    reconstructed_boundary);

            const auto output_path =
                output_base.parent_path() /
                (
                    output_base.stem().string() +
                    "_" + std::to_string(horizon) +
                    "s.csv"
                );

            write_forecast(
                output_path,
                forecast_config,
                reference_forecast,
                reconstructed_forecast,
                initial_inventory_error);
        }

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';

        return 1;
    }
}
