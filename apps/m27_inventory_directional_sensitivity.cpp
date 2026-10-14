#include "state_finder/joint_tangent_linear.hpp"
#include "state_finder/steady_state_initializer.hpp"

#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace state_finder;

namespace {

constexpr int n = 20;
constexpr int steps = 60;
constexpr double dt = 10.0;

std::vector<double> load_vector(
    const std::filesystem::path& path)
{
    std::ifstream file(path);

    if (!file)
        throw std::runtime_error(
            "Cannot open: " + path.string());

    std::string line;
    std::getline(file, line);

    if (line != "coordinate,reconstructed_scaled_value")
        throw std::runtime_error(
            "Unexpected CSV header: " + path.string());

    std::vector<double> values;

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::istringstream input(line);

        std::string index_field;
        std::string value_field;
        std::string extra;

        if (!std::getline(input, index_field, ',') ||
            !std::getline(input, value_field, ',') ||
            std::getline(input, extra, ','))
            throw std::runtime_error(
                "Malformed CSV row");

        const auto index =
            std::stoul(index_field);

        if (index != values.size())
            throw std::runtime_error(
                "Unexpected coordinate ordering");

        const double value =
            std::stod(value_field);

        if (!std::isfinite(value))
            throw std::runtime_error(
                "Nonfinite coordinate");

        values.push_back(value);
    }

    if (values.size() != 163)
        throw std::runtime_error(
            "Expected 163 coordinates");

    return values;
}

std::vector<double> nodal_pressures(
    std::span<const double> state)
{
    std::vector<double> pressures(n+1);

    pressures[0] = state[0];

    for (int i = 1; i <= n; ++i)
        pressures[i] = state[2*i+1];

    return pressures;
}

double terminal_inventory(
    std::span<const double> state,
    const ForwardConfiguration& config)
{
    return pipe_sim::linepack_cz(
        n,
        nodal_pressures(state),
        config.parameters);
}

ForwardConfiguration configuration()
{
    constexpr double diameter = 0.6;
    constexpr double pressure = 8.0e6;
    constexpr double flow = 100.0;

    const double pi = std::acos(-1.0);
    const double area =
        pi*diameter*diameter/4.0;

    return ForwardConfiguration{
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
            pressure, flow, flow, 1.0
        },
        pipe_sim::NewtonOptions{
            1.0e-9, 1.0e-12, 30, false
        }
    };
}

JointInverseVariables reference_variables(
    const ForwardConfiguration& config)
{
    constexpr double pressure = 8.0e6;
    constexpr double flow = 100.0;

    JointInverseVariables truth;

    truth.initial =
        initialize_steady_state(config, flow);

    truth.boundary.inlet_pressure.assign(
        steps+1, pressure);

    truth.boundary.outlet_flow.assign(
        steps+1, flow);

    for (int k = 0; k <= steps; ++k) {
        const double t = k*dt;

        const double fraction =
            std::clamp(
                (t-120.0)/120.0,
                0.0,
                1.0);

        const double smooth =
            fraction*fraction*(3.0-2.0*fraction);

        truth.boundary.outlet_flow[k] =
            flow+10.0*smooth;
    }

    return truth;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const bool gradient_mode =
            argc == 4
            && std::string(argv[1]) == "--gradient";

        if (!gradient_mode && argc != 3)
            throw std::runtime_error(
                "Usage: m27_inventory_directional_sensitivity "
                "<base.csv> <direction.csv> "
                "OR --gradient <base.csv> <output.csv>");

        const auto base =
            load_vector(argv[gradient_mode ? 2 : 1]);

        const std::vector<double> direction =
            gradient_mode
                ? std::vector<double>(163, 0.0)
                : load_vector(argv[2]);

        const auto config = configuration();

        const auto truth =
            reference_variables(config);

        const JointVariableScales scales{};

        const auto variables =
            unpack_joint_variables(
                base,
                truth,
                scales);

        const auto trajectory =
            record_joint_trajectory(
                config,
                variables);

        if (gradient_mode) {
            const std::filesystem::path output_path =
                argv[3];

            if (std::filesystem::exists(output_path))
                throw std::runtime_error(
                    "Refusing to overwrite: "
                    + output_path.string());

            // Compute all 163 coordinate derivatives while
            // reusing the same nonlinear trajectory.
            std::vector<double> gradient(163);

            double max_backward_error = 0.0;

            for (std::size_t j = 0; j < 163; ++j) {
                std::vector<double> unit(163, 0.0);
                unit[j] = 1.0;

                const auto tangent =
                    evaluate_joint_tangent_linear(
                        config,
                        trajectory,
                        unit,
                        scales);

                gradient[j] = terminal_inventory(
                    tangent.state(steps, n),
                    config);

                max_backward_error = std::max(
                    max_backward_error,
                    tangent.max_linear_backward_error);
            }

            std::ofstream output(output_path);

            if (!output)
                throw std::runtime_error(
                    "Cannot create gradient CSV");

            output << "coordinate,inventory_gradient_kg\n";
            output << std::setprecision(17);

            for (std::size_t j = 0;
                 j < gradient.size();
                 ++j)
                output << j << ','
                       << gradient[j] << '\n';

            if (!output)
                throw std::runtime_error(
                    "Gradient CSV write failed");

            std::cout << std::setprecision(17)
                << "gradient_dimension="
                << gradient.size() << '\n'
                << "max_linear_backward_error="
                << max_backward_error << '\n'
                << "gradient_output="
                << output_path << '\n';

            return 0;
        }

        const auto tangent =
            evaluate_joint_tangent_linear(
                config,
                trajectory,
                direction,
                scales);

        const double inventory =
            terminal_inventory(
                trajectory.state(steps),
                config);

        const double derivative =
            terminal_inventory(
                tangent.state(steps,n),
                config);

        const auto reference =
            record_joint_trajectory(
                config,
                truth);

        const double true_inventory =
            terminal_inventory(
                reference.state(steps),
                config);

        const double inventory_error =
            inventory-true_inventory;

        std::cout << std::setprecision(17)
            << "terminal_inventory_kg="
            << inventory << '\n'
            << "terminal_inventory_true_kg="
            << true_inventory << '\n'
            << "terminal_inventory_error_kg="
            << inventory_error << '\n'
            << "terminal_inventory_relative_error_pct="
            << 100.0*inventory_error/true_inventory
            << '\n'
            << "directional_inventory_derivative_kg="
            << derivative << '\n'
            << "max_linear_backward_error="
            << tangent.max_linear_backward_error
            << '\n';

        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "M27 inventory directional sensitivity FAIL: "
            << e.what() << '\n';

        return 1;
    }
}
