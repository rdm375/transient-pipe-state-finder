#include "state_finder/joint_trajectory.hpp"
#include "state_finder/joint_scaled_problem.hpp"
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

double trajectory_linepack(
    const JointTrajectory& trajectory,
    int k,
    const ForwardConfiguration& config)
{
    const int n = config.n;
    const auto state = trajectory.state(k);

    std::vector<double> nodal(
        static_cast<std::size_t>(n + 1)
    );

    nodal[0] = state[0];

    for (int i = 0; i < n; ++i) {
        nodal[i + 1] = state[2*i + 3];
    }

    return pipe_sim::linepack_cz(
        n, nodal, config.parameters
    );
}


std::vector<std::vector<double>> read_modes(
    const std::string& filename,
    std::size_t dimension)
{
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error(
            "Cannot open mode file: " + filename
        );

    std::vector<std::vector<double>> rows;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream stream(line);
        std::string field;
        std::vector<double> row;

        while (std::getline(stream, field, ',')) {
            row.push_back(std::stod(field));
        }

        rows.push_back(std::move(row));
    }

    if (rows.size() != dimension)
        throw std::runtime_error(
            "Incorrect mode matrix row count"
        );

    for (const auto& row : rows) {
        if (row.size() != dimension)
            throw std::runtime_error(
                "Incorrect mode matrix column count"
            );
    }

    return rows;
}

} // namespace

int main()
{
    try {
        constexpr int n = 20;
        constexpr int steps = 60;

        constexpr double length = 50000.0;
        constexpr double dt = 10.0;
        constexpr double diameter = 0.6;
        constexpr double pressure = 8.0e6;
        constexpr double flow = 100.0;

        const double pi = std::acos(-1.0);
        const double area = pi * diameter * diameter / 4.0;

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                length / n,
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

        JointInverseVariables truth;

        truth.initial =
            initialize_steady_state(config, flow);

        truth.boundary.inlet_pressure.assign(
            steps + 1, pressure
        );

        truth.boundary.outlet_flow.assign(
            steps + 1, flow
        );

        for (int k = 0; k <= steps; ++k) {
            const double t = k * dt;

            const double fraction = std::clamp(
                (t - 120.0) / 120.0,
                0.0,
                1.0
            );

            const double smooth =
                fraction * fraction * (3.0 - 2.0 * fraction);

            truth.boundary.outlet_flow[k] =
                flow + 10.0 * smooth;
        }

        const auto reference =
            record_joint_trajectory(config, truth);

        constexpr std::size_t initial_dimension = 41;

        const auto modes = read_modes(
            "benchmarks/results-m24/"
            "initial_observability_modes.csv",
            initial_dimension
        );

        const std::size_t full_dimension =
            joint_variable_dimension(n, steps);

        JointVariableScales scales;

        const auto output =
            std::filesystem::path(
                "benchmarks/results-m24/mode_decay.csv"
            );

        std::ofstream file(output);

        if (!file)
            throw std::runtime_error(
                "Cannot create mode-decay output"
            );

        file << std::setprecision(17);

        file << "mode,time_s,"
             << "pressure_linf_Pa,"
             << "flow_linf_kg_s,"
             << "initial_scaled_amplitude,"
             << "inventory_error_kg,"
             << "inventory_change_error_kg,"
             << "mass_conservation_defect_kg\n";

        // The final columns are the weakest singular modes.
        constexpr int modes_to_test = 6;

        // Dimensionless displacement in optimization space.
        constexpr double amplitude = 1.0;

        for (int m = 0; m < modes_to_test; ++m) {
            const std::size_t column =
                initial_dimension - 1 - m;

            std::vector<double> direction(
                full_dimension, 0.0
            );

            for (std::size_t i = 0;
                 i < initial_dimension;
                 ++i)
            {
                direction[i] =
                    amplitude * modes[i][column];
            }

            auto perturbed = unpack_joint_variables(
                direction, truth, scales
            );

            // Fixed-boundary experiment.
            perturbed.boundary = truth.boundary;

            const auto trajectory =
                record_joint_trajectory(config, perturbed);

            const double initial_inventory_error =
                trajectory_linepack(trajectory, 0, config)
                - trajectory_linepack(reference, 0, config);

            double previous_inventory_error =
                initial_inventory_error;

            for (int k = 0; k <= steps; ++k) {
                const auto a = reference.state(k);
                const auto b = trajectory.state(k);

                double pressure_error = 0.0;
                double flow_error = 0.0;

                // State ordering:
                // [p0, inlet_flow, face_flow0, p1, ...]
                for (int i = 0; i <= n; ++i) {
                    const std::size_t index =
                        (i == 0) ? 0 : 2*i + 1;

                    pressure_error = std::max(
                        pressure_error,
                        std::abs(b[index] - a[index])
                    );
                }

                flow_error = std::abs(b[1] - a[1]);

                for (int i = 0; i < n; ++i) {
                    const std::size_t index = 2*i + 2;

                    flow_error = std::max(
                        flow_error,
                        std::abs(b[index] - a[index])
                    );
                }

                const double inventory_error =
                    trajectory_linepack(trajectory, k, config)
                    - trajectory_linepack(reference, k, config);

                const double inventory_change_error =
                    inventory_error - initial_inventory_error;

                double conservation_defect = 0.0;

                if (k > 0) {
                    const auto current =
                        trajectory.state(k);

                    const auto previous =
                        trajectory.state(k - 1);

                    const auto reference_current =
                        reference.state(k);

                    const auto reference_previous =
                        reference.state(k - 1);

                    const double inlet_error_previous =
                        previous[1] - reference_previous[1];

                    const double inlet_error_current =
                        current[1] - reference_current[1];

                    const double inlet_error_weighted =
                        0.35 * inlet_error_previous
                        + 0.65 * inlet_error_current;

                    conservation_defect =
                        (inventory_error
                         - previous_inventory_error)
                        - dt * inlet_error_weighted;
                }

                file << m << ','
                     << k * dt << ','
                     << pressure_error << ','
                     << flow_error << ','
                     << amplitude << ','
                     << inventory_error << ','
                     << inventory_change_error << ','
                     << conservation_defect << '\n';

                previous_inventory_error = inventory_error;
            }
        }

        std::cout
            << "M2.4 weak-mode decay completed\n"
            << "modes_tested=" << modes_to_test << '\n'
            << "duration_s=" << steps * dt << '\n'
            << "output=" << output.string() << '\n';

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
