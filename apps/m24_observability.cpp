#include "state_finder/joint_tangent_linear.hpp"
#include "state_finder/joint_trajectory.hpp"
#include "state_finder/steady_state_initializer.hpp"
#include "state_finder/four_channel_objective.hpp"
#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace state_finder;

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

        // Reproduce the M2.3-A reference experiment.
        JointInverseVariables truth;

        truth.initial = initialize_steady_state(config, flow);

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

        const auto trajectory =
            record_joint_trajectory(config, truth);

        BoundaryObservationRecord observations;

        observations.measured = trajectory.predicted;
        observations.include_initial = true;

        // This is the same scaled coordinate system
        // used by the joint inverse optimizer.
        JointVariableScales scales;

        const std::size_t columns =
            joint_variable_dimension(n, steps);

        const auto reference_residuals =
            evaluate_four_channel_objective(
                observations,
                FourChannelPrediction{trajectory.predicted}
            ).residuals;

        const std::size_t rows =
            reference_residuals.size();

        if (columns != 163 || rows != 244) {
            throw std::runtime_error(
                "Unexpected joint Jacobian dimensions"
            );
        }

        for (double r : reference_residuals) {
            if (std::abs(r) > 1.0e-10) {
                throw std::runtime_error(
                    "Reference measurement residual is nonzero"
                );
            }
        }

        std::vector<double> matrix(rows * columns, 0.0);
        std::vector<double> direction(columns, 0.0);

        double worst_backward_error = 0.0;

        for (std::size_t j = 0; j < columns; ++j) {
            std::fill(
                direction.begin(),
                direction.end(),
                0.0
            );

            direction[j] = 1.0;

            const auto tangent =
                evaluate_joint_tangent_linear(
                    config,
                    trajectory,
                    direction,
                    scales
                );

            const auto response =
                normalized_joint_tangent_residuals(
                    tangent,
                    observations
                );

            if (response.size() != rows) {
                throw std::runtime_error(
                    "Tangent response dimension mismatch"
                );
            }

            for (std::size_t i = 0; i < rows; ++i) {
                if (!std::isfinite(response[i])) {
                    throw std::runtime_error(
                        "Nonfinite Jacobian entry"
                    );
                }

                matrix[i * columns + j] = response[i];
            }

            worst_backward_error = std::max(
                worst_backward_error,
                tangent.max_linear_backward_error
            );
        }

        const std::filesystem::path output =
            "benchmarks/results-m24";

        std::filesystem::create_directories(output);

        const auto path =
            output / "joint_measurement_jacobian.csv";

        std::ofstream file(path);

        if (!file) {
            throw std::runtime_error(
                "Cannot open Jacobian output file"
            );
        }

        file << std::setprecision(17);

        for (std::size_t j = 0; j < columns; ++j) {
            if (j) file << ',';
            file << "x" << j;
        }

        file << '\n';

        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < columns; ++j) {
                if (j) file << ',';
                file << matrix[i * columns + j];
            }

            file << '\n';
        }

        if (!file) {
            throw std::runtime_error(
                "Failed writing Jacobian"
            );
        }

        std::cout << std::setprecision(12)
                  << "M2.4 joint observability\n"
                  << "rows=" << rows << '\n'
                  << "columns=" << columns << '\n'
                  << "max_reference_residual=0\n"
                  << "max_linear_backward_error="
                  << worst_backward_error << '\n'
                  << "output=" << path.string() << '\n';

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
