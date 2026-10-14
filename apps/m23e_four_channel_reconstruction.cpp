#include "state_finder/joint_gauss_newton_operator.hpp"
#include "state_finder/joint_regularization_preconditioner.hpp"
#include "state_finder/joint_scaled_problem.hpp"
#include "state_finder/steady_state_initializer.hpp"
#include "state_finder/joint_trajectory.hpp"
#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <array>
#include <chrono>
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

void check(bool ok, const std::string& message) {
    if (!ok)
        throw std::runtime_error(message);
}

std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;

    while (std::getline(ss, field, ','))
        fields.push_back(field);

    return fields;
}

std::size_t column(
    const std::vector<std::string>& header,
    const std::string& name)
{
    auto it = std::find(header.begin(), header.end(), name);

    check(it != header.end(), "Missing CSV column: " + name);

    return static_cast<std::size_t>(it - header.begin());
}

struct Data {
    BoundaryObservationRecord observations;
    std::vector<double> time;
    BoundaryMeasurements truth;
};

Data load(
    const std::string& filename,
    int steps,
    double dt)
{
    std::ifstream in(filename);

    check(bool(in), "Cannot open measurement CSV: " + filename);

    std::string line;

    check(bool(std::getline(in, line)), "Empty measurement CSV");

    auto header = split(line);

    const std::array<std::string, 9> names = {
        "time_s",
        "inlet_pressure_observed_Pa",
        "inlet_flow_observed_kg_s",
        "outlet_pressure_observed_Pa",
        "outlet_flow_observed_kg_s",
        "inlet_pressure_truth_Pa",
        "inlet_flow_truth_kg_s",
        "outlet_pressure_truth_Pa",
        "outlet_flow_truth_kg_s"
    };

    std::array<std::size_t, 9> indices{};

    for (std::size_t j = 0; j < names.size(); ++j)
        indices[j] = column(header, names[j]);

    Data data;

    auto& m = data.observations.measured;
    auto& t = data.truth;

    while (std::getline(in, line)) {
        if (line.empty())
            continue;

        const auto fields = split(line);

        std::array<double, 9> values{};

        for (std::size_t j = 0; j < values.size(); ++j) {
            check(
                indices[j] < fields.size(),
                "Short measurement CSV row"
            );

            std::size_t consumed = 0;

            values[j] = std::stod(
                fields[indices[j]], &consumed
            );

            check(
                consumed == fields[indices[j]].size() &&
                std::isfinite(values[j]),
                "Invalid numeric measurement CSV field"
            );
        }

        data.time.push_back(values[0]);

        m.inlet_pressure.push_back(values[1]);
        m.inlet_flow.push_back(values[2]);
        m.outlet_pressure.push_back(values[3]);
        m.outlet_flow.push_back(values[4]);

        t.inlet_pressure.push_back(values[5]);
        t.inlet_flow.push_back(values[6]);
        t.outlet_pressure.push_back(values[7]);
        t.outlet_flow.push_back(values[8]);
    }

    const auto count = static_cast<std::size_t>(steps + 1);

    check(
        data.time.size() == count,
        "Expected exactly 61 measurement rows"
    );

    for (std::size_t k = 0; k < count; ++k) {
        check(
            std::abs(data.time[k] - dt*k) < 1.e-8,
            "Measurement timestamps mismatch"
        );
    }

    data.observations.include_initial = true;

    data.observations.scales.inlet_pressure = 1000.0;
    data.observations.scales.inlet_flow = 0.1;
    data.observations.scales.outlet_pressure = 1000.0;
    data.observations.scales.outlet_flow = 0.1;

    validate_boundary_observations(
        data.observations, count
    );

    return data;
}

void boundary_csv(
    const std::string& filename,
    const Data& data,
    const BoundaryMeasurements& predicted)
{
    std::ofstream out(filename);

    check(bool(out), "Cannot write " + filename);

    out << std::setprecision(17);

    out <<
        "time_s,"
        "inlet_pressure_observed_Pa,"
        "inlet_pressure_reconstructed_Pa,"
        "inlet_pressure_truth_Pa,"
        "inlet_flow_observed_kg_s,"
        "inlet_flow_reconstructed_kg_s,"
        "inlet_flow_truth_kg_s,"
        "outlet_pressure_observed_Pa,"
        "outlet_pressure_reconstructed_Pa,"
        "outlet_pressure_truth_Pa,"
        "outlet_flow_observed_kg_s,"
        "outlet_flow_reconstructed_kg_s,"
        "outlet_flow_truth_kg_s\n";

    const auto& m = data.observations.measured;
    const auto& t = data.truth;

    for (std::size_t k = 0; k < data.time.size(); ++k) {
        out
            << data.time[k] << ','
            << m.inlet_pressure[k] << ','
            << predicted.inlet_pressure[k] << ','
            << t.inlet_pressure[k] << ','
            << m.inlet_flow[k] << ','
            << predicted.inlet_flow[k] << ','
            << t.inlet_flow[k] << ','
            << m.outlet_pressure[k] << ','
            << predicted.outlet_pressure[k] << ','
            << t.outlet_pressure[k] << ','
            << m.outlet_flow[k] << ','
            << predicted.outlet_flow[k] << ','
            << t.outlet_flow[k] << '\n';
    }
}

double rms_error(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    check(
        a.size() == b.size() && !a.empty(),
        "RMS dimension mismatch"
    );

    double sum = 0.0;

    for (std::size_t k = 0; k < a.size(); ++k)
        sum += (a[k] - b[k]) * (a[k] - b[k]);

    return std::sqrt(sum / a.size());
}

} // namespace

int main(int argc, char** argv)
{
    try {
        check(
            argc == 1 || argc == 3,
            "Usage: executable [measurements.csv output_directory]"
        );

        const std::string input =
            argc == 3
            ? argv[1]
            : "benchmarks/results-m23d/measurements.csv";

        const std::string output =
            argc == 3
            ? argv[2]
            : "benchmarks/results-m23e";

        constexpr int n = 20;
        constexpr int steps = 60;

        constexpr double length = 50000.0;
        constexpr double dt = 10.0;
        constexpr double diameter = 0.6;
        constexpr double pressure = 8.e6;
        constexpr double flow = 100.0;

        const double pi = std::acos(-1.0);
        const double area = pi*diameter*diameter/4.0;

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                length/n,
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
                1.e-9, 1.e-12, 30, false
            }
        };

        const Data data = load(input, steps, dt);
        const auto& obs = data.observations;

        JointInverseVariables baseline;

        // Observed values initialize the adjustable histories.
        // They are not held fixed during optimization.

        baseline.initial = initialize_steady_state(
            config,
            obs.measured.inlet_flow.front()
        );

        baseline.boundary.inlet_pressure =
            obs.measured.inlet_pressure;

        baseline.boundary.outlet_flow =
            obs.measured.outlet_flow;

        baseline.initial.pressure[0] =
            baseline.boundary.inlet_pressure[0];

        // Adjust the initial pressure profile to the
        // observed inlet-pressure starting point.

        const double shift =
            baseline.boundary.inlet_pressure[0] - pressure;

        for (std::size_t i = 1;
             i < baseline.initial.pressure.size();
             ++i)
        {
            baseline.initial.pressure[i] += shift;
        }


        // M2.3-G: Deliberately perturb the initial internal state.
        //
        // Boundary histories remain exact initially but adjustable.
        // This forces the optimizer to recover the initial state
        // rather than starting at the reference solution.

        constexpr double pressure_perturbation = 50000.0;
        constexpr double flow_perturbation = 2.0;

        for (int i = 1; i < n; ++i) {
            const double xi =
                static_cast<double>(i) / n;

            baseline.initial.pressure[i] +=
                pressure_perturbation *
                std::sin(pi * xi);
        }

        for (int i = 0; i < n; ++i) {
            const double xi =
                (static_cast<double>(i) + 0.5) / n;

            baseline.initial.flow[i] +=
                flow_perturbation *
                std::sin(2.0 * pi * xi);
        }

        std::cout
            << "M2.3-G initial-state perturbation:"
            << " pressure=" << pressure_perturbation
            << " Pa, flow=" << flow_perturbation
            << " kg/s\n";

        JointVariableScales scales;
        JointRegularization regularization;

        JointScaledProblem problem(
            config,
            obs,
            baseline,
            scales,
            regularization
        );

        const std::vector<double> initial(
            joint_variable_dimension(n, steps),
            0.0
        );

        check(
            initial.size() == 163,
            "Unexpected optimization dimension"
        );

        const auto before = problem.evaluate(initial);

        const ObjectiveFunction objective =
            [&problem](
                const std::vector<double>& x,
                bool gradient)
            {
                return problem.evaluate_for_optimizer(
                    x, gradient
                );
            };

        const auto factory =
            make_joint_gauss_newton_factory(
                config,
                obs,
                baseline,
                scales,
                regularization
            );

        const auto preconditioner =
            make_joint_regularization_preconditioner_factory(
                n,
                steps,
                regularization
            );

        GaussNewtonOptions options;

        options.max_iterations = 500;
        options.max_cg_iterations = 300;
        options.gradient_tolerance = 1.e-7;
        options.initial_damping = 1.e-3;

        GaussNewtonDiagnostics diagnostics;

        const auto start =
            std::chrono::steady_clock::now();

        const auto result = minimize_gauss_newton(
            objective,
            initial,
            factory,
            options,
            &diagnostics,
            preconditioner
        );

        const double elapsed =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start
            ).count();

        check(
            result.variables.size() == initial.size(),
            "Reconstruction dimension mismatch"
        );

        const auto after =
            problem.evaluate(result.variables);

        check(
            after.total <= before.total + 1.e-8,
            "Reconstruction increased objective"
        );

        const auto variables =
            unpack_joint_variables(
                result.variables,
                baseline,
                scales
            );

        const auto trajectory =
            record_joint_trajectory(config, variables);

        const auto measurement =
            evaluate_four_channel_objective(
                obs,
                FourChannelPrediction{
                    trajectory.predicted
                }
            );

        check(
            measurement.residuals.size() == 244,
            "Four-channel residual count mismatch"
        );

        std::filesystem::create_directories(output);

        boundary_csv(
            output + "/reconstructed_boundary.csv",
            data,
            trajectory.predicted
        );

        // Reconstructed initial state.

        std::ofstream initial_csv(
            output + "/reconstructed_initial_state.csv"
        );

        check(
            bool(initial_csv),
            "Cannot write initial-state CSV"
        );

        initial_csv << std::setprecision(17)
                    << "node,x_m,pressure_Pa,"
                       "face_flow_kg_s\n";

        for (int i = 0; i <= n; ++i) {
            initial_csv
                << i << ','
                << i*length/n << ','
                << (
                    i == 0
                    ? variables.boundary.inlet_pressure[0]
                    : variables.initial.pressure[i]
                )
                << ',';

            if (i < n)
                initial_csv << variables.initial.flow[i];

            initial_csv << '\n';
        }

        initial_csv
            << "# initial_inlet_flow_kg_s,"
            << variables.initial.inlet_flow
            << '\n';

        // Discrete inventory and conservation.

        std::vector<double> nodal(n+1);
        std::vector<double> linepack(steps+1);

        for (int k = 0; k <= steps; ++k) {
            const auto state = trajectory.state(k);

            nodal[0] = state[0];

            for (int i = 0; i < n; ++i)
                nodal[i+1] = state[2*i+3];

            linepack[k] = pipe_sim::linepack_cz(
                n,
                nodal,
                config.parameters
            );
        }

        std::ofstream inventory(
            output + "/reconstructed_inventory.csv"
        );

        check(
            bool(inventory),
            "Cannot write inventory CSV"
        );

        inventory
            << std::setprecision(17)
            << "time_s,linepack_kg,"
               "physical_imbalance_kg_s,"
               "inventory_rate_kg_s,"
               "mass_defect_kg_s\n";

        double max_defect = 0.0;
        double cumulative_defect = 0.0;

        for (int k = 0; k <= steps; ++k) {
            const auto& b = trajectory.predicted;

            const double imbalance =
                b.inlet_flow[k] - b.outlet_flow[k];

            double rate = 0.0;
            double defect = 0.0;

            if (k > 0) {
                rate =
                    (linepack[k] - linepack[k-1])/dt;

                const double old_imbalance =
                    b.inlet_flow[k-1] -
                    b.outlet_flow[k-1];

                defect =
                    rate -
                    (
                        config.parameters.theta*imbalance +
                        (1.0-config.parameters.theta)*
                        old_imbalance
                    );

                max_defect = std::max(
                    max_defect,
                    std::abs(defect)
                );

                cumulative_defect += dt*defect;
            }

            inventory
                << k*dt << ','
                << linepack[k] << ','
                << imbalance << ','
                << rate << ','
                << defect << '\n';
        }

        // Machine-readable summary.

        std::ofstream summary(
            output + "/reconstruction_summary.json"
        );

        check(
            bool(summary),
            "Cannot write summary JSON"
        );

        summary
            << std::setprecision(17)
            << "{\n"
            << "  \"measurement_count\": 244,\n"
            << "  \"variable_count\": 163,\n"
            << "  \"initial_objective\": "
            << before.total << ",\n"
            << "  \"final_objective\": "
            << after.total << ",\n"
            << "  \"measurement_objective\": "
            << measurement.total << ",\n"
            << "  \"inlet_pressure_objective\": "
            << measurement.inlet_pressure << ",\n"
            << "  \"inlet_flow_objective\": "
            << measurement.inlet_flow << ",\n"
            << "  \"outlet_pressure_objective\": "
            << measurement.outlet_pressure << ",\n"
            << "  \"outlet_flow_objective\": "
            << measurement.outlet_flow << ",\n"
            << "  \"inlet_pressure_truth_rmse_Pa\": "
            << rms_error(
                trajectory.predicted.inlet_pressure,
                data.truth.inlet_pressure
            ) << ",\n"
            << "  \"inlet_flow_truth_rmse_kg_s\": "
            << rms_error(
                trajectory.predicted.inlet_flow,
                data.truth.inlet_flow
            ) << ",\n"
            << "  \"outlet_pressure_truth_rmse_Pa\": "
            << rms_error(
                trajectory.predicted.outlet_pressure,
                data.truth.outlet_pressure
            ) << ",\n"
            << "  \"outlet_flow_truth_rmse_kg_s\": "
            << rms_error(
                trajectory.predicted.outlet_flow,
                data.truth.outlet_flow
            ) << ",\n"
            << "  \"max_mass_defect_kg_s\": "
            << max_defect << ",\n"
            << "  \"cumulative_mass_defect_kg\": "
            << cumulative_defect << ",\n"
            << "  \"iterations\": "
            << result.iterations << ",\n"
            << "  \"gradient_norm\": "
            << result.gradient_norm << ",\n"
            << "  \"converged\": "
            << (result.converged ? "true" : "false")
            << ",\n"
            << "  \"seconds\": "
            << elapsed << ",\n"
            << "  \"pcg_iterations\": "
            << diagnostics.pcg_iterations << ",\n"
            << "  \"operator_calls\": "
            << diagnostics.operator_calls << "\n"
            << "}\n";

        std::cout
            << std::setprecision(12)
            << "M2.3-E four-channel reconstruction\n"
            << "measurements="
            << measurement.residuals.size()
            << " variables="
            << result.variables.size() << '\n'
            << "initial_objective="
            << before.total
            << " final_objective="
            << after.total << '\n'
            << "channel_objectives="
            << measurement.inlet_pressure << ','
            << measurement.inlet_flow << ','
            << measurement.outlet_pressure << ','
            << measurement.outlet_flow << '\n'
            << "iterations="
            << result.iterations
            << " converged="
            << result.converged
            << " reason="
            << result.reason << '\n'
            << "max_mass_defect_kg_s="
            << max_defect << '\n'
            << "results="
            << output << '\n';

        check(
            max_defect < 1.e-5,
            "Discrete mass conservation check failed"
        );

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL M2.3-E: "
            << e.what() << '\n';

        return 1;
    }
}
