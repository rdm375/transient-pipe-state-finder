#include "state_finder/joint_tangent_linear.hpp"
#include "state_finder/joint_scaled_problem.hpp"
#include "state_finder/steady_state_initializer.hpp"

#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace state_finder;

namespace {

constexpr int n = 20;
constexpr int steps = 60;
constexpr int nx = 2*n+1;

using Key = std::pair<double, std::string>;
using Direction = std::array<double, nx>;

struct DirectionRecord {
    Direction values{};
    std::array<bool, nx> present{};
    int worst_index = -1;
};

std::map<Key, DirectionRecord> read_directions(
    const std::string& filename)
{
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error(
            "Cannot open adversarial directions");

    std::string line;
    std::getline(file, line); // Header.

    std::map<Key, DirectionRecord> result;

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::stringstream stream(line);
        std::string field;
        std::vector<std::string> columns;

        while (std::getline(stream, field, ','))
            columns.push_back(field);

        if (columns.size() != 6)
            throw std::runtime_error(
                "Invalid adversarial CSV row");

        const double epsilon = std::stod(columns[0]);
        const std::string quantity = columns[1];
        const double time = std::stod(columns[2]);
        const int worst_index = std::stoi(columns[3]);
        const int coordinate = std::stoi(columns[4]);
        const double value = std::stod(columns[5]);

        if (std::abs(epsilon-1.0) > 1e-12 ||
            std::abs(time-600.0) > 1e-9)
            continue;

        if (quantity != "pressure" &&
            quantity != "flow" &&
            quantity != "inventory")
            continue;

        if (coordinate < 0 || coordinate >= nx ||
            !std::isfinite(value))
            throw std::runtime_error(
                "Invalid adversarial coordinate");

        auto& record = result[{epsilon, quantity}];

        if (record.present[coordinate])
            throw std::runtime_error(
                "Duplicate adversarial coordinate");

        if (record.worst_index >= 0 &&
            record.worst_index != worst_index)
            throw std::runtime_error(
                "Inconsistent worst output index");

        record.values[coordinate] = value;
        record.present[coordinate] = true;
        record.worst_index = worst_index;
    }

    if (result.size() != 3)
        throw std::runtime_error(
            "Expected three adversarial directions");

    for (const auto& [key, record] : result) {
        if (!std::all_of(
                record.present.begin(),
                record.present.end(),
                [](bool x) { return x; }))
            throw std::runtime_error(
                "Incomplete adversarial direction");
    }

    return result;
}

std::vector<double> nodal_pressures(
    std::span<const double> state)
{
    std::vector<double> result(n+1);

    result[0] = state[0];

    for (int i=0; i<n; ++i)
        result[i+1] = state[2*i+3];

    return result;
}

double linepack(
    std::span<const double> state,
    const ForwardConfiguration& config)
{
    const auto pressure = nodal_pressures(state);

    return pipe_sim::linepack_cz(
        n, pressure, config.parameters);
}

double measurement_norm(
    const JointTrajectory& reference,
    const JointTrajectory& perturbed,
    double& maximum_component)
{
    // These scales must match the observation normalization
    // used to construct joint_measurement_jacobian.csv.
    //
    // Verify against the M2.4 observability exporter
    // before interpreting this as ||J_x v||.
    constexpr double inlet_pressure_scale = 1.0e4;
    constexpr double inlet_flow_scale = 1.0;
    constexpr double outlet_pressure_scale = 1.0e4;
    constexpr double outlet_flow_scale = 1.0;

    double sum = 0.0;
    maximum_component = 0.0;

    for (int k=0; k<=steps; ++k) {
        const auto& a = reference.predicted;
        const auto& b = perturbed.predicted;

        const std::array<double, 4> residuals{
            (b.inlet_pressure[k]-a.inlet_pressure[k])
                /inlet_pressure_scale,
            (b.inlet_flow[k]-a.inlet_flow[k])
                /inlet_flow_scale,
            (b.outlet_pressure[k]-a.outlet_pressure[k])
                /outlet_pressure_scale,
            (b.outlet_flow[k]-a.outlet_flow[k])
                /outlet_flow_scale
        };

        for (double r : residuals) {
            sum += r*r;
            maximum_component = std::max(
                maximum_component, std::abs(r));
        }
    }

    return std::sqrt(sum);
}

} // namespace

int main()
{
    try {
        constexpr double length = 50000.0;
        constexpr double dt = 10.0;
        constexpr double diameter = 0.6;
        constexpr double pressure = 8.0e6;
        constexpr double flow = 100.0;

        const double pi = std::acos(-1.0);
        const double area =
            pi*diameter*diameter/4.0;

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                length/n, dt, 0.65,
                diameter, area, 288.15,
                0.9, 500.0, 1.1e-5, 4.5e-5
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
            steps+1, pressure);

        truth.boundary.outlet_flow.assign(
            steps+1, flow);

        for (int k=0; k<=steps; ++k) {
            const double t = k*dt;

            const double fraction = std::clamp(
                (t-120.0)/120.0, 0.0, 1.0);

            const double smooth =
                fraction*fraction*(3.0-2.0*fraction);

            truth.boundary.outlet_flow[k] =
                flow+10.0*smooth;
        }

        const auto reference =
            record_joint_trajectory(config, truth);

        const auto directions = read_directions(
            "benchmarks/results-m24/"
            "adversarial_initial_states.csv");

        const std::filesystem::path directory(
            "benchmarks/results-m24");

        std::filesystem::create_directories(directory);

        std::ofstream output(
            directory/"adversarial_nonlinear_trajectories.csv");

        if (!output)
            throw std::runtime_error(
                "Cannot create nonlinear validation CSV");

        output << std::setprecision(17);

        output
            << "quantity,amplitude,time_s,"
            << "pressure_linf_nonlinear_Pa,"
            << "pressure_linf_linear_Pa,"
            << "flow_linf_nonlinear_kg_s,"
            << "flow_linf_linear_kg_s,"
            << "inventory_nonlinear_kg,"
            << "inventory_linear_kg,"
            << "pressure_linearization_error_linf_Pa,"
            << "flow_linearization_error_linf_kg_s,"
            << "inventory_linearization_error_kg,"
            << "measurement_norm_nonlinear,"
            << "measurement_norm_linear,"
            << "max_normalized_measurement_component,"
            << "conservation_defect_kg,"
            << "minimum_pressure_Pa\n";

        const std::array<double,4> amplitudes{
            0.01, 0.1, 0.5, 1.0
        };

        const auto dimension =
            joint_variable_dimension(n,steps);

        JointVariableScales scales;

        for (const auto& [key, record] : directions) {
            const auto& quantity = key.second;

            std::vector<double> direction(
                dimension, 0.0);

            for (int j=0; j<nx; ++j)
                direction[j] = record.values[j];

            const auto tangent =
                evaluate_joint_tangent_linear(
                    config, reference, direction, scales);

            for (double amplitude : amplitudes) {
                auto scaled_direction = direction;

                for (double& value : scaled_direction)
                    value *= amplitude;

                auto perturbed = unpack_joint_variables(
                    scaled_direction, truth, scales);

                // Keep the complete boundary history fixed.
                perturbed.boundary = truth.boundary;

                const auto nonlinear =
                    record_joint_trajectory(
                        config, perturbed);

                double max_component = 0.0;

                const double nonlinear_measurement =
                    measurement_norm(
                        reference, nonlinear, max_component);

                // Compute the tangent measurement norm
                // using the same normalization.
                double tangent_measurement_squared = 0.0;

                for (int k=0; k<=steps; ++k) {
                    const auto& y = tangent.response;

                    const std::array<double,4> r{
                        y.inlet_pressure[k]/1.0e4,
                        y.inlet_flow[k]/1.0,
                        y.outlet_pressure[k]/1.0e4,
                        y.outlet_flow[k]/1.0
                    };

                    for (double value : r)
                        tangent_measurement_squared +=
                            value*value;
                }

                const double linear_measurement =
                    amplitude*std::sqrt(
                        tangent_measurement_squared);

                double previous_inventory_error = 0.0;

                for (int k=0; k<=steps; ++k) {
                    const auto a = reference.state(k);
                    const auto b = nonlinear.state(k);
                    const auto d = tangent.state(k,n);

                    double pressure_nonlinear = 0.0;
                    double pressure_linear = 0.0;
                    double flow_nonlinear = 0.0;
                    double flow_linear = 0.0;

                    double pressure_linearization_error = 0.0;
                    double flow_linearization_error = 0.0;

                    double minimum_pressure =
                        std::numeric_limits<double>::infinity();

                    for (int i=0; i<=n; ++i) {
                        const int j =
                            i == 0 ? 0 : 2*i+1;

                        const double nonlinear_error =
                            b[j]-a[j];

                        const double linear_error =
                            amplitude*d[j];

                        pressure_nonlinear = std::max(
                            pressure_nonlinear,
                            std::abs(nonlinear_error));

                        pressure_linear = std::max(
                            pressure_linear,
                            std::abs(linear_error));

                        pressure_linearization_error = std::max(
                            pressure_linearization_error,
                            std::abs(
                                nonlinear_error-linear_error));

                        minimum_pressure = std::min(
                            minimum_pressure,b[j]);
                    }

                    for (int i=0; i<=n; ++i) {
                        const int j =
                            i == 0 ? 1 : 2*i;

                        const double nonlinear_error =
                            b[j]-a[j];

                        const double linear_error =
                            amplitude*d[j];

                        flow_nonlinear = std::max(
                            flow_nonlinear,
                            std::abs(nonlinear_error));

                        flow_linear = std::max(
                            flow_linear,
                            std::abs(linear_error));

                        flow_linearization_error = std::max(
                            flow_linearization_error,
                            std::abs(
                                nonlinear_error-linear_error));
                    }

                    const double inventory_nonlinear =
                        linepack(b,config)-
                        linepack(a,config);

                    const double inventory_linear =
                        amplitude*linepack(d,config);

                    double conservation_defect = 0.0;

                    if (k > 0) {
                        const auto current =
                            nonlinear.state(k);
                        const auto previous =
                            nonlinear.state(k-1);

                        const auto reference_current =
                            reference.state(k);
                        const auto reference_previous =
                            reference.state(k-1);

                        const double inlet_previous =
                            previous[1]-
                            reference_previous[1];

                        const double inlet_current =
                            current[1]-
                            reference_current[1];

                        const double weighted_inlet =
                            (1.0-config.parameters.theta)
                                *inlet_previous
                            +config.parameters.theta
                                *inlet_current;

                        conservation_defect =
                            (inventory_nonlinear-
                             previous_inventory_error)
                            -dt*weighted_inlet;
                    }

                    output
                        << quantity << ','
                        << amplitude << ','
                        << k*dt << ','
                        << pressure_nonlinear << ','
                        << pressure_linear << ','
                        << flow_nonlinear << ','
                        << flow_linear << ','
                        << inventory_nonlinear << ','
                        << inventory_linear << ','
                        << pressure_linearization_error << ','
                        << flow_linearization_error << ','
                        << std::abs(
                            inventory_nonlinear-
                            inventory_linear) << ','
                        << nonlinear_measurement << ','
                        << linear_measurement << ','
                        << max_component << ','
                        << conservation_defect << ','
                        << minimum_pressure << '\n';

                    previous_inventory_error =
                        inventory_nonlinear;
                }

                std::cout
                    << quantity
                    << " amplitude=" << amplitude
                    << " nonlinear_measurement="
                    << nonlinear_measurement
                    << " linear_measurement="
                    << linear_measurement
                    << '\n';
            }
        }

        if (!output)
            throw std::runtime_error(
                "Nonlinear validation CSV write failed");

        std::cout
            << "M2.4-F2C nonlinear validation completed\n"
            << "experiments=12\n"
            << "output="
            << (directory/
                "adversarial_nonlinear_trajectories.csv")
            << '\n';

        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "M2.4-F2C FAIL: "
            << e.what() << '\n';

        return 1;
    }
}
