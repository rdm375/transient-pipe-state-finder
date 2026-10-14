#include "state_finder/joint_trajectory.hpp"
#include "state_finder/joint_scaled_problem.hpp"
#include "state_finder/steady_state_initializer.hpp"

#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace state_finder;

namespace {

constexpr int n = 20;
constexpr int steps = 60;
constexpr double dt = 10.0;

std::vector<double> pressures(std::span<const double> state)
{
    std::vector<double> result(n+1);

    result[0] = state[0];

    for (int i=1; i<=n; ++i)
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

double measurement_norm(
    const BoundaryMeasurements& reference,
    const BoundaryMeasurements& perturbed)
{
    double norm2 = 0.0;

    for (int k=0; k<=steps; ++k) {
        const double residuals[] = {
            (perturbed.inlet_pressure[k]
                -reference.inlet_pressure[k])/1.0e4,

            perturbed.inlet_flow[k]
                -reference.inlet_flow[k],

            (perturbed.outlet_pressure[k]
                -reference.outlet_pressure[k])/1.0e4,

            perturbed.outlet_flow[k]
                -reference.outlet_flow[k]
        };

        for (double value : residuals)
            norm2 += value*value;
    }

    return std::sqrt(norm2);
}

} // namespace

int main()
{
    try {
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

        for (int k=0; k<=steps; ++k) {
            const double t = k*dt;

            const double fraction = std::clamp(
                (t-120.0)/120.0,
                0.0,
                1.0);

            truth.boundary.outlet_flow[k] =
                flow+10.0*fraction*fraction*
                    (3.0-2.0*fraction);
        }

        const auto reference =
            record_joint_trajectory(config, truth);

        const auto dimension =
            joint_variable_dimension(n, steps);

        const JointVariableScales scales{};

        std::cout
            << "READY " << dimension << '\n'
            << std::flush;

        std::string line;

        while (std::getline(std::cin, line)) {
            if (line == "QUIT")
                break;

            try {
                const bool measurement_mode =
                    line.rfind("MEASURE,", 0) == 0;

                std::istringstream input(
                    measurement_mode
                        ? line.substr(8)
                        : line);

                std::vector<double> direction(
                    dimension, 0.0);

                for (std::size_t j=0;
                     j<dimension;
                     ++j)
                {
                    std::string field;

                    if (!std::getline(input,field,','))
                        throw std::runtime_error(
                            "Missing coordinate");

                    direction[j] = std::stod(field);

                    if (!std::isfinite(direction[j]))
                        throw std::runtime_error(
                            "Nonfinite coordinate");
                }

                std::string extra;

                if (std::getline(input,extra,','))
                    throw std::runtime_error(
                        "Extra coordinate");

                const auto perturbed =
                    unpack_joint_variables(
                        direction, truth, scales);

                const auto trajectory =
                    record_joint_trajectory(
                        config, perturbed);

                if (measurement_mode) {
                    std::cout
                        << std::setprecision(17)
                        << "MEASURE";

                    for (int k=0; k<=steps; ++k) {
                        std::cout << ','
                            << (trajectory.predicted.inlet_pressure[k]
                                -reference.predicted.inlet_pressure[k])
                                /1.0e4;

                        std::cout << ','
                            << (trajectory.predicted.inlet_flow[k]
                                -reference.predicted.inlet_flow[k]);

                        std::cout << ','
                            << (trajectory.predicted.outlet_pressure[k]
                                -reference.predicted.outlet_pressure[k])
                                /1.0e4;

                        std::cout << ','
                            << (trajectory.predicted.outlet_flow[k]
                                -reference.predicted.outlet_flow[k]);
                    }

                    std::cout << '\n' << std::flush;
                    continue;
                }

                const auto initial =
                    trajectory.state(0);

                const auto terminal =
                    trajectory.state(steps);

                const auto reference_terminal =
                    reference.state(steps);

                const double outlet_pressure =
                    terminal[2*n+1]
                    -reference_terminal[2*n+1];

                const double inlet_flow =
                    terminal[1]
                    -reference_terminal[1];

                const double linepack =
                    inventory(terminal,config)
                    -inventory(reference_terminal,config);

                double min_pressure =
                    std::numeric_limits<double>::infinity();

                double max_mass_defect = 0.0;

                for (int k=0; k<=steps; ++k) {
                    const auto state =
                        trajectory.state(k);

                    for (double p : pressures(state))
                        min_pressure =
                            std::min(min_pressure,p);

                    if (k == 0)
                        continue;

                    const double previous_inventory =
                        inventory(
                            trajectory.state(k-1),config);

                    const double current_inventory =
                        inventory(state,config);

                    const double inlet_previous =
                        trajectory.state(k-1)[1];

                    const double inlet_current =
                        state[1];

                    const double outlet_previous =
                        perturbed.boundary.outlet_flow[k-1];

                    const double outlet_current =
                        perturbed.boundary.outlet_flow[k];

                    const double theta =
                        config.parameters.theta;

                    const double inlet_weighted =
                        (1.0-theta)*inlet_previous
                        +theta*inlet_current;

                    const double outlet_weighted =
                        (1.0-theta)*outlet_previous
                        +theta*outlet_current;

                    const double defect =
                        (current_inventory
                            -previous_inventory)
                        -dt*(inlet_weighted
                            -outlet_weighted);

                    max_mass_defect = std::max(
                        max_mass_defect,
                        std::abs(defect));
                }

                const double compatibility =
                    std::abs(
                        initial[0]
                        -perturbed.boundary.inlet_pressure[0]);

                std::cout
                    << std::setprecision(17)
                    << "OK,"
                    << outlet_pressure << ','
                    << inlet_flow << ','
                    << linepack << ','
                    << measurement_norm(
                        reference.predicted,
                        trajectory.predicted) << ','
                    << min_pressure << ','
                    << max_mass_defect << ','
                    << compatibility
                    << '\n'
                    << std::flush;

            } catch (const std::exception& e) {
                std::cout
                    << "ERROR," << e.what()
                    << '\n' << std::flush;
            }
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "M25 nonlinear replay fatal: "
            << e.what() << '\n';

        return 1;
    }
}
