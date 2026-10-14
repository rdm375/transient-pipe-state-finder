#include "state_finder/joint_tangent_linear.hpp"
#include "state_finder/steady_state_initializer.hpp"

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
        constexpr int initial_dimension = 2*n + 1;

        constexpr double length = 50000.0;
        constexpr double dt = 10.0;
        constexpr double diameter = 0.6;
        constexpr double pressure = 8.0e6;
        constexpr double flow = 100.0;

        const double pi = std::acos(-1.0);
        const double area = pi*diameter*diameter/4.0;

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

        const auto directory =
            std::filesystem::path(
                "benchmarks/results-m25");

        std::filesystem::create_directories(directory);

        std::ofstream out(
            directory/"operational_sensitivity.csv");

        if (!out)
            throw std::runtime_error(
                "Cannot create sensitivity CSV");

        out << std::setprecision(17);

        out << "time_s,column,quantity,index,sensitivity\n";

        const auto dimension =
            joint_variable_dimension(n, steps);

        const std::size_t width =
            static_cast<std::size_t>(2*n+2);

        double max_backward_error = 0.0;

        for (std::size_t column=0;
             column<dimension;
             ++column)
        {
            std::vector<double> direction(
                dimension, 0.0);

            direction[column] = 1.0;

            const auto tangent =
                evaluate_joint_tangent_linear(
                    config, reference, direction);

            max_backward_error = std::max(
                max_backward_error,
                tangent.max_linear_backward_error);

            if (tangent.states.size() !=
                static_cast<std::size_t>(steps+1)*width)
            {
                throw std::runtime_error(
                    "Incomplete tangent history");
            }

            for (int k=0; k<=steps; ++k) {
                const auto state = tangent.state(k,n);
                const double time = k*dt;

                std::vector<double> nodal(n+1);

                nodal[0] = state[0];

                for (int i=0; i<n; ++i)
                    nodal[i+1] = state[2*i+3];

                // Pressure sensitivities: 21 nodes.
                for (int i=0; i<=n; ++i) {
                    out << time << ','
                        << column << ','
                        << "pressure" << ','
                        << i << ','
                        << nodal[i] << '\n';
                }

                // Flow sensitivities:
                // 0 = inlet
                // 1..n = internal faces
                // n+1 = prescribed outlet.
                out << time << ','
                    << column << ','
                    << "flow,0,"
                    << state[1] << '\n';

                for (int i=0; i<n; ++i) {
                    out << time << ','
                        << column << ','
                        << "flow,"
                        << i+1 << ','
                        << state[2*i+2] << '\n';
                }

                // Prescribed outlet flow is adjustable
                // in the joint parameterization.
                out << time << ','
                    << column << ','
                    << "flow,"
                    << n+1 << ','
                    << tangent.response.outlet_flow[k]
                    << '\n';

                // Constant-Z isothermal linepack is linear
                // in nodal pressure. Applying the same
                // inventory functional to tangent pressures
                // therefore gives its exact derivative.
                const double inventory_sensitivity =
                    pipe_sim::linepack_cz(
                        n, nodal, config.parameters);

                out << time << ','
                    << column << ','
                    << "inventory,0,"
                    << inventory_sensitivity << '\n';
            }
        }

        if (!out)
            throw std::runtime_error(
                "Sensitivity CSV write failed");

        std::cout
            << "M2.5-B1 joint operational sensitivity PASS\n"
            << "joint_dimension="
            << dimension << '\n'
            << "time_samples=" << steps+1 << '\n'
            << "pressure_outputs=" << n+1 << '\n'
            << "flow_outputs=" << n+2 << '\n'
            << "inventory_outputs=1\n"
            << "max_linear_backward_error="
            << max_backward_error << '\n'
            << "output="
            << (directory/"operational_sensitivity.csv")
            << '\n';

        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "M2.5-B1 joint operational sensitivity FAIL: "
            << e.what() << '\n';
        return 1;
    }
}
