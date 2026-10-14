#include "state_finder/joint_tangent_linear.hpp"
#include "state_finder/steady_state_initializer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace state_finder;

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

bool close(double a, double b, double tolerance)
{
    return std::abs(a-b) <=
        tolerance*std::max({
            1.0, std::abs(a), std::abs(b)
        });
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
        const double area =
            pi*diameter*diameter/4.0;

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

        const auto trajectory =
            record_joint_trajectory(config, truth);

        const auto dimension =
            joint_variable_dimension(n, steps);

        std::vector<double> direction(
            dimension, 0.0);

        // Exercise initial pressure, face flow,
        // inlet flow, and boundary perturbations.
        direction[0] = 0.25;
        direction[n] = -0.15;
        direction[2*n] = 0.1;

        const std::size_t boundary_offset =
            static_cast<std::size_t>(2*n+1);

        direction[boundary_offset] = 0.2;
        direction[boundary_offset+1] = -0.1;

        direction[
            boundary_offset+(steps+1)
        ] = 0.3;

        const auto tangent =
            evaluate_joint_tangent_linear(
                config, trajectory, direction);

        const std::size_t width =
            static_cast<std::size_t>(2*n+2);

        require(
            tangent.states.size() ==
                static_cast<std::size_t>(steps+1)*width,
            "Incorrect tangent history dimension");

        require(
            tangent.final_state.size() == width,
            "Incorrect final tangent dimension");

        const auto initial = tangent.state(0, n);
        const auto final = tangent.state(steps, n);

        require(
            close(initial[0], 2000.0, 1e-12),
            "Incorrect initial inlet-pressure tangent");

        require(
            close(initial[1], 1.0, 1e-12),
            "Incorrect initial inlet-flow tangent");

        require(
            close(initial[2], -1.5, 1e-12),
            "Incorrect initial face-flow tangent");

        require(
            close(initial[3], 25000.0, 1e-12),
            "Incorrect initial pressure tangent");

        for (std::size_t i=0; i<width; ++i) {
            require(
                close(final[i],
                      tangent.final_state[i],
                      1e-12),
                "Final tangent history mismatch");
        }

        for (int k=0; k<=steps; ++k) {
            const auto state = tangent.state(k, n);

            require(
                close(
                    state[0],
                    tangent.response.inlet_pressure[k],
                    1e-8),
                "Inlet-pressure history mismatch");

            require(
                close(
                    state[1],
                    tangent.response.inlet_flow[k],
                    1e-12),
                "Inlet-flow history mismatch");

            require(
                close(
                    state[width-1],
                    tangent.response.outlet_pressure[k],
                    1e-12),
                "Outlet-pressure history mismatch");
        }

        std::cout
            << "M2.4-F1 tangent history PASS\n"
            << "steps=" << steps << '\n'
            << "state_width=" << width << '\n'
            << "stored_values="
            << tangent.states.size() << '\n'
            << "max_linear_backward_error="
            << tangent.max_linear_backward_error
            << '\n';

        return 0;

    } catch (const std::exception& e) {
        std::cerr
            << "M2.4-F1 tangent history FAIL: "
            << e.what() << '\n';

        return 1;
    }
}
