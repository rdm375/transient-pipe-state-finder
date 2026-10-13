#include "state_finder/joint_trajectory.hpp"

#include "pipe_sim/transient_step.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace state_finder {

std::span<const double> JointTrajectory::state(int k) const
{
    if (n < 2 || k < 0 || k > steps ||
        states.size() !=
            static_cast<std::size_t>(steps+1)*(2*n+2))
    {
        throw std::out_of_range(
            "Invalid joint trajectory state index");
    }

    const auto width =
        static_cast<std::size_t>(2*n+2);

    return {
        states.data() +
            static_cast<std::size_t>(k)*width,
        width
    };
}

JointTrajectory record_joint_trajectory(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables)
{
    const int n = configuration.n;
    const int steps = configuration.steps;

    if (n < 2 || steps < 0)
        throw std::invalid_argument(
            "Invalid joint trajectory dimensions");

    const auto count =
        static_cast<std::size_t>(steps + 1);

    const auto width =
        static_cast<std::size_t>(2*n + 2);

    if (variables.initial.pressure.size() !=
            static_cast<std::size_t>(n+1) ||
        variables.initial.flow.size() !=
            static_cast<std::size_t>(n) ||
        variables.boundary.inlet_pressure.size() != count ||
        variables.boundary.outlet_flow.size() != count)
        throw std::invalid_argument(
            "Joint trajectory dimension mismatch");

    JointTrajectory result;
    result.n = n;
    result.steps = steps;

    result.states.resize(count*width);

    auto& y = result.predicted;

    y.inlet_pressure = variables.boundary.inlet_pressure;
    y.outlet_flow = variables.boundary.outlet_flow;
    y.inlet_flow.resize(count);
    y.outlet_pressure.resize(count);

    auto pressure = variables.initial.pressure;
    auto flow = variables.initial.flow;

    pressure[0] = variables.boundary.inlet_pressure[0];

    double inlet_flow = variables.initial.inlet_flow;

    // Initial interleaved state.
    result.states[0] = pressure[0];
    result.states[1] = inlet_flow;

    for (int i = 0; i < n; ++i) {
        result.states[2*i+2] = flow[i];
        result.states[2*i+3] = pressure[i+1];
    }

    y.inlet_flow[0] = inlet_flow;
    y.outlet_pressure[0] = pressure[n];

    pipe_sim::NewtonWorkspace workspace(n);

    std::vector<double> state(width);
    std::vector<double> next_pressure(n+1);
    std::vector<double> next_flow(n);

    for (int k = 1; k <= steps; ++k) {
        const pipe_sim::TransientBoundary boundary{
            inlet_flow,
            variables.boundary.outlet_flow[k-1],
            variables.boundary.outlet_flow[k],
            variables.boundary.inlet_pressure[k]
        };

        const auto step = pipe_sim::transient_step(
            n,
            pressure,
            flow,
            boundary,
            configuration.parameters,
            configuration.newton,
            state,
            workspace);

        if (step.info != 0)
            throw std::runtime_error(
                "Joint adjoint forward trajectory failed");

        next_pressure[0] = state[0];

        for (int i = 0; i < n; ++i) {
            next_flow[i] = state[2*i+2];
            next_pressure[i+1] = state[2*i+3];
        }

        y.inlet_flow[k] = state[1];
        y.outlet_pressure[k] = state[width-1];

        std::copy(
            state.begin(),
            state.end(),
            result.states.begin() +
                static_cast<std::size_t>(k)*width);

        pressure.swap(next_pressure);
        flow.swap(next_flow);
        inlet_flow = state[1];
    }

    return result;
}

} // namespace state_finder
