#include "state_finder/forecast_continuation.hpp"
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

double relative_error(double a, double b)
{
    return std::abs(a-b) /
        std::max({1.0, std::abs(a), std::abs(b)});
}

ForwardConfiguration configuration(int steps)
{
    constexpr int n = 20;

    constexpr double diameter = 0.6;
    constexpr double dt = 10.0;

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
            8.0e6,
            100.0,
            100.0,
            1.0
        },
        pipe_sim::NewtonOptions{
            1.0e-9,
            1.0e-12,
            30,
            false
        }
    };
}

JointInverseVariables reference_variables(
    const ForwardConfiguration& config)
{
    JointInverseVariables variables;

    variables.initial =
        initialize_steady_state(config, 100.0);

    variables.boundary.inlet_pressure.assign(
        config.steps+1, 8.0e6);

    variables.boundary.outlet_flow.resize(
        config.steps+1);

    for (int k = 0; k <= config.steps; ++k) {
        const double t =
            k*config.parameters.dt;

        const double f = std::clamp(
            (t-120.0)/120.0,
            0.0,
            1.0);

        variables.boundary.outlet_flow[k] =
            100.0+10.0*f*f*(3.0-2.0*f);
    }

    return variables;
}

AdjustableBoundaryHistory slice_boundary(
    const AdjustableBoundaryHistory& source,
    int first,
    int count)
{
    AdjustableBoundaryHistory result;

    result.inlet_pressure.assign(
        source.inlet_pressure.begin()+first,
        source.inlet_pressure.begin()+first+count+1);

    result.outlet_flow.assign(
        source.outlet_flow.begin()+first,
        source.outlet_flow.begin()+first+count+1);

    return result;
}

}

int main()
{
    try {
        constexpr int split = 60;
        constexpr int total = 120;

        const auto full_config =
            configuration(total);

        const auto variables =
            reference_variables(full_config);

        const auto full =
            record_joint_trajectory(
                full_config, variables);

        const auto half_config =
            configuration(split);

        JointInverseVariables first_variables;

        first_variables.initial =
            variables.initial;

        first_variables.boundary =
            slice_boundary(
                variables.boundary,
                0,
                split);

        const auto first =
            record_joint_trajectory(
                half_config,
                first_variables);

        const auto second_boundary =
            slice_boundary(
                variables.boundary,
                split,
                split);

        const auto second =
            continue_joint_trajectory(
                half_config,
                first.state(split),
                second_boundary);

        double maximum_relative_error = 0.0;
        double maximum_absolute_error = 0.0;

        for (int k = 0; k <= total; ++k) {
            const auto reference =
                full.state(k);

            const auto candidate =
                k <= split
                ? first.state(k)
                : second.state(k-split);

            require(
                reference.size() == candidate.size(),
                "State dimension mismatch");

            for (std::size_t j = 0;
                 j < reference.size(); ++j)
            {
                maximum_absolute_error =
                    std::max(
                        maximum_absolute_error,
                        std::abs(
                            reference[j]-candidate[j]));

                maximum_relative_error =
                    std::max(
                        maximum_relative_error,
                        relative_error(
                            reference[j],candidate[j]));
            }
        }

        std::cout
            << "M28_A1_TRAJECTORY_SPLITTING\n"
            << "maximum_absolute_error="
            << maximum_absolute_error << '\n'
            << "maximum_relative_error="
            << maximum_relative_error << '\n';

        require(
            maximum_relative_error < 1.0e-11,
            "Trajectory continuation mismatch");

        std::cout << "PASS\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return 1;
    }
}
