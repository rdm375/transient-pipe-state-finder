#include "state_finder/four_channel_objective.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace state_finder;

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

bool close(double a, double b, double tol = 1e-12)
{
    return std::abs(a-b) <=
        tol * std::max({1.0, std::abs(a), std::abs(b)});
}

} // namespace

int main()
{
    try {
        BoundaryObservationRecord observations;

        observations.include_initial = true;

        observations.measured.inlet_pressure =
            {8.0e6, 8.0e6};

        observations.measured.inlet_flow =
            {100.0, 101.0};

        observations.measured.outlet_pressure =
            {7.9e6, 7.9e6};

        observations.measured.outlet_flow =
            {100.0, 102.0};

        observations.scales.inlet_pressure = 1.0e4;
        observations.scales.inlet_flow = 1.0;
        observations.scales.outlet_pressure = 1.0e4;
        observations.scales.outlet_flow = 1.0;

        FourChannelPrediction prediction;

        prediction.boundary = observations.measured;

        // At t=0, every normalized residual equals +1.
        prediction.boundary.inlet_pressure[0] += 1.0e4;
        prediction.boundary.inlet_flow[0] += 1.0;
        prediction.boundary.outlet_pressure[0] += 1.0e4;
        prediction.boundary.outlet_flow[0] += 1.0;

        auto result = evaluate_four_channel_objective(
            observations, prediction);

        require(close(result.total, 2.0),
                "Four-channel objective mismatch");

        require(close(result.inlet_pressure, 0.5),
                "Inlet-pressure contribution mismatch");

        require(close(result.inlet_flow, 0.5),
                "Inlet-flow contribution mismatch");

        require(close(result.outlet_pressure, 0.5),
                "Outlet-pressure contribution mismatch");

        require(close(result.outlet_flow, 0.5),
                "Outlet-flow contribution mismatch");

        require(result.residuals.size() == 8,
                "Residual dimension mismatch");

        for (int j = 0; j < 4; ++j)
            require(close(result.residuals[j], 1.0),
                    "Incorrect residual ordering");

        for (int j = 4; j < 8; ++j)
            require(close(result.residuals[j], 0.0),
                    "Unexpected second-sample residual");

        const auto gradient =
            four_channel_observation_gradient(
                observations, prediction);

        require(close(gradient.inlet_pressure[0], 1.0e-4),
                "Inlet-pressure gradient mismatch");

        require(close(gradient.inlet_flow[0], 1.0),
                "Inlet-flow gradient mismatch");

        require(close(gradient.outlet_pressure[0], 1.0e-4),
                "Outlet-pressure gradient mismatch");

        require(close(gradient.outlet_flow[0], 1.0),
                "Outlet-flow gradient mismatch");

        // Test the include_initial=false convention.
        observations.include_initial = false;

        result = evaluate_four_channel_objective(
            observations, prediction);

        require(close(result.total, 0.0),
                "Initial-sample exclusion failed");

        require(result.residuals.size() == 4,
                "Excluded-sample residual dimension mismatch");

        std::cout
            << "PASS four-channel objective\n"
            << "channels=4\n"
            << "samples=2\n"
            << "normalized_initial_objective=2\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
