#pragma once

#include "state_finder/forward_model.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace state_finder {

struct InitialStateScales {
    double pressure = 1.0e5;
    double flow = 10.0;
};

inline std::vector<double> pack_scaled_state(
    const InitialState& state,
    const InitialState& baseline,
    InitialStateScales scales = {})
{
    const auto n = state.flow.size();

    if (state.pressure.size() != n + 1 ||
        baseline.pressure.size() != n + 1 ||
        baseline.flow.size() != n ||
        !(scales.pressure > 0.0) ||
        !(scales.flow > 0.0))
    {
        throw std::invalid_argument(
            "Invalid initial-state dimensions or scales");
    }

    std::vector<double> x(2*n + 1);

    for (std::size_t i = 0; i < n; ++i) {
        x[i] =
            (state.pressure[i+1] -
             baseline.pressure[i+1]) /
            scales.pressure;

        x[n+i] =
            (state.flow[i] -
             baseline.flow[i]) /
            scales.flow;
    }

    x[2*n] =
        (state.inlet_flow -
         baseline.inlet_flow) /
        scales.flow;

    return x;
}

inline InitialState unpack_scaled_state(
    const std::vector<double>& x,
    const InitialState& baseline,
    InitialStateScales scales = {})
{
    const auto n = baseline.flow.size();

    if (baseline.pressure.size() != n + 1 ||
        x.size() != 2*n + 1 ||
        !(scales.pressure > 0.0) ||
        !(scales.flow > 0.0))
    {
        throw std::invalid_argument(
            "Invalid scaled initial-state dimensions");
    }

    InitialState state = baseline;

    // Inlet pressure is prescribed and never optimized.
    for (std::size_t i = 0; i < n; ++i) {
        state.pressure[i+1] =
            baseline.pressure[i+1] +
            scales.pressure*x[i];

        state.flow[i] =
            baseline.flow[i] +
            scales.flow*x[n+i];
    }

    state.inlet_flow =
        baseline.inlet_flow +
        scales.flow*x[2*n];

    return state;
}

inline std::vector<double> scale_adjoint_gradient(
    const std::vector<double>& gradient,
    std::size_t n,
    InitialStateScales scales = {})
{
    if (gradient.size() != 2*n + 1)
        throw std::invalid_argument(
            "Invalid adjoint gradient dimension");

    std::vector<double> scaled = gradient;

    for (std::size_t i = 0; i < n; ++i) {
        scaled[i] *= scales.pressure;
        scaled[n+i] *= scales.flow;
    }

    scaled[2*n] *= scales.flow;

    return scaled;
}

} // namespace state_finder
