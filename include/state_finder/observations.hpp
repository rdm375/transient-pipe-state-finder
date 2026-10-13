#pragma once

#include <vector>

namespace state_finder {

struct Observations {
    // Samples are indexed by time step, including t=0.
    std::vector<double> inlet_flow;
    std::vector<double> outlet_pressure;

    // Positive scaling factors, in physical units.
    double sigma_flow = 1.0;
    double sigma_pressure = 1.0e4;

    // Whether to include the initial observation.
    bool include_initial = false;
};

} // namespace state_finder
