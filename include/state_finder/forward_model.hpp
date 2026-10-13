#pragma once

#include "pipe_sim/integrate.hpp"

#include <vector>

namespace state_finder {

struct InitialState {
    std::vector<double> pressure;
    std::vector<double> flow;
    double inlet_flow;
};

struct ForwardConfiguration {
    int n;
    int steps;
    pipe_sim::TransientParameters parameters;
    pipe_sim::DemandRamp demand;
    pipe_sim::NewtonOptions newton;
};

struct ForwardResult {
    int info = 0;
    int completed_steps = 0;
    double max_mass_defect = 0.0;

    std::vector<double> time;
    std::vector<double> inlet_flow;
    std::vector<double> outlet_flow;
    std::vector<double> outlet_pressure;
    std::vector<double> linepack;

    std::vector<double> final_pressure;
    std::vector<double> final_flow;
    double final_inlet_flow = 0.0;
};

ForwardResult evaluate_forward(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state);

} // namespace state_finder
