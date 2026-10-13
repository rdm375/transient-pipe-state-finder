#pragma once

#include "pipe_sim/integrate.hpp"
#include "pipe_sim/transient.hpp"
#include "pipe_sim/newton.hpp"

#include "state_finder/observations.hpp"

#include <span>
#include <vector>

namespace state_finder {

// Sampled boundary conditions at t_k = k * dt.
//
// Both arrays must have steps + 1 entries.
struct AdjustableBoundaryHistory {
    std::vector<double> inlet_pressure;
    std::vector<double> outlet_flow;
};

struct BoundaryHistoryIntegrationResult {
    pipe_sim::IntegrationResult integration;
    BoundaryMeasurements predicted;
    std::vector<double> linepack;
    std::vector<int> newton_iterations;
};

// Integrates the existing isothermal pipe equations using
// independently adjustable boundary histories.
//
// The initial inlet pressure must agree with pressure[0].
//
// pressure, flow, and inlet_flow are updated on successful
// integration, following the existing integrator convention.
BoundaryHistoryIntegrationResult integrate_boundary_history(
    int n,
    std::span<double> pressure,
    std::span<double> flow,
    double& inlet_flow,
    const AdjustableBoundaryHistory& boundary,
    int steps,
    const pipe_sim::TransientParameters& parameters,
    const pipe_sim::NewtonOptions& options);

} // namespace state_finder
