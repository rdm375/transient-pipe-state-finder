#pragma once

#include "state_finder/forward_model.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace state_finder {

struct Trajectory {
    int n = 0;
    int steps = 0;
    int info = 0;
    int completed_steps = 0;
    double max_mass_defect = 0.0;

    // Row-major storage: (steps + 1) x (2*n + 2).
    std::vector<double> states;

    std::vector<double> time;
    std::vector<double> inlet_flow;
    std::vector<double> outlet_flow;
    std::vector<double> outlet_pressure;
    std::vector<double> linepack;
    std::vector<int> newton_iterations;

    [[nodiscard]] std::size_t state_size() const noexcept
    {
        return static_cast<std::size_t>(2*n + 2);
    }

    [[nodiscard]] std::span<const double> state(int k) const;
};

Trajectory record_trajectory(
    const ForwardConfiguration& configuration,
    const InitialState& initial_state);

} // namespace state_finder
