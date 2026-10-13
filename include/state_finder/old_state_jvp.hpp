#pragma once

#include "pipe_sim/transient.hpp"

#include <span>

namespace state_finder {

// Compute B*v, where
//
// B = dR(new_state, old_state)/d(old_state).
//
// State ordering:
// [p0, inlet_flow, face_flow0, p1, face_flow1, p2, ...]
//
// Prescribed boundary histories are held fixed.
void old_state_jvp(
    int n,
    std::span<const double> old_state,
    const pipe_sim::TransientParameters& par,
    std::span<const double> direction,
    std::span<double> result);

} // namespace state_finder
