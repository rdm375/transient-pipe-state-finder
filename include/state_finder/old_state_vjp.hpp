#pragma once

#include "pipe_sim/transient.hpp"

#include <span>

namespace state_finder {

// Compute B^T * lambda, where
//
// B = dR(new_state, old_state)/d(old_state).
//
// Both states use the interleaved representation:
// [p0, inlet_flow, face_flow0, p1, face_flow1, p2, ...].
//
// Prescribed boundary histories are held fixed.
//
// This function differentiates the mathematical residual.
// It does not differentiate Newton iterations or friction caches.
void old_state_vjp(
    int n,
    std::span<const double> old_state,
    const pipe_sim::TransientParameters& par,
    std::span<const double> lambda,
    std::span<double> result);

} // namespace state_finder
