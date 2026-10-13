#pragma once

#include "state_finder/forward_model.hpp"

namespace state_finder {

// Construct an analytical steady isothermal pipe profile.
//
// Uses the Swamee-Jain Darcy friction approximation and
// the constant-mass-flow pressure-squared relation.
//
// The result is an initial-state approximation, not
// necessarily an exact steady solution of the discrete
// transient equations.
//
// Throws std::invalid_argument for invalid inputs and
// std::runtime_error if the pressure profile is inadmissible.
InitialState initialize_steady_state(
    const ForwardConfiguration& configuration,
    double mass_flow);

} // namespace state_finder
