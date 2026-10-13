#include "state_finder/finite_difference.hpp"

#include <cmath>
#include <stdexcept>

namespace state_finder {

std::vector<double> pack_initial_state(
    const InitialState& state)
{
    if (state.pressure.size() != state.flow.size() + 1 ||
        state.flow.size() < 2) {
        throw std::invalid_argument("Invalid initial state");
    }

    std::vector<double> variables;
    variables.reserve(2 * state.flow.size() + 1);

    for (std::size_t i = 1; i < state.pressure.size(); ++i)
        variables.push_back(state.pressure[i]);

    for (double value : state.flow)
        variables.push_back(value);

    variables.push_back(state.inlet_flow);

    return variables;
}

InitialState unpack_initial_state(
    const InitialState& reference,
    const std::vector<double>& variables)
{
    const std::size_t n = reference.flow.size();

    if (reference.pressure.size() != n + 1 ||
        variables.size() != 2*n + 1) {
        throw std::invalid_argument(
            "Invalid optimization-vector dimensions");
    }

    InitialState result = reference;

    for (std::size_t i = 0; i < n; ++i) {
        result.pressure[i+1] = variables[i];
        result.flow[i] = variables[n+i];
    }

    result.inlet_flow = variables[2*n];

    return result;
}

DirectionalDifference central_directional_difference(
    const ForwardConfiguration& configuration,
    const InitialState& reference,
    const Observations& observations,
    const std::vector<double>& direction,
    double step)
{
    const auto variables = pack_initial_state(reference);

    if (direction.size() != variables.size())
        throw std::invalid_argument("Invalid direction dimensions");

    if (!std::isfinite(step) || step <= 0.0)
        throw std::invalid_argument("Invalid difference step");

    std::vector<double> plus = variables;
    std::vector<double> minus = variables;

    bool nonzero = false;

    for (std::size_t i = 0; i < variables.size(); ++i) {
        if (!std::isfinite(direction[i]))
            throw std::invalid_argument("Nonfinite direction");

        const double delta = step * direction[i];

        if (!std::isfinite(delta))
            throw std::invalid_argument("Nonfinite perturbation");

        plus[i] += delta;
        minus[i] -= delta;

        if (plus[i] != variables[i] || minus[i] != variables[i])
            nonzero = true;
    }

    if (!nonzero)
        throw std::invalid_argument(
            "Perturbation is below floating-point resolution");

    const auto state_plus =
        unpack_initial_state(reference, plus);

    const auto state_minus =
        unpack_initial_state(reference, minus);

    const double jp = evaluate_objective(
        configuration, state_plus, observations).value;

    const double jm = evaluate_objective(
        configuration, state_minus, observations).value;

    return {
        step,
        jp,
        jm,
        (jp-jm)/(2.0*step)
    };
}

} // namespace state_finder
