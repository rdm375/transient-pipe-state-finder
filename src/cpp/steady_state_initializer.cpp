#include "state_finder/steady_state_initializer.hpp"

#include <cmath>
#include <stdexcept>

namespace state_finder {

InitialState initialize_steady_state(
    const ForwardConfiguration& configuration,
    double mass_flow)
{
    const auto& par = configuration.parameters;

    if (configuration.n <= 0 ||
        !std::isfinite(mass_flow) ||
        mass_flow <= 0.0 ||
        !std::isfinite(par.dx) ||
        par.dx <= 0.0 ||
        !std::isfinite(par.diameter) ||
        par.diameter <= 0.0 ||
        !std::isfinite(par.area) ||
        par.area <= 0.0 ||
        !std::isfinite(par.viscosity) ||
        par.viscosity <= 0.0 ||
        !std::isfinite(par.roughness) ||
        par.roughness < 0.0 ||
        !std::isfinite(par.z) ||
        par.z <= 0.0 ||
        !std::isfinite(par.gas_constant) ||
        par.gas_constant <= 0.0 ||
        !std::isfinite(par.temperature) ||
        par.temperature <= 0.0 ||
        !std::isfinite(configuration.demand.inlet_pressure) ||
        configuration.demand.inlet_pressure <= 0.0)
    {
        throw std::invalid_argument(
            "Invalid steady-state initialization parameters");
    }

    const double reynolds =
        mass_flow*par.diameter /
        (par.area*par.viscosity);

    const double argument =
        par.roughness/(3.7*par.diameter) +
        5.74/std::pow(reynolds, 0.9);

    if (!std::isfinite(argument) ||
        argument <= 0.0 ||
        argument == 1.0)
    {
        throw std::runtime_error(
            "Invalid Swamee-Jain friction argument");
    }

    const double friction =
        0.25/std::pow(std::log10(argument), 2.0);

    const double coefficient =
        friction*par.z*par.gas_constant*
        par.temperature*mass_flow*mass_flow /
        (par.diameter*par.area*par.area);

    if (!std::isfinite(coefficient) ||
        coefficient < 0.0)
    {
        throw std::runtime_error(
            "Invalid steady-state pressure coefficient");
    }

    InitialState result;

    result.pressure.resize(configuration.n + 1);
    result.flow.assign(configuration.n, mass_flow);
    result.inlet_flow = mass_flow;

    const double inlet_pressure =
        configuration.demand.inlet_pressure;

    for (int i = 0; i <= configuration.n; ++i) {
        const double x = i*par.dx;

        const double squared =
            inlet_pressure*inlet_pressure -
            coefficient*x;

        if (!std::isfinite(squared) ||
            squared <= 0.0)
        {
            throw std::runtime_error(
                "Steady-state pressure becomes nonpositive");
        }

        result.pressure[i] = std::sqrt(squared);
    }

    return result;
}

} // namespace state_finder
