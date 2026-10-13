#include "state_finder/boundary_history_integrator.hpp"

#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace state_finder;
using namespace pipe_sim;

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double relative_error(double a, double b)
{
    return std::abs(a-b) /
        std::max({1.0, std::abs(a), std::abs(b)});
}

double maximum_error(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    require(a.size() == b.size(), "Vector size mismatch");

    double error = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        error = std::max(error, relative_error(a[i], b[i]));

    return error;
}

} // namespace

int main()
{
    try {
        constexpr int n = 20;
        constexpr int steps = 40;

        const double pi = std::acos(-1.0);

        TransientParameters par{
            1000.0,
            15.0,
            0.65,
            1.0,
            pi / 4.0,
            288.15,
            0.9,
            500.0,
            1.1e-5,
            4.5e-5
        };

        NewtonOptions options{
            1.0e-9,
            1.0e-12,
            30,
            false
        };

        DemandRamp demand{
            8.0e6,
            100.0,
            105.0,
            300.0
        };

        // Start from the same smooth, physically plausible
        // pressure and flow profiles.
        std::vector<double> initial_pressure(n+1);
        std::vector<double> initial_flow(n, 100.0);

        const double mdot = 100.0;

        const double reynolds =
            mdot * par.diameter /
            (par.area * par.viscosity);

        const double argument =
            par.roughness / (3.7 * par.diameter) +
            5.74 / std::pow(reynolds, 0.9);

        const double friction =
            0.25 / std::pow(std::log10(argument), 2);

        const double coefficient =
            friction * par.z * par.gas_constant *
            par.temperature * mdot * mdot /
            (par.diameter * par.area * par.area);

        for (int i = 0; i <= n; ++i) {
            initial_pressure[i] = std::sqrt(
                demand.inlet_pressure * demand.inlet_pressure
                - coefficient * i * par.dx
            );
        }

        AdjustableBoundaryHistory boundary;

        boundary.inlet_pressure.assign(
            steps+1, demand.inlet_pressure
        );

        boundary.outlet_flow.resize(steps+1);

        for (int k = 0; k <= steps; ++k) {
            const double time = k * par.dt;

            const double fraction =
                demand.ramp_duration > 0.0
                ? std::min(1.0, time / demand.ramp_duration)
                : 1.0;

            boundary.outlet_flow[k] =
                demand.initial_outlet_flow +
                fraction * (
                    demand.final_outlet_flow -
                    demand.initial_outlet_flow
                );
        }

        // Original integrator.
        auto pressure_old = initial_pressure;
        auto flow_old = initial_flow;
        double inlet_old = mdot;

        std::vector<double> time(steps+1);
        std::vector<double> inlet_history(steps+1);
        std::vector<double> outlet_history(steps+1);
        std::vector<double> pressure_history(steps+1);
        std::vector<double> linepack_history(steps+1);
        std::vector<int> iterations(steps+1);

        IntegrationHistory history{
            time,
            inlet_history,
            outlet_history,
            pressure_history,
            linepack_history,
            iterations
        };

        IntegrationWorkspace workspace(n);

        const auto original = integrate_transient(
            n,
            pressure_old,
            flow_old,
            inlet_old,
            demand,
            steps,
            par,
            options,
            history,
            workspace
        );

        require(original.info == 0,
                "Original integration failed");

        require(original.completed_steps == steps,
                "Original integration incomplete");

        // Adjustable-boundary integrator.
        auto pressure_new = initial_pressure;
        auto flow_new = initial_flow;
        double inlet_new = mdot;

        const auto adjustable = integrate_boundary_history(
            n,
            pressure_new,
            flow_new,
            inlet_new,
            boundary,
            steps,
            par,
            options
        );

        require(adjustable.integration.info == 0,
                "Adjustable integration failed");

        require(adjustable.integration.completed_steps == steps,
                "Adjustable integration incomplete");

        const auto& predicted = adjustable.predicted;

        const double inlet_error = maximum_error(
            inlet_history, predicted.inlet_flow
        );

        const double outlet_error = maximum_error(
            outlet_history, predicted.outlet_flow
        );

        const double pressure_error = maximum_error(
            pressure_history, predicted.outlet_pressure
        );

        const double linepack_error = maximum_error(
            linepack_history, adjustable.linepack
        );

        const double final_pressure_error = maximum_error(
            pressure_old, pressure_new
        );

        const double final_flow_error = maximum_error(
            flow_old, flow_new
        );

        const double final_inlet_error =
            relative_error(inlet_old, inlet_new);

        const double conservation_error =
            relative_error(
                original.max_mass_defect,
                adjustable.integration.max_mass_defect
            );

        const double worst_error = std::max({
            inlet_error,
            outlet_error,
            pressure_error,
            linepack_error,
            final_pressure_error,
            final_flow_error,
            final_inlet_error,
            conservation_error
        });

        std::cout
            << "inlet_flow_relative_error="
            << inlet_error << '\n'
            << "outlet_flow_relative_error="
            << outlet_error << '\n'
            << "outlet_pressure_relative_error="
            << pressure_error << '\n'
            << "linepack_relative_error="
            << linepack_error << '\n'
            << "final_pressure_relative_error="
            << final_pressure_error << '\n'
            << "final_flow_relative_error="
            << final_flow_error << '\n'
            << "conservation_diagnostic_difference="
            << conservation_error << '\n'
            << "worst_relative_error="
            << worst_error << '\n'
            << "original_max_mass_defect="
            << original.max_mass_defect << '\n'
            << "adjustable_max_mass_defect="
            << adjustable.integration.max_mass_defect << '\n';

        require(worst_error < 1.0e-10,
                "Original/adjustable trajectory parity failed");

        // Verify that changing an adjustable inlet-pressure
        // sample actually changes the forward solution.
        auto perturbed_boundary = boundary;

        for (int k = 1; k <= steps; ++k)
            perturbed_boundary.inlet_pressure[k] += 1000.0;

        auto pressure_perturbed = initial_pressure;
        auto flow_perturbed = initial_flow;
        double inlet_perturbed = mdot;

        const auto perturbed = integrate_boundary_history(
            n,
            pressure_perturbed,
            flow_perturbed,
            inlet_perturbed,
            perturbed_boundary,
            steps,
            par,
            options
        );

        require(perturbed.integration.info == 0,
                "Perturbed integration failed");

        double response = 0.0;

        for (int k = 1; k <= steps; ++k) {
            response = std::max(
                response,
                std::abs(
                    perturbed.predicted.outlet_pressure[k] -
                    predicted.outlet_pressure[k]
                )
            );
        }

        std::cout
            << "inlet_pressure_perturbation_pa=1000\n"
            << "maximum_outlet_pressure_response_pa="
            << response << '\n';

        require(response > 0.0,
                "Adjustable inlet pressure has no effect");

        std::cout
            << "PASS boundary-history trajectory parity\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
