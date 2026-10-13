#include "state_finder/joint_trajectory.hpp"
#include "state_finder/steady_state_initializer.hpp"
#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace state_finder;

int main()
{
    try {
        constexpr int n = 20;
        constexpr int steps = 60;

        constexpr double length = 50000.0;
        constexpr double dt = 10.0;
        constexpr double diameter = 0.6;
        constexpr double pressure = 8.0e6;
        constexpr double flow = 100.0;

        const double pi = std::acos(-1.0);
        const double area = pi*diameter*diameter/4.0;

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                length/n,
                dt,
                0.65,
                diameter,
                area,
                288.15,
                0.9,
                500.0,
                1.1e-5,
                4.5e-5
            },
            pipe_sim::DemandRamp{
                pressure, flow, flow, 1.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9, 1.0e-12, 30, false
            }
        };

        JointInverseVariables truth;

        // Analytical steady profile at the prescribed
        // inlet pressure and initial mass flow.
        truth.initial =
            initialize_steady_state(config, flow);

        truth.boundary.inlet_pressure.assign(
            steps+1, pressure);

        truth.boundary.outlet_flow.assign(
            steps+1, flow);

        // Smooth demand ramp:
        // 100 -> 110 kg/s over 120 seconds,
        // beginning at t = 120 seconds.
        for (int k = 0; k <= steps; ++k) {
            const double t = k*dt;

            const double fraction = std::clamp(
                (t-120.0)/120.0, 0.0, 1.0);

            const double smooth =
                fraction*fraction*(3.0-2.0*fraction);

            truth.boundary.outlet_flow[k] =
                flow + 10.0*smooth;
        }

        // Engineering demand-ramp trajectory.
        // Initial state uses the analytical steady profile.

        // Check whether the analytical steady profile
        // remains stationary under constant boundary inputs.
        JointInverseVariables stationary = truth;

        stationary.boundary.outlet_flow.assign(
            steps+1, flow);

        const auto stationary_trajectory =
            record_joint_trajectory(config, stationary);

        const auto initial_state =
            stationary_trajectory.state(0);

        const auto final_state =
            stationary_trajectory.state(steps);

        double stationary_pressure_drift = 0.0;
        double stationary_flow_drift = 0.0;

        for (int i = 0; i <= n; ++i) {
            const std::size_t index =
                (i == 0) ? 0 : 2*i+1;

            stationary_pressure_drift = std::max(
                stationary_pressure_drift,
                std::abs(
                    final_state[index] -
                    initial_state[index]));
        }

        stationary_flow_drift = std::abs(
            final_state[1] - initial_state[1]);

        for (int i = 0; i < n; ++i) {
            const std::size_t index = 2*i+2;

            stationary_flow_drift = std::max(
                stationary_flow_drift,
                std::abs(
                    final_state[index] -
                    initial_state[index]));
        }

        std::cout << std::setprecision(10)
            << "steady_initial_outlet_pressure_Pa="
            << truth.initial.pressure.back() << '\n'
            << "stationary_pressure_drift_Pa="
            << stationary_pressure_drift << '\n'
            << "stationary_flow_drift_kg_s="
            << stationary_flow_drift << '\n';

        // Apply the outlet-demand disturbance.
        const auto trajectory =
            record_joint_trajectory(config, truth);

        const auto& b = trajectory.predicted;

        std::filesystem::create_directories(
            "benchmarks/results-m23a");

        std::ofstream csv(
            "benchmarks/results-m23a/reference_boundary.csv");

        if (!csv)
            throw std::runtime_error(
                "Cannot open reference CSV");

        csv << std::setprecision(17);

        csv
            << "time_s,inlet_pressure_Pa,"
            << "inlet_flow_kg_s,"
            << "outlet_pressure_Pa,"
            << "outlet_flow_kg_s\n";

        for (int k = 0; k <= steps; ++k) {
            csv
                << k*dt << ','
                << b.inlet_pressure[k] << ','
                << b.inlet_flow[k] << ','
                << b.outlet_pressure[k] << ','
                << b.outlet_flow[k] << '\n';
        }

        // --------------------------------------------------
        // M2.3-B: Discrete linepack and mass conservation.
        //
        // Reuse the simulator's own trapezoidal inventory
        // definition, including boundary half-cells.
        // --------------------------------------------------

        std::ofstream inventory_csv(
            "benchmarks/results-m23a/reference_inventory.csv");

        if (!inventory_csv)
            throw std::runtime_error(
                "Cannot open inventory CSV");

        inventory_csv << std::setprecision(17);

        inventory_csv
            << "time_s,linepack_kg,"
            << "inlet_flow_kg_s,"
            << "outlet_flow_kg_s,"
            << "physical_imbalance_kg_s,"
            << "inventory_rate_kg_s,"
            << "mass_defect_kg_s,"
            << "cumulative_mass_defect_kg\n";

        std::vector<double> linepack(steps+1);

        std::vector<double> nodal_pressure(n+1);

        for (int k = 0; k <= steps; ++k) {
            const auto state = trajectory.state(k);

            nodal_pressure[0] = state[0];

            for (int i = 0; i < n; ++i)
                nodal_pressure[i+1] =
                    state[2*i+3];

            linepack[k] = pipe_sim::linepack_cz(
                n,
                nodal_pressure,
                config.parameters);
        }

        double max_mass_defect = 0.0;
        double cumulative_mass_defect = 0.0;
        double integrated_boundary_imbalance = 0.0;

        for (int k = 0; k <= steps; ++k) {
            const double imbalance =
                b.inlet_flow[k] -
                b.outlet_flow[k];

            double inventory_rate = 0.0;
            double mass_defect = 0.0;

            if (k > 0) {
                inventory_rate =
                    (linepack[k] - linepack[k-1])/dt;

                const double previous_imbalance =
                    b.inlet_flow[k-1] -
                    b.outlet_flow[k-1];

                // Same grouping as pipe_sim::integrate_transient.
                const double weighted_imbalance =
                    config.parameters.theta * imbalance
                    + (1.0-config.parameters.theta) *
                      previous_imbalance;

                mass_defect =
                    inventory_rate - weighted_imbalance;

                cumulative_mass_defect +=
                    dt*mass_defect;

                integrated_boundary_imbalance +=
                    dt*weighted_imbalance;

                max_mass_defect = std::max(
                    max_mass_defect,
                    std::abs(mass_defect));
            }

            inventory_csv
                << k*dt << ','
                << linepack[k] << ','
                << b.inlet_flow[k] << ','
                << b.outlet_flow[k] << ','
                << imbalance << ','
                << inventory_rate << ','
                << mass_defect << ','
                << cumulative_mass_defect << '\n';
        }

        const double linepack_change =
            linepack.back() - linepack.front();

        const double integrated_balance_error =
            linepack_change -
            integrated_boundary_imbalance;

        std::cout << std::setprecision(12)
            << "initial_linepack_kg="
            << linepack.front() << '\n'
            << "final_linepack_kg="
            << linepack.back() << '\n'
            << "linepack_change_kg="
            << linepack_change << '\n'
            << "integrated_boundary_imbalance_kg="
            << integrated_boundary_imbalance << '\n'
            << "max_mass_defect_kg_s="
            << max_mass_defect << '\n'
            << "cumulative_mass_defect_kg="
            << cumulative_mass_defect << '\n'
            << "integrated_balance_error_kg="
            << integrated_balance_error << '\n';

        if (!(linepack_change < 0.0))
            throw std::runtime_error(
                "Expected linepack depletion");

        // The simulator uses double precision and a
        // nonlinear Newton solve. Start with a conservative
        // engineering tolerance; inspect measured defects
        // before tightening it.
        constexpr double mass_defect_tolerance = 1.0e-5;

        if (!std::isfinite(max_mass_defect) ||
            max_mass_defect > mass_defect_tolerance)
        {
            throw std::runtime_error(
                "Discrete mass conservation check failed");
        }

        double minimum_pressure = pressure;
        double maximum_pressure = pressure;

        for (int k = 0; k <= steps; ++k) {
            const auto state = trajectory.state(k);

            for (int i = 0; i <= n; ++i) {
                // Interleaved layout:
                // [p0, inlet_flow, face_flow0, p1, ...].
                const std::size_t index =
                    (i == 0) ? 0 : 2*i + 1;

                const double p = state[index];

                if (!std::isfinite(p) || p <= 0.0)
                    throw std::runtime_error(
                        "Nonphysical pressure");

                minimum_pressure =
                    std::min(minimum_pressure, p);

                maximum_pressure =
                    std::max(maximum_pressure, p);
            }
        }

        std::cout << std::setprecision(10)
            << "cells=" << n << '\n'
            << "steps=" << steps << '\n'
            << "duration_s=" << steps*dt << '\n'
            << "length_m=" << length << '\n'
            << "diameter_m=" << diameter << '\n'
            << "initial_flow_kg_s=" << flow << '\n'
            << "final_outlet_demand_kg_s="
            << b.outlet_flow.back() << '\n'
            << "minimum_pressure_Pa="
            << minimum_pressure << '\n'
            << "maximum_pressure_Pa="
            << maximum_pressure << '\n'
            << "initial_outlet_pressure_Pa="
            << b.outlet_pressure.front() << '\n'
            << "final_outlet_pressure_Pa="
            << b.outlet_pressure.back() << '\n'
            << "final_inlet_flow_kg_s="
            << b.inlet_flow.back() << '\n'
            << "PASS M2.3-A reference trajectory\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL M2.3-A: "
            << error.what() << '\n';

        return 1;
    }
}
