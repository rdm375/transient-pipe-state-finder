#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct Reference {
    double outlet_pressure;
    double inlet_flow;
    double outlet_flow;
    double linepack;
    double mass_defect;
    long long newton_iterations;
};

Reference reference_for(int steps)
{
    if (steps == 80) {
        return {
            7846825.3670041412,
            103.19772530073323,
            105.0,
            4799630.9833060475,
            3.9544278962466706e-10,
            160
        };
    }

    if (steps == 8000) {
        return {
            7843657.8677442567,
            104.99999993783844,
            105.0,
            4798420.4378263094,
            6.2161561231732776e-08,
            1049
        };
    }

    throw std::invalid_argument("Unsupported reference step count");
}

void check(
    const char* name,
    double actual,
    double expected,
    double tolerance)
{
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance) {

        std::cerr << std::setprecision(17)
                  << "FAIL " << name
                  << ": actual=" << actual
                  << " expected=" << expected
                  << " tolerance=" << tolerance
                  << '\n';

        throw std::runtime_error("Numerical parity failure");
    }
}

void run_case(int steps)
{
    using namespace pipe_sim;

    constexpr int n = 100;

    const double pi = std::acos(-1.0);
    const double diameter = 1.0;
    const double area = pi * diameter * diameter / 4.0;

    const TransientParameters par{
        100000.0 / n,
        15.0,
        0.65,
        diameter,
        area,
        288.15,
        0.9,
        500.0,
        1.1e-5,
        4.5e-5
    };

    const DemandRamp demand{
        8.0e6,
        100.0,
        105.0,
        600.0
    };

    const NewtonOptions options{
        1.0e-9,
        1.0e-12,
        30,
        false
    };

    std::vector<double> pressure(n + 1);
    std::vector<double> flow(n, 100.0);

    double inlet_flow = 100.0;

    const double reynolds =
        inlet_flow * diameter / (area * par.viscosity);

    const double friction_arg =
        par.roughness / (3.7 * diameter) +
        5.74 / std::pow(reynolds, 0.9);

    const double friction =
        0.25 / std::pow(std::log10(friction_arg), 2.0);

    const double coefficient =
        friction * par.z * par.gas_constant *
        par.temperature * inlet_flow * inlet_flow /
        (diameter * area * area);

    for (int i = 0; i <= n; ++i) {
        pressure[i] = std::sqrt(
            demand.inlet_pressure * demand.inlet_pressure -
            coefficient * i * par.dx);
    }

    std::vector<double> time(steps + 1);
    std::vector<double> hmin(steps + 1);
    std::vector<double> hout(steps + 1);
    std::vector<double> hpout(steps + 1);
    std::vector<double> hline(steps + 1);
    std::vector<int> hnit(steps + 1);

    const IntegrationHistory history{
        time, hmin, hout, hpout, hline, hnit
    };

    IntegrationWorkspace workspace(n);

    const auto result = integrate_transient(
        n,
        pressure,
        flow,
        inlet_flow,
        demand,
        steps,
        par,
        options,
        history,
        workspace
    );

    if (result.info != 0 ||
        result.completed_steps != steps) {

        throw std::runtime_error(
            "Forward integration failed");
    }

    const Reference ref = reference_for(steps);

    const auto relative_tolerance = [](double value) {
        return 5.0e-13 * std::abs(value);
    };

    check(
        "outlet pressure",
        pressure[n],
        ref.outlet_pressure,
        relative_tolerance(ref.outlet_pressure)
    );

    check(
        "inlet flow",
        inlet_flow,
        ref.inlet_flow,
        relative_tolerance(ref.inlet_flow)
    );

    check(
        "outlet flow",
        hout[steps],
        ref.outlet_flow,
        relative_tolerance(ref.outlet_flow)
    );

    check(
        "linepack",
        hline[steps],
        ref.linepack,
        relative_tolerance(ref.linepack)
    );

    check(
        "mass conservation defect",
        result.max_mass_defect,
        ref.mass_defect,
        1.0e-9
    );

    long long total_iterations = 0;

    for (int k = 1; k <= steps; ++k)
        total_iterations += hnit[k];

    if (total_iterations != ref.newton_iterations) {
        throw std::runtime_error(
            "Newton iteration count mismatch");
    }

    // Independently verify global mass conservation.
    double max_balance_error = 0.0;

    for (int k = 1; k <= steps; ++k) {

        const double inventory_rate =
            (hline[k] - hline[k - 1]) / par.dt;

        const double inlet_average =
            par.theta * hmin[k] +
            (1.0 - par.theta) * hmin[k - 1];

        const double outlet_average =
            par.theta * hout[k] +
            (1.0 - par.theta) * hout[k - 1];

        const double defect =
            inventory_rate -
            (inlet_average - outlet_average);

        max_balance_error = std::max(
            max_balance_error,
            std::abs(defect)
        );
    }

    check(
        "independent mass conservation",
        max_balance_error,
        result.max_mass_defect,
        1.0e-10
    );

    std::cout << std::setprecision(17)
              << "PASS transient parity"
              << " cells=" << n
              << " steps=" << steps
              << " outlet_pressure=" << pressure[n]
              << " inlet_flow=" << inlet_flow
              << " linepack=" << hline[steps]
              << " mass_defect=" << max_balance_error
              << '\n';
}

} // namespace

int main()
{
    try {
        run_case(80);
        run_case(8000);
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
