#include "state_finder/adjoint.hpp"
#include "state_finder/finite_difference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace state_finder;

using Clock = std::chrono::steady_clock;

template<class Function>
auto timed(Function&& function)
{
    const auto start = Clock::now();
    auto result = function();
    const auto finish = Clock::now();

    const double seconds =
        std::chrono::duration<double>(
            finish-start).count();

    return std::pair{std::move(result), seconds};
}

InitialState make_initial_state(
    const ForwardConfiguration& config)
{
    const int n = config.n;
    const auto& par = config.parameters;

    InitialState state{
        std::vector<double>(n+1),
        std::vector<double>(n, 100.0),
        100.0
    };

    const double mdot = 100.0;
    const double reynolds =
        mdot*par.diameter/(par.area*par.viscosity);

    const double argument =
        par.roughness/(3.7*par.diameter)
        + 5.74/std::pow(reynolds, 0.9);

    const double friction =
        0.25/std::pow(std::log10(argument), 2);

    const double coefficient =
        friction*par.z*par.gas_constant*
        par.temperature*mdot*mdot/
        (par.diameter*par.area*par.area);

    const double inlet_pressure =
        config.demand.inlet_pressure;

    for (int i = 0; i <= n; ++i) {
        const double squared =
            inlet_pressure*inlet_pressure
            - coefficient*i*par.dx;

        if (!(squared > 0.0))
            throw std::runtime_error(
                "Invalid reference pressure profile");

        state.pressure[i] = std::sqrt(squared);
    }

    return state;
}

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw std::runtime_error("Dot-product size mismatch");

    double result = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        result += a[i]*b[i];

    return result;
}

void run(
    int steps,
    std::ofstream& csv)
{
    constexpr int n = 100;

    const double area = std::acos(-1.0)/4.0;

    const ForwardConfiguration config{
        n,
        steps,
        pipe_sim::TransientParameters{
            100000.0/n,
            15.0,
            0.65,
            1.0,
            area,
            288.15,
            0.9,
            500.0,
            1.1e-5,
            4.5e-5
        },
        pipe_sim::DemandRamp{
            8.0e6,
            100.0,
            105.0,
            600.0
        },
        pipe_sim::NewtonOptions{
            1.0e-9,
            1.0e-12,
            30,
            false
        }
    };

    const auto truth = make_initial_state(config);

    const auto [reference, reference_seconds] =
        timed([&] {
            return evaluate_forward(config, truth);
        });

    if (reference.info != 0 ||
        reference.completed_steps != steps)
        throw std::runtime_error(
            "Reference trajectory failed");

    Observations observations;

    observations.inlet_flow =
        reference.inlet_flow;

    observations.outlet_pressure =
        reference.outlet_pressure;

    observations.sigma_flow = 1.0;
    observations.sigma_pressure = 1.0e4;
    observations.include_initial = true;

    auto candidate = truth;

    for (int i = 1; i <= n; ++i)
        candidate.pressure[i] +=
            1000.0*std::sin(0.031*i);

    for (int i = 0; i < n; ++i)
        candidate.flow[i] +=
            0.5*std::cos(0.047*(i+1));

    candidate.inlet_flow += 0.7;

    const auto [forward, forward_seconds] =
        timed([&] {
            return evaluate_forward(
                config, candidate);
        });

    if (forward.info != 0 ||
        forward.completed_steps != steps)
        throw std::runtime_error(
            "Candidate trajectory failed");

    const auto [adjoint, adjoint_seconds] =
        timed([&] {
            return evaluate_adjoint(
                config, candidate, observations);
        });

    const auto objective =
        evaluate_objective(
            config, candidate, observations);

    if (adjoint.objective != objective.value)
        throw std::runtime_error(
            "Objective parity failure");

    std::vector<double> direction(2*n+1);

    for (int i = 0; i < 2*n+1; ++i) {
        const double scale =
            i < n ? 1.0e5 : 10.0;

        direction[i] =
            scale*std::sin(0.071*(i+1));
    }

    const double derivative =
        dot(adjoint.gradient, direction);

    double scaled_gradient_norm = 0.0;

    for (int i = 0; i < 2*n+1; ++i) {
        const double scale =
            i < n ? 1.0e5 : 10.0;

        const double value =
            scale*adjoint.gradient[i];

        scaled_gradient_norm += value*value;
    }

    scaled_gradient_norm =
        std::sqrt(scaled_gradient_norm);

    const double stepsizes[] = {
        1.0e-2,
        3.0e-3,
        1.0e-3,
        3.0e-4,
        1.0e-4,
        3.0e-5,
        1.0e-5,
        3.0e-6,
        1.0e-6,
        3.0e-7,
        1.0e-7,
        3.0e-8
    };

    std::cout
        << "\nsteps=" << steps
        << " reference_seconds=" << reference_seconds
        << " forward_seconds=" << forward_seconds
        << " adjoint_seconds=" << adjoint_seconds
        << " objective=" << adjoint.objective
        << " scaled_gradient_norm="
        << scaled_gradient_norm
        << " max_backward_error="
        << adjoint.max_linear_backward_error
        << " max_mass_defect="
        << forward.max_mass_defect
        << '\n';

    for (double h : stepsizes) {
        const auto [fd, fd_seconds] =
            timed([&] {
                return central_directional_difference(
                    config,
                    candidate,
                    observations,
                    direction,
                    h);
            });

        const double absolute_error =
            std::abs(fd.derivative-derivative);

        const double normalized_error =
            absolute_error/
            std::max(1.0, std::abs(derivative));

        csv
            << steps << ','
            << h << ','
            << reference_seconds << ','
            << forward_seconds << ','
            << adjoint_seconds << ','
            << fd_seconds << ','
            << adjoint.objective << ','
            << scaled_gradient_norm << ','
            << derivative << ','
            << fd.derivative << ','
            << absolute_error << ','
            << normalized_error << ','
            << adjoint.max_linear_backward_error << ','
            << forward.max_mass_defect
            << '\n';

        std::cout
            << "  h=" << h
            << " FD=" << fd.derivative
            << " adjoint=" << derivative
            << " normalized_error=" << normalized_error
            << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 3)
            throw std::runtime_error(
                "Usage: adjoint_long_trajectory STEPS OUTPUT.csv");

        const int steps = std::stoi(argv[1]);

        if (steps != 80 &&
            steps != 800 &&
            steps != 8000)
            throw std::runtime_error(
                "Supported steps: 80, 800, 8000");

        std::ofstream csv(argv[2]);

        if (!csv)
            throw std::runtime_error(
                "Cannot open output CSV");

        csv << std::setprecision(17);

        csv
            << "steps,h,reference_seconds,"
            << "forward_seconds,adjoint_seconds,"
            << "fd_seconds,objective,"
            << "scaled_gradient_norm,"
            << "adjoint_directional,fd_directional,"
            << "absolute_error,normalized_error,"
            << "max_linear_backward_error,"
            << "max_mass_defect\n";

        std::cout << std::setprecision(16);

        run(steps, csv);

        std::cout
            << "PASS long-trajectory adjoint benchmark\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
