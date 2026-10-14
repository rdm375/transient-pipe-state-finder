#include "state_finder/correlated_joint_gauss_newton.hpp"
#include "state_finder/joint_regularization_preconditioner.hpp"
#include "state_finder/steady_state_initializer.hpp"

#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
void dsyev_(
    const char* jobz,
    const char* uplo,
    const int* n,
    double* a,
    const int* lda,
    double* w,
    double* work,
    const int* lwork,
    int* info);
}

using namespace state_finder;

namespace {

constexpr int n = 20;
constexpr int steps = 60;
constexpr double dt = 10.0;
constexpr double tau = 120.0;

std::vector<double> load_noise(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file)
        throw std::runtime_error(
            "Cannot open measurement file: " + filename);

    std::string line;
    std::getline(file, line);

    if (line !=
        "measurement_index,noise_normalized,whitened_noise")
        throw std::runtime_error(
            "Unexpected measurement CSV header");

    std::vector<double> noise;
    noise.reserve(4*(steps+1));

    while (std::getline(file, line)) {
        if (line.empty())
            continue;

        std::istringstream input(line);
        std::string index_field;
        std::string noise_field;
        std::string whitened_field;
        std::string extra;

        if (!std::getline(input, index_field, ',') ||
            !std::getline(input, noise_field, ',') ||
            !std::getline(input, whitened_field, ',') ||
            std::getline(input, extra, ','))
            throw std::runtime_error(
                "Malformed measurement CSV row");

        const auto index = std::stoul(index_field);

        if (index != noise.size())
            throw std::runtime_error(
                "Unexpected measurement CSV ordering");

        const double value = std::stod(noise_field);

        if (!std::isfinite(value))
            throw std::runtime_error(
                "Nonfinite measurement noise");

        noise.push_back(value);
    }

    if (noise.size() != 4*(steps+1))
        throw std::runtime_error(
            "Expected exactly 244 measurements");

    return noise;
}

std::vector<double> pressures(
    std::span<const double> state)
{
    std::vector<double> result(n+1);
    result[0] = state[0];

    for (int i = 1; i <= n; ++i)
        result[i] = state[2*i+1];

    return result;
}

double inventory(
    std::span<const double> state,
    const ForwardConfiguration& config)
{
    return pipe_sim::linepack_cz(
        n, pressures(state), config.parameters);
}

double norm(const std::vector<double>& values)
{
    return std::sqrt(
        std::inner_product(
            values.begin(),
            values.end(),
            values.begin(),
            0.0));
}

} // namespace


// M27_C7_SPECTRUM
// Undamped measurement information and terminal-inventory sensitivity.
void run_m27_c7_spectrum(
    const ForwardConfiguration& config,
    const BoundaryObservationRecord& observations,
    const JointInverseVariables& baseline,
    const TemporalMeasurementCovariance& covariance,
    JointVariableScales scales,
    const std::vector<double>& x,
    const std::string& prefix)
{
    const int m = static_cast<int>(x.size());

    if (m != 163)
        throw std::runtime_error("Unexpected M27-C7 dimension");

    // Column-major storage for LAPACK.
    std::vector<double> H(m*m, 0.0);

    const auto A =
        make_correlated_joint_gauss_newton_operator(
            config, observations, baseline, x,
            0.0, covariance, scales);

    for (int j = 0; j < m; ++j) {
        std::vector<double> e(m, 0.0);
        e[j] = 1.0;

        const auto column = A(e);

        if (column.size() != static_cast<std::size_t>(m))
            throw std::runtime_error(
                "M27-C7 operator dimension mismatch");

        for (int i = 0; i < m; ++i) {
            if (!std::isfinite(column[i]))
                throw std::runtime_error(
                    "Nonfinite information matrix");

            H[i + m*j] = column[i];
        }
    }

    double max_entry = 0.0;
    double max_asymmetry = 0.0;

    for (int j = 0; j < m; ++j) {
        for (int i = 0; i < m; ++i) {
            max_entry = std::max(
                max_entry, std::abs(H[i+m*j]));

            max_asymmetry = std::max(
                max_asymmetry,
                std::abs(H[i+m*j]-H[j+m*i]));
        }
    }

    const double relative_asymmetry =
        max_asymmetry/std::max(1.0, max_entry);

    if (relative_asymmetry > 1.0e-8)
        throw std::runtime_error(
            "M27-C7 information matrix fails symmetry check");

    // Symmetrize roundoff only after checking the defect.
    for (int j = 0; j < m; ++j) {
        for (int i = j+1; i < m; ++i) {
            const double v =
                0.5*(H[i+m*j]+H[j+m*i]);

            H[i+m*j] = v;
            H[j+m*i] = v;
        }
    }

    std::vector<double> eigenvalues(m);
    int info = 0;
    int lwork = -1;
    double query = 0.0;

    const char jobz = 'V';
    const char uplo = 'U';

    dsyev_(
        &jobz, &uplo, &m,
        H.data(), &m,
        eigenvalues.data(),
        &query, &lwork, &info);

    if (info != 0 || !std::isfinite(query))
        throw std::runtime_error(
            "M27-C7 LAPACK workspace query failed");

    lwork = std::max(3*m, static_cast<int>(query));
    std::vector<double> work(lwork);

    dsyev_(
        &jobz, &uplo, &m,
        H.data(), &m,
        eigenvalues.data(),
        work.data(), &lwork, &info);

    if (info != 0)
        throw std::runtime_error(
            "M27-C7 LAPACK eigensolver failed");

    // Terminal inventory gradient in scaled coordinates.
    // Central differences use the full nonlinear trajectory.
    const auto terminal_inventory =
        [&](const std::vector<double>& z)
    {
        const auto variables =
            unpack_joint_variables(z, baseline, scales);

        const auto trajectory =
            record_joint_trajectory(config, variables);

        return inventory(
            trajectory.state(config.steps), config);
    };

    const double inventory_at_x = terminal_inventory(x);
    std::vector<double> inventory_gradient(m);

    constexpr double h = 1.0e-4;

    for (int j = 0; j < m; ++j) {
        auto plus = x;
        auto minus = x;

        plus[j] += h;
        minus[j] -= h;

        inventory_gradient[j] =
            (terminal_inventory(plus)
             - terminal_inventory(minus))/(2.0*h);

        if (!std::isfinite(inventory_gradient[j]))
            throw std::runtime_error(
                "Nonfinite inventory gradient");
    }

    const double largest = eigenvalues.back();

    if (!(largest > 0.0))
        throw std::runtime_error(
            "M27-C7 has no positive information eigenvalue");

    // Report several thresholds; do not assert a unique
    // numerical rank based on one arbitrary cutoff.
    const std::array<double, 4> thresholds{
        1.0e-6, 1.0e-8, 1.0e-10, 1.0e-12
    };

    std::ofstream spectrum(prefix + "_eigenvalues.csv");

    if (!spectrum)
        throw std::runtime_error(
            "Cannot create M27-C7 eigenvalue CSV");

    spectrum << std::setprecision(17);
    spectrum
        << "mode,eigenvalue,relative_eigenvalue,"
        << "inventory_sensitivity_kg,"
        << "initial_fraction,inlet_fraction,outlet_fraction\n";

    std::ofstream weak(prefix + "_weak_modes.csv");

    if (!weak)
        throw std::runtime_error(
            "Cannot create M27-C7 weak-mode CSV");

    weak << std::setprecision(17);
    weak
        << "mode,eigenvalue,inventory_sensitivity_kg,"
        << "initial_fraction,inlet_fraction,outlet_fraction\n";

    // Joint coordinate ordering:
    // initial: 2*n+1 = 41
    // inlet:   steps+1 = 61
    // outlet:  steps+1 = 61
    const int initial_end = 2*config.n+1;
    const int inlet_end = initial_end+config.steps+1;

    double maximum_eigen_residual = 0.0;

    for (int k = 0; k < m; ++k) {
        double sensitivity = 0.0;
        double initial_fraction = 0.0;
        double inlet_fraction = 0.0;
        double outlet_fraction = 0.0;

        std::vector<double> v(m);

        for (int i = 0; i < m; ++i) {
            const double value = H[i+m*k];
            v[i] = value;

            sensitivity +=
                inventory_gradient[i]*value;

            const double energy = value*value;

            if (i < initial_end)
                initial_fraction += energy;
            else if (i < inlet_end)
                inlet_fraction += energy;
            else
                outlet_fraction += energy;
        }

        // Verify the eigenpair against the original
        // matrix-free operator.
        const auto Av = A(v);
        double residual_sq = 0.0;

        for (int i = 0; i < m; ++i) {
            const double r =
                Av[i]-eigenvalues[k]*v[i];
            residual_sq += r*r;
        }

        maximum_eigen_residual = std::max(
            maximum_eigen_residual,
            std::sqrt(residual_sq)/
                std::max(1.0, largest));

        spectrum
            << k << ','
            << eigenvalues[k] << ','
            << eigenvalues[k]/largest << ','
            << sensitivity << ','
            << initial_fraction << ','
            << inlet_fraction << ','
            << outlet_fraction << '\n';

        if (k < 10) {
            weak
                << k << ','
                << eigenvalues[k] << ','
                << sensitivity << ','
                << initial_fraction << ','
                << inlet_fraction << ','
                << outlet_fraction << '\n';
        }
    }

    spectrum.close();
    weak.close();

    std::cout << std::setprecision(17)
        << "M27_C7_SPECTRUM\n"
        << "dimension=" << m << '\n'
        << "inventory_kg=" << inventory_at_x << '\n'
        << "minimum_eigenvalue=" << eigenvalues.front() << '\n'
        << "maximum_eigenvalue=" << largest << '\n'
        << "relative_asymmetry=" << relative_asymmetry << '\n'
        << "maximum_eigenpair_residual="
        << maximum_eigen_residual << '\n';

    for (double threshold : thresholds) {
        int rank = 0;

        for (double value : eigenvalues) {
            if (value > threshold*largest)
                ++rank;
        }

        std::cout
            << "rank_at_relative_threshold_"
            << threshold << '=' << rank << '\n';
    }

    std::cout
        << "eigenvalues_file="
        << prefix << "_eigenvalues.csv\n"
        << "weak_modes_file="
        << prefix << "_weak_modes.csv\n";
}

int main(int argc, char** argv)
{
    try {
        const bool spectrum_mode =
            argc == 6 &&
            std::string(argv[3]) == "--spectrum";

        if (!spectrum_mode && argc != 4 && argc != 5)
            throw std::runtime_error(
                "Usage: state_finder_m27_correlated_reconstruction "
                "<corrected-noise.csv> "
                "<max-iterations> "
                "<solution-output.csv> "
                "[--preconditioned|--dense]");

        const std::string mode =
            (!spectrum_mode && argc == 5) ? argv[4] : "";

        const bool use_preconditioner =
            mode == "--preconditioned";

        const bool use_dense =
            mode == "--dense";

        if (!mode.empty() &&
            !use_preconditioner &&
            !use_dense)
            throw std::runtime_error(
                "Unknown option; expected --preconditioned or --dense");

        const std::string measurement_file = argv[1];

        const std::string iteration_argument = argv[2];

        std::size_t parsed_characters = 0;

        const int max_iterations = std::stoi(
            iteration_argument,
            &parsed_characters);

        if (parsed_characters != iteration_argument.size()
            || max_iterations < 1
            || max_iterations > 1000)
            throw std::runtime_error(
                "Invalid maximum iteration count");

        const std::filesystem::path solution_file =
            spectrum_mode ? std::filesystem::path{} :
                            std::filesystem::path(argv[3]);

        if (!spectrum_mode &&
            std::filesystem::exists(solution_file))
            throw std::runtime_error(
                "Refusing to overwrite existing solution: "
                + solution_file.string());

        constexpr double diameter = 0.6;
        constexpr double flow = 100.0;
        constexpr double pin = 8.0e6;

        const double pi = std::acos(-1.0);
        const double area = pi*diameter*diameter/4.0;

        ForwardConfiguration config{
            n,
            steps,
            pipe_sim::TransientParameters{
                50000.0/n,
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
                pin, flow, flow, 1.0
            },
            pipe_sim::NewtonOptions{
                1.0e-9, 1.0e-12, 30, false
            }
        };

        JointInverseVariables truth;

        truth.initial =
            initialize_steady_state(config, flow);

        truth.boundary.inlet_pressure.assign(
            steps+1, pin);

        truth.boundary.outlet_flow.assign(
            steps+1, flow);

        for (int k = 0; k <= steps; ++k) {
            const double t = k*dt;

            const double fraction = std::clamp(
                (t-120.0)/120.0,
                0.0,
                1.0);

            truth.boundary.outlet_flow[k] =
                flow + 10.0*fraction*fraction*
                (3.0-2.0*fraction);
        }

        const auto reference =
            record_joint_trajectory(config, truth);

        BoundaryObservationRecord observations;
        observations.include_initial = true;
        observations.measured = reference.predicted;

        const auto noise = load_noise(measurement_file);

        // Engineering-normalized measurement convention:
        // [pin/1e4, min/1, pout/1e4, mout/1].
        for (int k = 0; k <= steps; ++k) {
            observations.measured.inlet_pressure[k] +=
                1.0e4*noise[4*k];

            observations.measured.inlet_flow[k] +=
                noise[4*k+1];

            observations.measured.outlet_pressure[k] +=
                1.0e4*noise[4*k+2];

            observations.measured.outlet_flow[k] +=
                noise[4*k+3];
        }

        const JointVariableScales scales{};

        TemporalMeasurementCovariance covariance(
            steps+1,
            dt,
            tau);

        const std::size_t dimension =
            joint_variable_dimension(n, steps);

        if (dimension != 163 ||
            covariance.dimension() != 244)
            throw std::runtime_error(
                "Unexpected M27 problem dimensions");

        // The baseline equals the synthetic truth.
        // This is an initial guess, not a constraint.
        const JointInverseVariables baseline = truth;

        const std::vector<double> initial(
            dimension, 0.0);

        const ObjectiveFunction objective =
            [&](const std::vector<double>& x,
                bool need_gradient)
            {
                return evaluate_correlated_joint_objective(
                    config,
                    observations,
                    baseline,
                    x,
                    covariance,
                    scales,
                    need_gradient);
            };

        const auto factory =
            make_correlated_joint_gauss_newton_factory(
                config,
                observations,
                baseline,
                covariance,
                scales);

        if (spectrum_mode) {
            const std::string point = argv[4];
            const std::string prefix = argv[5];

            std::vector<double> x(dimension, 0.0);

            if (point != "truth") {
                std::ifstream input(point);

                if (!input)
                    throw std::runtime_error(
                        "Cannot open reconstruction: " + point);

                std::string line;
                std::getline(input, line);

                if (line !=
                    "coordinate,reconstructed_scaled_value")
                    throw std::runtime_error(
                        "Unexpected reconstruction CSV header");

                for (std::size_t j = 0; j < dimension; ++j) {
                    if (!std::getline(input, line))
                        throw std::runtime_error(
                            "Truncated reconstruction CSV");

                    std::istringstream row(line);
                    std::string coordinate;
                    std::string value;
                    std::string extra;

                    if (!std::getline(row, coordinate, ',') ||
                        !std::getline(row, value, ',') ||
                        std::getline(row, extra, ',') ||
                        std::stoul(coordinate) != j)
                        throw std::runtime_error(
                            "Malformed reconstruction CSV");

                    x[j] = std::stod(value);

                    if (!std::isfinite(x[j]))
                        throw std::runtime_error(
                            "Nonfinite reconstruction coordinate");
                }

                if (std::getline(input, line))
                    throw std::runtime_error(
                        "Unexpected extra reconstruction row");
            }

            run_m27_c7_spectrum(
                config, observations, baseline,
                covariance, scales, x, prefix);

            return 0;
        }

        GaussNewtonOptions options;
        options.max_iterations = max_iterations;
        options.max_cg_iterations = 200;
        options.gradient_tolerance = 1.0e-7;
        options.initial_damping = 1.0e-3;
        options.dense_linear_solver = use_dense;

        const auto initial_evaluation =
            objective(initial, true);

        if (!initial_evaluation.valid)
            throw std::runtime_error(
                "Initial objective evaluation failed");

        const auto start =
            std::chrono::steady_clock::now();

        GaussNewtonDiagnostics diagnostics;

        GaussNewtonPreconditionerFactory preconditioner;

        if (use_preconditioner) {
            preconditioner =
                make_joint_regularization_preconditioner_factory(
                    n,
                    steps,
                    JointRegularization{});
        }

        const auto result =
            minimize_gauss_newton(
                objective,
                initial,
                factory,
                options,
                &diagnostics,
                preconditioner);

        const auto stop =
            std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(
                stop-start).count();

        const auto recovered_variables =
            unpack_joint_variables(
                result.variables,
                baseline,
                scales);

        const auto recovered =
            record_joint_trajectory(
                config,
                recovered_variables);

        const double terminal_inventory_true =
            inventory(reference.state(steps), config);

        const double terminal_inventory_estimated =
            inventory(recovered.state(steps), config);

        const double terminal_inventory_error =
            terminal_inventory_estimated
            - terminal_inventory_true;

        if (!std::isfinite(terminal_inventory_true)
            || terminal_inventory_true <= 0.0)
            throw std::runtime_error(
                "Invalid reference terminal inventory");

        const double terminal_inventory_relative_error =
            terminal_inventory_error
            / terminal_inventory_true;

        const double terminal_inventory_absolute_error =
            std::abs(terminal_inventory_error);

        const double terminal_inventory_absolute_relative_error =
            std::abs(terminal_inventory_relative_error);

        double minimum_pressure =
            std::numeric_limits<double>::infinity();

        double maximum_mass_defect = 0.0;

        for (int k = 0; k <= steps; ++k) {
            const auto state = recovered.state(k);

            for (double pressure : pressures(state))
                minimum_pressure =
                    std::min(minimum_pressure, pressure);

            if (k == 0)
                continue;

            const double previous_inventory =
                inventory(recovered.state(k-1), config);

            const double current_inventory =
                inventory(state, config);

            const double theta = config.parameters.theta;

            const double inlet_weighted =
                (1.0-theta)*recovered.state(k-1)[1]
                +theta*state[1];

            const double outlet_weighted =
                (1.0-theta)*
                    recovered_variables.boundary.outlet_flow[k-1]
                +theta*
                    recovered_variables.boundary.outlet_flow[k];

            const double defect =
                (current_inventory-previous_inventory)
                -dt*(inlet_weighted-outlet_weighted);

            maximum_mass_defect = std::max(
                maximum_mass_defect,
                std::abs(defect));
        }

        const auto final_evaluation =
            objective(result.variables, true);

        if (!final_evaluation.valid)
            throw std::runtime_error(
                "Final objective evaluation failed");

        const auto weighted_noise =
            covariance.apply_inverse(noise);

        const double truth_objective =
            0.5*std::inner_product(
                noise.begin(),
                noise.end(),
                weighted_noise.begin(),
                0.0);

        std::cout << std::setprecision(17)
            << "M27_NATIVE_CORRELATED_RECONSTRUCTION\n"
            << "dimensions=" << dimension << '\n'
            << "max_iterations=" << max_iterations << '\n'
            << "preconditioned=" << use_preconditioner << '\n'
            << "dense_linear_solver=" << use_dense << '\n'
            << "measurements=" << noise.size() << '\n'
            << "tau_s=" << tau << '\n'
            << "normalized_sensor_sigma=0.5\n"
            << "truth_objective=" << truth_objective << '\n'
            << "initial_objective="
            << initial_evaluation.objective << '\n'
            << "final_objective="
            << result.final_objective << '\n'
            << "verified_final_objective="
            << final_evaluation.objective << '\n'
            << "iterations=" << result.iterations << '\n'
            << "converged=" << result.converged << '\n'
            << "reason=" << result.reason << '\n'
            << "gradient_norm=" << result.gradient_norm << '\n'
            << "objective_evaluations="
            << result.objective_evaluations << '\n'
            << "gradient_evaluations="
            << result.gradient_evaluations << '\n'
            << "pcg_solves="
            << diagnostics.pcg_solves << '\n'
            << "maximum_true_residual_ratio="
            << diagnostics.maximum_true_residual_ratio << '\n'
            << "final_damping="
            << diagnostics.final_damping << '\n'
            << "final_forcing_tolerance="
            << diagnostics.final_forcing_tolerance << '\n'
            << "dense_solves="
            << diagnostics.dense_solves << '\n'
            << "dense_operator_calls="
            << diagnostics.dense_operator_calls << '\n'
            << "pcg_iterations="
            << diagnostics.pcg_iterations << '\n'
            << "operator_calls="
            << diagnostics.operator_calls << '\n'
            << "damping_retries="
            << diagnostics.damping_retries << '\n'
            << "gradient_fallbacks="
            << diagnostics.gradient_fallbacks << '\n'
            << "scaled_solution_norm="
            << norm(result.variables) << '\n'
            << "terminal_inventory_true_kg="
            << terminal_inventory_true << '\n'
            << "terminal_inventory_estimated_kg="
            << terminal_inventory_estimated << '\n'
            << "terminal_inventory_error_kg="
            << terminal_inventory_error << '\n'
            << "terminal_inventory_absolute_error_kg="
            << terminal_inventory_absolute_error << '\n'
            << "terminal_inventory_relative_error="
            << terminal_inventory_relative_error << '\n'
            << "terminal_inventory_relative_error_pct="
            << 100.0*terminal_inventory_relative_error << '\n'
            << "terminal_inventory_absolute_relative_error_pct="
            << 100.0*terminal_inventory_absolute_relative_error
            << '\n'
            << "minimum_pressure_pa="
            << minimum_pressure << '\n'
            << "maximum_mass_defect_kg="
            << maximum_mass_defect << '\n'
            << "elapsed_seconds=" << seconds << '\n';

        if (result.final_objective >
            initial_evaluation.objective + 1.0e-8)
            throw std::runtime_error(
                "Native reconstruction increased objective");

        {
            std::ofstream output(solution_file);

            if (!output)
                throw std::runtime_error(
                    "Cannot create solution file: "
                    + solution_file.string());

            output << "coordinate,reconstructed_scaled_value\n";

            output << std::setprecision(17);

            for (std::size_t j = 0;
                 j < result.variables.size();
                 ++j)
            {
                output
                    << j << ','
                    << result.variables[j]
                    << '\n';
            }

            output.flush();

            if (!output)
                throw std::runtime_error(
                    "Failed writing solution file");
        }

        std::cout
            << "solution_file="
            << solution_file.string()
            << '\n';

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr
            << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
