#include "state_finder/forward_model.hpp"
#include "state_finder/tangent_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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

namespace {

using namespace state_finder;

constexpr int n = 100;
constexpr int steps = 80;
constexpr int dimension = 2*n + 1;
constexpr int observations_count = 2*(steps + 1);

void require(bool condition, const std::string& message)
{
    if (!condition)
        throw std::runtime_error(message);
}

ForwardConfiguration configuration()
{
    const double area = std::acos(-1.0)/4.0;

    return {
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
}

InitialState reference_state(
    const ForwardConfiguration& config)
{
    const auto& par = config.parameters;

    InitialState state{
        std::vector<double>(n+1),
        std::vector<double>(n, 100.0),
        100.0
    };

    const double mdot = 100.0;

    const double reynolds =
        mdot*par.diameter /
        (par.area*par.viscosity);

    const double argument =
        par.roughness/(3.7*par.diameter) +
        5.74/std::pow(reynolds, 0.9);

    const double friction =
        0.25/std::pow(std::log10(argument), 2.0);

    const double coefficient =
        friction*par.z*par.gas_constant*
        par.temperature*mdot*mdot /
        (par.diameter*par.area*par.area);

    for (int i = 0; i <= n; ++i) {
        const double squared =
            config.demand.inlet_pressure*
            config.demand.inlet_pressure -
            coefficient*i*par.dx;

        require(squared > 0.0,
                "Invalid reference pressure");

        state.pressure[i] = std::sqrt(squared);
    }

    return state;
}

void write_matrix(
    const std::string& filename,
    const std::vector<double>& matrix,
    int rows,
    int columns)
{
    std::ofstream out(filename);

    require(static_cast<bool>(out),
            "Cannot open " + filename);

    out << std::setprecision(17);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            if (j)
                out << ',';

            out << matrix[
                static_cast<std::size_t>(i)*columns+j];
        }

        out << '\n';
    }

    require(static_cast<bool>(out),
            "Failed writing " + filename);
}

} // namespace

int main(int argc, char** argv)
{
    try {
        require(argc <= 2,
                "Usage: m22_information_spectrum [OUTPUT_DIR]");

        const std::string output =
            argc == 2
            ? argv[1]
            : "benchmarks/results-m22";

        std::filesystem::create_directories(output);

        const auto config = configuration();
        const auto truth = reference_state(config);

        const InitialStateScales scales{
            1.0e5,
            10.0
        };

        // Exactly the M2.1 observation scales.
        constexpr double sigma_flow = 1.0;
        constexpr double sigma_pressure = 1.0e4;

        const auto tangent = evaluate_tangent_matrix(
            config,
            truth,
            sigma_flow,
            sigma_pressure,
            scales);

        require(tangent.steps == steps,
                "Unexpected tangent horizon");

        require(tangent.columns == dimension,
                "Unexpected tangent dimension");

        const auto& K = tangent.observations;

        require(
            K.size() ==
                static_cast<std::size_t>(
                    observations_count*dimension),
            "Unexpected sensitivity matrix size");

        for (double value : K)
            require(std::isfinite(value),
                    "Nonfinite sensitivity");

        // G = K^T K, stored column-major for LAPACK.
        std::vector<double> G(
            dimension*dimension, 0.0);

        for (int j = 0; j < dimension; ++j) {
            for (int i = 0; i <= j; ++i) {
                long double sum = 0.0L;

                for (int row = 0;
                     row < observations_count;
                     ++row)
                {
                    const auto offset =
                        static_cast<std::size_t>(row)*
                        dimension;

                    sum +=
                        static_cast<long double>(
                            K[offset+i])*
                        static_cast<long double>(
                            K[offset+j]);
                }

                const double value =
                    static_cast<double>(sum);

                G[j*dimension+i] = value;
                G[i*dimension+j] = value;
            }
        }

        write_matrix(
            output + "/K.csv",
            K,
            observations_count,
            dimension);

        // G is symmetric, so row-major and
        // column-major representations coincide.
        write_matrix(
            output + "/G.csv",
            G,
            dimension,
            dimension);

        std::vector<double> eigenvalues(dimension);

        const char jobz = 'V';
        const char uplo = 'U';

        int info = 0;
        int lwork = -1;
        double workspace_query = 0.0;

        dsyev_(
            &jobz, &uplo,
            &dimension,
            G.data(), &dimension,
            eigenvalues.data(),
            &workspace_query, &lwork,
            &info);

        require(info == 0,
                "LAPACK workspace query failed");

        lwork = std::max(
            3*dimension,
            static_cast<int>(
                std::ceil(workspace_query)));

        std::vector<double> work(lwork);

        dsyev_(
            &jobz, &uplo,
            &dimension,
            G.data(), &dimension,
            eigenvalues.data(),
            work.data(), &lwork,
            &info);

        require(info == 0,
                "LAPACK eigendecomposition failed");

        // LAPACK returns ascending eigenvalues.
        // Eigenvectors are columns of G after dsyev.
        std::ofstream spectrum(
            output + "/spectrum.csv");

        std::ofstream modes(
            output + "/eigenvectors.csv");

        require(
            static_cast<bool>(spectrum) &&
            static_cast<bool>(modes),
            "Cannot open spectral output files");

        spectrum << std::setprecision(17);
        modes << std::setprecision(17);

        spectrum
            << "mode,eigenvalue,singular_value\n";

        modes << "coordinate";

        for (int j = 0; j < dimension; ++j)
            modes << ",mode_" << j;

        modes << '\n';

        for (int j = 0; j < dimension; ++j) {
            const double lambda = eigenvalues[j];

            spectrum
                << j << ','
                << lambda << ','
                << std::sqrt(
                    std::max(0.0, lambda))
                << '\n';
        }

        for (int i = 0; i < dimension; ++i) {
            modes << i;

            for (int j = 0; j < dimension; ++j)
                modes << ',' << G[j*dimension+i];

            modes << '\n';
        }

        require(
            static_cast<bool>(spectrum) &&
            static_cast<bool>(modes),
            "Failed writing spectral results");

        const double lambda_max =
            eigenvalues.back();

        require(
            std::isfinite(lambda_max) &&
            lambda_max > 0.0,
            "Invalid maximum eigenvalue");

        // Diagnostic threshold for eigenvalues
        // of G, relative to its largest eigenvalue.
        //
        // This is a numerical threshold, not a
        // claim of structural identifiability.
        constexpr double relative_threshold = 1.0e-10;

        const double threshold =
            relative_threshold*lambda_max;

        int numerical_rank = 0;

        for (double lambda : eigenvalues)
            if (lambda > threshold)
                ++numerical_rank;

        const int nullity =
            dimension-numerical_rank;

        std::cout << std::setprecision(17);

        std::cout
            << "m22_cells=" << n << '\n'
            << "m22_steps=" << steps << '\n'
            << "m22_state_dimension="
            << dimension << '\n'
            << "m22_observation_dimension="
            << observations_count << '\n'
            << "m22_structural_nullity_lower_bound="
            << dimension-observations_count << '\n'
            << "m22_lambda_max="
            << lambda_max << '\n'
            << "m22_lambda_min="
            << eigenvalues.front() << '\n'
            << "m22_relative_eigenvalue_threshold="
            << relative_threshold << '\n'
            << "m22_numerical_rank="
            << numerical_rank << '\n'
            << "m22_numerical_nullity="
            << nullity << '\n'
            << "m22_max_linear_backward_error="
            << tangent.max_linear_backward_error
            << '\n'
            << "m22_spectrum_csv="
            << output + "/spectrum.csv" << '\n'
            << "m22_modes_csv="
            << output + "/eigenvectors.csv" << '\n';

        std::cout
            << "PASS M2.2a information spectrum\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL M2.2a: "
            << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
