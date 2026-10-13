#include "state_finder/tangent_linear.hpp"
#include "state_finder/adjoint.hpp"
#include "state_finder/observation_adjoint.hpp"
#include "state_finder/trajectory.hpp"
#include "state_finder/forward_model.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace state_finder;

int n = 100;
constexpr int steps = 800;

void require(bool condition, const char* message)
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
        mdot*par.diameter/
        (par.area*par.viscosity);

    const double argument =
        par.roughness/(3.7*par.diameter) +
        5.74/std::pow(reynolds, 0.9);

    const double friction =
        0.25/std::pow(std::log10(argument), 2.0);

    const double coefficient =
        friction*par.z*par.gas_constant*
        par.temperature*mdot*mdot/
        (par.diameter*par.area*par.area);

    for (int i = 0; i <= n; ++i) {
        const double squared =
            config.demand.inlet_pressure*
            config.demand.inlet_pressure -
            coefficient*i*par.dx;

        require(
            squared > 0.0,
            "Invalid reference pressure"
        );

        state.pressure[i] = std::sqrt(squared);
    }

    return state;
}

} // namespace



#include <array>
#include <limits>
#include <span>

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

using Vector = std::vector<double>;

double dot(const Vector& a, const Vector& b)
{
    require(a.size() == b.size(), "Dot dimension mismatch");

    double sum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        sum += a[i]*b[i];

    return sum;
}

double norm(const Vector& a)
{
    return std::sqrt(dot(a, a));
}

void axpy(Vector& y, double alpha, const Vector& x)
{
    require(y.size() == x.size(), "AXPY dimension mismatch");

    for (std::size_t i = 0; i < y.size(); ++i)
        y[i] += alpha*x[i];
}

struct EigenResult {
    Vector values;
    std::vector<Vector> vectors;
};

EigenResult symmetric_eigensystem(
    const Vector& matrix,
    int m)
{
    require(
        matrix.size() == static_cast<std::size_t>(m*m),
        "Invalid projected matrix");

    Vector a = matrix;
    Vector values(m);

    int info = 0;
    int lwork = -1;
    double workspace_size = 0.0;

    const char jobz = 'V';
    const char uplo = 'U';

    dsyev_(
        &jobz, &uplo, &m,
        a.data(), &m,
        values.data(),
        &workspace_size, &lwork,
        &info);

    require(info == 0, "LAPACK workspace query failed");

    lwork = std::max(
        3*m,
        static_cast<int>(std::ceil(workspace_size)));

    Vector work(lwork);

    dsyev_(
        &jobz, &uplo, &m,
        a.data(), &m,
        values.data(),
        work.data(), &lwork,
        &info);

    require(info == 0, "Projected eigensolver failed");

    EigenResult result;

    // LAPACK returns ascending eigenvalues.
    for (int i = m-1; i >= 0; --i) {
        result.values.push_back(values[i]);

        Vector vector(m);

        for (int j = 0; j < m; ++j)
            vector[j] = a[i*m+j];

        result.vectors.push_back(std::move(vector));
    }

    return result;
}

struct NormalOperator {
    ForwardConfiguration config;
    Trajectory trajectory;

    int start_step = 0;
    int calls = 0;

    double operator_seconds = 0.0;

    static constexpr double sigma_m = 0.01;
    static constexpr double sigma_p = 10.0;

    InitialStateScales scales{1.0e5, 10.0};

    Vector prior;

    NormalOperator(
        const ForwardConfiguration& c,
        const InitialState& initial,
        int first)
        : config(c),
          trajectory(record_trajectory(c, initial)),
          start_step(first),
          prior(2*c.n+1, 1.0)
    {
        require(
            trajectory.info == 0 &&
            trajectory.completed_steps == config.steps,
            "Reference trajectory failed");

        require(
            start_step >= 0 &&
            start_step <= config.steps,
            "Invalid observation window");

        for (int i = 0; i < config.n; ++i)
            prior[i] = 1.0e4;
    }

    Vector apply(const Vector& v)
    {
        require(
            v.size() == prior.size(),
            "Invalid operator input dimension");

        const auto begin =
            std::chrono::steady_clock::now();

        Vector direction(v.size());

        for (std::size_t j = 0; j < v.size(); ++j) {
            const double scale =
                j < static_cast<std::size_t>(config.n)
                ? scales.pressure
                : scales.flow;

            direction[j] = prior[j]*v[j]/scale;
        }

        const auto tangent =
            evaluate_tangent_linear(
                config,
                trajectory,
                direction,
                scales);

        Vector forcing(2*(config.steps+1), 0.0);

        for (int k = start_step; k <= config.steps; ++k) {
            forcing[2*k] =
                tangent.inlet_flow[k]/sigma_m;

            forcing[2*k+1] =
                tangent.outlet_pressure[k]/sigma_p;
        }

        const auto adjoint =
            apply_observation_adjoint(
                config,
                trajectory,
                forcing,
                start_step,
                sigma_m,
                sigma_p);

        Vector result(v.size());

        for (std::size_t j = 0; j < v.size(); ++j)
            result[j] = prior[j]*adjoint.gradient[j];

        const auto end =
            std::chrono::steady_clock::now();

        operator_seconds +=
            std::chrono::duration<double>(end-begin).count();

        ++calls;

        return result;
    }
};

struct LanczosResult {
    Vector eigenvalues;
    std::vector<Vector> eigenvectors;
    Vector residuals;

    int iterations = 0;
    bool converged = false;
};

LanczosResult lanczos(
    NormalOperator& op,
    int wanted = 3,
    int maximum = 100,
    double tolerance = 1.0e-9)
{
    const int dim = static_cast<int>(op.prior.size());

    maximum = std::min(maximum, dim);

    std::mt19937_64 rng(20261013);
    std::normal_distribution<double> normal(0.0, 1.0);

    Vector q(dim);

    for (double& x : q)
        x = normal(rng);

    const double initial_norm = norm(q);

    for (double& x : q)
        x /= initial_norm;

    std::vector<Vector> basis;
    Vector alpha;
    Vector beta;

    double previous_beta = 0.0;

    LanczosResult output;

    for (int iteration = 0; iteration < maximum; ++iteration) {
        Vector z = op.apply(q);

        const double a = dot(q, z);

        axpy(z, -a, q);

        if (!basis.empty())
            axpy(z, -previous_beta, basis.back());

        basis.push_back(q);
        alpha.push_back(a);

        // Two-pass full reorthogonalization.
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto& basis_vector : basis) {
                const double coefficient =
                    dot(basis_vector, z);

                axpy(z, -coefficient, basis_vector);
            }
        }

        const double next_beta = norm(z);
        const int m = static_cast<int>(basis.size());

        // Column-major projected tridiagonal matrix.
        Vector projected(m*m, 0.0);

        for (int i = 0; i < m; ++i) {
            projected[i*m+i] = alpha[i];

            if (i > 0) {
                projected[(i-1)*m+i] = beta[i-1];
                projected[i*m+(i-1)] = beta[i-1];
            }
        }

        const auto eig =
            symmetric_eigensystem(projected, m);

        if (m >= wanted) {
            output.eigenvalues.clear();
            output.eigenvectors.clear();
            output.residuals.clear();

            bool all_converged = true;

            for (int mode = 0; mode < wanted; ++mode) {
                const double value = eig.values[mode];

                Vector vector(dim, 0.0);

                for (int j = 0; j < m; ++j)
                    axpy(
                        vector,
                        eig.vectors[mode][j],
                        basis[j]);

                const double vector_norm = norm(vector);

                for (double& x : vector)
                    x /= vector_norm;

                // Lanczos Ritz residual estimate.
                const double residual =
                    std::abs(
                        next_beta *
                        eig.vectors[mode][m-1]);

                const double relative_residual =
                    residual /
                    std::max(std::abs(value), 1.0);

                output.eigenvalues.push_back(value);
                output.eigenvectors.push_back(
                    std::move(vector));
                output.residuals.push_back(
                    relative_residual);

                if (relative_residual > tolerance)
                    all_converged = false;
            }

            output.iterations = m;

            if (all_converged) {
                output.converged = true;
                break;
            }
        }

        if (next_beta <
            1.0e-14*std::max(std::abs(a), 1.0)) {
            output.converged = m >= wanted;
            break;
        }

        if (iteration+1 == maximum)
            break;

        beta.push_back(next_beta);
        previous_beta = next_beta;

        q = std::move(z);

        for (double& x : q)
            x /= next_beta;
    }

    return output;
}

void write_results(
    const std::string& prefix,
    const LanczosResult& result)
{
    std::ofstream eigenvalues(prefix + "_eigenpairs.csv");

    require(
        static_cast<bool>(eigenvalues),
        "Cannot write eigenpairs");

    eigenvalues << std::setprecision(17);
    eigenvalues
        << "mode,eigenvalue,singular_value,relative_residual\n";

    for (std::size_t i = 0; i < result.eigenvalues.size(); ++i) {
        const double value = result.eigenvalues[i];

        eigenvalues
            << i+1 << ','
            << value << ','
            << std::sqrt(std::max(value, 0.0)) << ','
            << result.residuals[i] << '\n';
    }

    std::ofstream vectors(prefix + "_vectors.csv");

    require(
        static_cast<bool>(vectors),
        "Cannot write eigenvectors");

    vectors << std::setprecision(17);

    const int dim =
        static_cast<int>(result.eigenvectors[0].size());

    for (int j = 0; j < dim; ++j) {
        for (std::size_t i = 0;
             i < result.eigenvectors.size(); ++i) {
            if (i)
                vectors << ',';

            vectors << result.eigenvectors[i][j];
        }

        vectors << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    try {
        require(
            argc == 3 || argc == 4,
            "Usage: executable START_STEP OUTPUT_PREFIX [CELLS]");

        const int start_step = std::stoi(argv[1]);

        if (argc == 4)
            n = std::stoi(argv[3]);

        require(
            n >= 2 && n <= 10000,
            "Invalid number of cells");

        const auto config = configuration();
        const auto initial = reference_state(config);

        const auto total_start =
            std::chrono::steady_clock::now();

        const auto trajectory_start =
            std::chrono::steady_clock::now();

        NormalOperator op(config, initial, start_step);

        const auto trajectory_end =
            std::chrono::steady_clock::now();

        const double trajectory_seconds =
            std::chrono::duration<double>(
                trajectory_end-trajectory_start).count();

        const auto result = lanczos(op);

        const auto total_end =
            std::chrono::steady_clock::now();

        const double total_seconds =
            std::chrono::duration<double>(
                total_end-total_start).count();

        require(
            result.eigenvalues.size() == 3,
            "Insufficient eigenpairs");

        write_results(argv[2], result);

        std::cout << std::setprecision(17);

        std::cout
            << "M1.8c-3 in-process Lanczos\n"
            << "start_step=" << start_step << '\n'
            << "cells=" << n << '\n'
            << "state_dimension=" << 2*n+1 << '\n'
            << "trajectory_seconds="
            << trajectory_seconds << '\n'
            << "operator_calls=" << op.calls << '\n'
            << "operator_seconds="
            << op.operator_seconds << '\n'
            << "total_seconds=" << total_seconds << '\n'
            << "iterations=" << result.iterations << '\n'
            << "converged=" << result.converged << '\n';

        for (int i = 0; i < 3; ++i) {
            std::cout
                << "mode=" << i+1
                << " eigenvalue="
                << result.eigenvalues[i]
                << " residual="
                << result.residuals[i]
                << '\n';
        }

        require(
            result.converged,
            "Lanczos failed to converge");

        std::cout << "PASS M1.8c-3 in-process Lanczos\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL M1.8c-3: "
            << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
