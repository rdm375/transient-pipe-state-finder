#include "state_finder/temporal_measurement_covariance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using state_finder::TemporalMeasurementCovariance;

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

std::vector<double> dense_solve(
    std::vector<double> matrix,
    std::vector<double> rhs,
    int n)
{
    for (int k = 0; k < n; ++k) {
        int pivot = k;

        for (int i = k+1; i < n; ++i) {
            if (std::abs(matrix[i*n+k]) >
                std::abs(matrix[pivot*n+k]))
                pivot = i;
        }

        require(
            std::abs(matrix[pivot*n+k]) > 1.e-14,
            "Singular dense covariance");

        if (pivot != k) {
            for (int j = 0; j < n; ++j)
                std::swap(
                    matrix[k*n+j],
                    matrix[pivot*n+j]);

            std::swap(rhs[k], rhs[pivot]);
        }

        const double diagonal = matrix[k*n+k];

        for (int i = k+1; i < n; ++i) {
            const double factor =
                matrix[i*n+k]/diagonal;

            for (int j = k; j < n; ++j)
                matrix[i*n+j] -=
                    factor*matrix[k*n+j];

            rhs[i] -= factor*rhs[k];
        }
    }

    std::vector<double> solution(n);

    for (int i = n-1; i >= 0; --i) {
        double value = rhs[i];

        for (int j = i+1; j < n; ++j)
            value -= matrix[i*n+j]*solution[j];

        solution[i] = value/matrix[i*n+i];
    }

    return solution;
}

int main()
{
    constexpr int times = 8;
    constexpr int channels = 4;
    constexpr int dimension = times*channels;

    constexpr double dt = 10.0;
    constexpr double tau = 120.0;

    const std::array<double, 4> sigma{
        0.5, 0.5, 0.5, 0.5
    };

    TemporalMeasurementCovariance covariance(
        times, dt, tau, sigma);

    std::vector<double> dense(
        dimension*dimension, 0.0);

    std::vector<double> rhs(dimension);

    for (int k = 0; k < times; ++k) {
        for (int c = 0; c < channels; ++c) {
            const int row = 4*k+c;

            rhs[row] =
                std::sin(0.37*(row+1)) +
                0.2*std::cos(0.11*row);

            for (int j = 0; j < times; ++j) {
                const int col = 4*j+c;

                dense[row*dimension+col] =
                    sigma[c]*sigma[c]*
                    std::exp(
                        -std::abs(k-j)*dt/tau);
            }
        }
    }

    const auto reference =
        dense_solve(dense, rhs, dimension);

    const auto actual =
        covariance.apply_inverse(rhs);

    double maximum_error = 0.0;
    double maximum_reference = 0.0;

    for (int i = 0; i < dimension; ++i) {
        maximum_error = std::max(
            maximum_error,
            std::abs(actual[i]-reference[i]));

        maximum_reference = std::max(
            maximum_reference,
            std::abs(reference[i]));
    }

    const double relative_error =
        maximum_error/
        std::max(1.0, maximum_reference);

    std::cout
        << "M27 covariance inverse relative error: "
        << relative_error << '\n';

    require(
        relative_error < 1.e-11,
        "Covariance inverse mismatch");

    const double quadratic =
        covariance.quadratic_form(rhs);

    double reference_quadratic = 0.0;

    for (int i = 0; i < dimension; ++i)
        reference_quadratic +=
            rhs[i]*reference[i];

    const double quadratic_error =
        std::abs(quadratic-reference_quadratic)/
        std::max(1.0, std::abs(reference_quadratic));

    std::cout
        << "M27 quadratic-form relative error: "
        << quadratic_error << '\n';

    require(
        quadratic_error < 1.e-11,
        "Covariance quadratic-form mismatch");

    // Independent-noise limit.
    TemporalMeasurementCovariance independent(
        times, dt, 0.0, sigma);

    const auto independent_result =
        independent.apply_inverse(rhs);

    for (int i = 0; i < dimension; ++i)
        require(
            std::abs(independent_result[i]-4.0*rhs[i])
                < 1.e-12,
            "Independent covariance limit failed");

    // Single-time-sample edge case.
    TemporalMeasurementCovariance single(
        1, dt, tau, sigma);

    const std::vector<double> one{
        1.0, -2.0, 3.0, -4.0
    };

    const auto single_result =
        single.apply_inverse(one);

    for (std::size_t i = 0; i < one.size(); ++i)
        require(
            std::abs(single_result[i]-4.0*one[i])
                < 1.e-12,
            "Single-sample covariance failed");

    std::cout
        << "M27 covariance tests: PASS\n";
}
