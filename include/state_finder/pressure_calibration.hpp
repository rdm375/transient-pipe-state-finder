#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace state_finder {

struct PressureCalibrationBias {
    double inlet_pa = 0.0;
    double outlet_pa = 0.0;
};

// Measurement order at each time:
// [inlet pressure, inlet flow, outlet pressure, outlet flow].
//
// The pressure normalizer is an engineering scale,
// not a sensor standard deviation.
inline std::vector<double> apply_pressure_calibration(
    std::span<const double> normalized_residual,
    PressureCalibrationBias bias,
    double pressure_scale_pa = 1.0e4)
{
    if (normalized_residual.size() % 4 != 0)
        throw std::invalid_argument(
            "Calibration residual dimension must be divisible by four");

    if (!std::isfinite(pressure_scale_pa)
        || pressure_scale_pa <= 0.0)
        throw std::invalid_argument(
            "Invalid pressure calibration scale");

    if (!std::isfinite(bias.inlet_pa)
        || !std::isfinite(bias.outlet_pa))
        throw std::invalid_argument(
            "Nonfinite pressure calibration bias");

    std::vector<double> result(
        normalized_residual.begin(),
        normalized_residual.end());

    for (std::size_t k = 0; k < result.size()/4; ++k) {
        result[4*k] += bias.inlet_pa/pressure_scale_pa;
        result[4*k+2] += bias.outlet_pa/pressure_scale_pa;
    }

    return result;
}

// Transpose action B^T*w for the physical pressure biases.
//
// The input is an observation-space vector, typically
// C^-1*r. Output units are inverse pascals times
// the dimensionless objective scale.
inline std::array<double, 2>
pressure_calibration_transpose(
    std::span<const double> weighted_residual,
    double pressure_scale_pa = 1.0e4)
{
    if (weighted_residual.size() % 4 != 0)
        throw std::invalid_argument(
            "Calibration weighted residual dimension mismatch");

    if (!std::isfinite(pressure_scale_pa)
        || pressure_scale_pa <= 0.0)
        throw std::invalid_argument(
            "Invalid pressure calibration scale");

    std::array<double, 2> result{0.0, 0.0};

    for (std::size_t k = 0;
         k < weighted_residual.size()/4;
         ++k)
    {
        result[0] += weighted_residual[4*k]
                     /pressure_scale_pa;

        result[1] += weighted_residual[4*k+2]
                     /pressure_scale_pa;
    }

    return result;
}

} // namespace state_finder
