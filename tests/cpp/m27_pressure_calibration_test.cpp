#include "state_finder/pressure_calibration.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace state_finder;

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    require(a.size() == b.size(), "Dot dimension mismatch");

    return std::inner_product(
        a.begin(), a.end(), b.begin(), 0.0);
}

} // namespace

int main()
{
    try {
        constexpr std::size_t time_count = 61;

        std::vector<double> residual(4*time_count);
        std::vector<double> forcing(4*time_count);

        for (std::size_t i = 0; i < residual.size(); ++i) {
            residual[i] = std::sin(0.13*double(i));
            forcing[i] = std::cos(0.17*double(i));
        }

        const auto zero = apply_pressure_calibration(
            residual,
            PressureCalibrationBias{});

        require(
            zero == residual,
            "Zero-bias regression failed");

        const PressureCalibrationBias bias{
            2000.0,
            -3000.0
        };

        const auto shifted =
            apply_pressure_calibration(residual, bias);

        for (std::size_t k = 0; k < time_count; ++k) {
            require(
                std::abs(
                    shifted[4*k]-residual[4*k]-0.2
                ) < 1.0e-12,
                "Inlet pressure shift failed");

            require(
                shifted[4*k+1] == residual[4*k+1],
                "Inlet flow changed");

            require(
                std::abs(
                    shifted[4*k+2]-residual[4*k+2]+0.3
                ) < 1.0e-12,
                "Outlet pressure shift failed");

            require(
                shifted[4*k+3] == residual[4*k+3],
                "Outlet flow changed");
        }

        const auto transpose =
            pressure_calibration_transpose(forcing);

        const auto response =
            apply_pressure_calibration(
                std::vector<double>(residual.size(), 0.0),
                bias);

        const double lhs = dot(response, forcing);

        const double rhs =
            bias.inlet_pa*transpose[0]
            +bias.outlet_pa*transpose[1];

        const double tolerance =
            1.0e-11*std::max(
                {1.0, std::abs(lhs), std::abs(rhs)});

        require(
            std::abs(lhs-rhs) <= tolerance,
            "Calibration transpose identity failed");

        std::cout
            << "PASS: zero-bias regression\n"
            << "PASS: pressure-only calibration shifts\n"
            << "PASS: calibration transpose identity\n";

        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
