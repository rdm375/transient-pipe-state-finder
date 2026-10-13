#include "state_finder/boundary_derivatives.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace state_finder {
namespace {

void validate(int n, double theta, std::size_t size)
{
    if (n < 2 ||
        size != static_cast<std::size_t>(2*n+2) ||
        !std::isfinite(theta) ||
        theta < 0.0 || theta > 1.0)
        throw std::invalid_argument(
            "Invalid boundary derivative dimensions or theta");
}

} // namespace

void boundary_residual_jvp(
    int n,
    double theta,
    double inlet_pressure_new_direction,
    double outlet_flow_old_direction,
    double outlet_flow_new_direction,
    std::span<double> result)
{
    validate(n, theta, result.size());

    std::fill(result.begin(), result.end(), 0.0);

    result[0] = -inlet_pressure_new_direction;

    result[2*n+1] =
        (1.0-theta)*outlet_flow_old_direction
        + theta*outlet_flow_new_direction;
}

void boundary_residual_vjp(
    int n,
    double theta,
    std::span<const double> lambda,
    double& inlet_pressure_new_gradient,
    double& outlet_flow_old_gradient,
    double& outlet_flow_new_gradient)
{
    validate(n, theta, lambda.size());

    inlet_pressure_new_gradient -= lambda[0];

    outlet_flow_old_gradient +=
        (1.0-theta)*lambda[2*n+1];

    outlet_flow_new_gradient +=
        theta*lambda[2*n+1];
}

} // namespace state_finder
