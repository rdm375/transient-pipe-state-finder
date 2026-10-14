#include "state_finder/joint_tangent_linear.hpp"

#include "state_finder/boundary_derivatives.hpp"
#include "state_finder/old_state_jvp.hpp"
#include "state_finder/transpose_solver.hpp"

#include "pipe_sim/transient.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace state_finder {
namespace {

void positive_scale(double value)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument("Invalid joint tangent scale");
}

} // namespace

JointTangentResult evaluate_joint_tangent_linear(
    const ForwardConfiguration& configuration,
    const JointTrajectory& trajectory,
    std::span<const double> scaled_direction,
    JointVariableScales scales)
{
    const int n = configuration.n;
    const int steps = configuration.steps;
    if (n < 2 || steps < 0)
        throw std::invalid_argument("Invalid joint tangent dimensions");
    const auto count = static_cast<std::size_t>(steps+1);
    const auto width = static_cast<std::size_t>(2*n+2);
    const auto initial_count = static_cast<std::size_t>(2*n+1);
    if (trajectory.n != n || trajectory.steps != steps ||
        trajectory.states.size() != count*width ||
        scaled_direction.size() != joint_variable_dimension(n, steps))
        throw std::invalid_argument("Joint tangent trajectory/direction mismatch");

    positive_scale(scales.initial.pressure);
    positive_scale(scales.initial.flow);
    positive_scale(scales.inlet_pressure);
    positive_scale(scales.outlet_flow);
    for (double value : scaled_direction)
        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite joint tangent direction");

    // Initial p0 is the first adjustable inlet-pressure sample.
    std::vector<double> delta(width, 0.0);
    delta[0] = scales.inlet_pressure*scaled_direction[initial_count];
    delta[1] = scales.initial.flow*scaled_direction[2*n];
    for (int i=0; i<n; ++i) {
        delta[2*i+2] = scales.initial.flow*scaled_direction[n+i];
        delta[2*i+3] = scales.initial.pressure*scaled_direction[i];
    }

    JointTangentResult result;

    // Record the initial tangent state.
    result.states.reserve(count*width);
    result.states.insert(
        result.states.end(),
        delta.begin(),
        delta.end());

    auto& y = result.response;
    y.inlet_pressure.resize(count);
    y.inlet_flow.resize(count);
    y.outlet_pressure.resize(count);
    y.outlet_flow.resize(count);
    for (std::size_t k=0; k<count; ++k) {
        y.inlet_pressure[k] = scales.inlet_pressure*scaled_direction[initial_count+k];
        y.outlet_flow[k] = scales.outlet_flow*scaled_direction[initial_count+count+k];
    }
    y.inlet_flow[0] = delta[1];
    y.outlet_pressure[0] = delta[width-1];

    constexpr int kl=2, ku=1, ldab=6;
    std::vector<double> jacobian(static_cast<std::size_t>(ldab)*width);
    std::vector<double> rhs(width), boundary_term(width);
    for (int k=1; k<=steps; ++k) {
        old_state_jvp(n, trajectory.state(k-1),
                      configuration.parameters, delta, rhs);
        boundary_residual_jvp(n, configuration.parameters.theta,
                              y.inlet_pressure[k],
                              y.outlet_flow[k-1],
                              y.outlet_flow[k], boundary_term);
        for (std::size_t j=0; j<width; ++j)
            rhs[j] = -(rhs[j]+boundary_term[j]);
        pipe_sim::assemble_jacobian_banded(n, trajectory.state(k),
                                           configuration.parameters,
                                           jacobian, ldab);
        const auto solved=solve_banded(static_cast<int>(width), jacobian,
                                       ldab, kl, ku, rhs);
        result.max_linear_backward_error = std::max(
            result.max_linear_backward_error, solved.relative_backward_error);
        delta=solved.solution;
        for (double value : delta)
            if (!std::isfinite(value))
                throw std::runtime_error("Nonfinite joint tangent state");

        // Preserve the complete tangent trajectory.
        result.states.insert(
            result.states.end(),
            delta.begin(),
            delta.end());

        y.inlet_flow[k] = delta[1];
        y.outlet_pressure[k] = delta[width-1];
        const double tol=1e-8*std::max(1.0, std::abs(y.inlet_pressure[k]));
        if (std::abs(delta[0]-y.inlet_pressure[k]) > tol)
            throw std::runtime_error("Joint tangent inlet-pressure drift");
    }
    result.final_state=std::move(delta);
    return result;
}

JointTangentResult evaluate_joint_tangent_linear(
    const ForwardConfiguration& configuration,
    const JointInverseVariables& variables,
    std::span<const double> scaled_direction,
    JointVariableScales scales)
{
    const auto trajectory=record_joint_trajectory(configuration, variables);
    return evaluate_joint_tangent_linear(configuration, trajectory,
                                         scaled_direction, scales);
}

std::vector<double> normalized_joint_tangent_residuals(
    const JointTangentResult& tangent,
    const BoundaryObservationRecord& observations)
{
    const auto count=observations.measured.inlet_pressure.size();
    validate_boundary_observations(observations, count);
    const auto& y=tangent.response;
    if (y.inlet_pressure.size()!=count || y.inlet_flow.size()!=count ||
        y.outlet_pressure.size()!=count || y.outlet_flow.size()!=count)
        throw std::invalid_argument("Joint tangent observation dimension mismatch");
    const auto& s=observations.scales;
    const auto first=observations.include_initial ? std::size_t{0} : std::size_t{1};
    std::vector<double> result;
    result.reserve(4*(count-first));
    for (auto k=first; k<count; ++k) {
        result.push_back(y.inlet_pressure[k]/s.inlet_pressure);
        result.push_back(y.inlet_flow[k]/s.inlet_flow);
        result.push_back(y.outlet_pressure[k]/s.outlet_pressure);
        result.push_back(y.outlet_flow[k]/s.outlet_flow);
    }
    for (double value : result)
        if (!std::isfinite(value))
            throw std::runtime_error("Nonfinite normalized joint tangent");
    return result;
}

} // namespace state_finder
