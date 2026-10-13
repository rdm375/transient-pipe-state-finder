#include "state_finder/four_channel_objective.hpp"

#include <cmath>
#include <stdexcept>

namespace state_finder {
namespace {

void validate_prediction(
    const BoundaryMeasurements& predicted,
    std::size_t count)
{
    const auto validate = [count](
        const std::vector<double>& values)
    {
        if (values.size() != count)
            throw std::invalid_argument(
                "Four-channel prediction dimension mismatch");

        for (double value : values) {
            if (!std::isfinite(value))
                throw std::invalid_argument(
                    "Nonfinite four-channel prediction");
        }
    };

    validate(predicted.inlet_pressure);
    validate(predicted.inlet_flow);
    validate(predicted.outlet_pressure);
    validate(predicted.outlet_flow);
}

} // namespace

FourChannelObjectiveResult evaluate_four_channel_objective(
    const BoundaryObservationRecord& observations,
    const FourChannelPrediction& prediction)
{
    const auto count =
        observations.measured.inlet_pressure.size();

    validate_boundary_observations(observations, count);
    validate_prediction(prediction.boundary, count);

    const auto& m = observations.measured;
    const auto& p = prediction.boundary;
    const auto& s = observations.scales;

    FourChannelObjectiveResult result;

    const std::size_t first =
        observations.include_initial ? 0 : 1;

    result.residuals.reserve(4 * (count - first));

    const auto add = [&](double predicted,
                         double measured,
                         double sigma,
                         double& component)
    {
        const double residual =
            (predicted - measured) / sigma;

        if (!std::isfinite(residual))
            throw std::runtime_error(
                "Nonfinite four-channel residual");

        const double contribution =
            0.5 * residual * residual;

        if (!std::isfinite(contribution))
            throw std::runtime_error(
                "Nonfinite four-channel contribution");

        component += contribution;
        result.residuals.push_back(residual);
    };

    for (std::size_t k = first; k < count; ++k) {
        add(
            p.inlet_pressure[k],
            m.inlet_pressure[k],
            s.inlet_pressure,
            result.inlet_pressure);

        add(
            p.inlet_flow[k],
            m.inlet_flow[k],
            s.inlet_flow,
            result.inlet_flow);

        add(
            p.outlet_pressure[k],
            m.outlet_pressure[k],
            s.outlet_pressure,
            result.outlet_pressure);

        add(
            p.outlet_flow[k],
            m.outlet_flow[k],
            s.outlet_flow,
            result.outlet_flow);
    }

    result.total =
        result.inlet_pressure +
        result.inlet_flow +
        result.outlet_pressure +
        result.outlet_flow;

    if (!std::isfinite(result.total))
        throw std::runtime_error(
            "Nonfinite four-channel objective");

    return result;
}

BoundaryMeasurements four_channel_observation_gradient(
    const BoundaryObservationRecord& observations,
    const FourChannelPrediction& prediction)
{
    const auto count =
        observations.measured.inlet_pressure.size();

    validate_boundary_observations(observations, count);
    validate_prediction(prediction.boundary, count);

    const auto& m = observations.measured;
    const auto& p = prediction.boundary;
    const auto& s = observations.scales;

    BoundaryMeasurements gradient;

    gradient.inlet_pressure.assign(count, 0.0);
    gradient.inlet_flow.assign(count, 0.0);
    gradient.outlet_pressure.assign(count, 0.0);
    gradient.outlet_flow.assign(count, 0.0);

    const std::size_t first =
        observations.include_initial ? 0 : 1;

    for (std::size_t k = first; k < count; ++k) {
        gradient.inlet_pressure[k] =
            (p.inlet_pressure[k] - m.inlet_pressure[k]) /
            (s.inlet_pressure * s.inlet_pressure);

        gradient.inlet_flow[k] =
            (p.inlet_flow[k] - m.inlet_flow[k]) /
            (s.inlet_flow * s.inlet_flow);

        gradient.outlet_pressure[k] =
            (p.outlet_pressure[k] - m.outlet_pressure[k]) /
            (s.outlet_pressure * s.outlet_pressure);

        gradient.outlet_flow[k] =
            (p.outlet_flow[k] - m.outlet_flow[k]) /
            (s.outlet_flow * s.outlet_flow);
    }

    return gradient;
}

} // namespace state_finder
