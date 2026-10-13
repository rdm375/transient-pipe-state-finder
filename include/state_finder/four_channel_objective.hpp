#pragma once

#include "state_finder/observations.hpp"

#include <cstddef>
#include <vector>

namespace state_finder {

// Predicted, physically consistent boundary histories.
//
// The inlet pressure and outlet mass flow are adjustable
// boundary inputs. The remaining histories are produced
// by the transient forward model.
struct FourChannelPrediction {
    BoundaryMeasurements boundary;
};

struct FourChannelObjectiveResult {
    double total = 0.0;

    double inlet_pressure = 0.0;
    double inlet_flow = 0.0;
    double outlet_pressure = 0.0;
    double outlet_flow = 0.0;

    // Row-major time ordering:
    //   [pin, min, pout, mout] at each sample.
    //
    // Residuals are normalized by measurement scales.
    std::vector<double> residuals;
};

// Evaluates the four-channel measurement contribution.
// Does not include state or boundary-history regularization.
FourChannelObjectiveResult evaluate_four_channel_objective(
    const BoundaryObservationRecord& observations,
    const FourChannelPrediction& prediction);

// Gradient of the measurement objective with respect
// to the predicted boundary histories.
//
// This is the observation-space gradient. The gradients
// with respect to the initial state and adjustable boundary
// inputs require the forward-model Jacobian or adjoint.
BoundaryMeasurements four_channel_observation_gradient(
    const BoundaryObservationRecord& observations,
    const FourChannelPrediction& prediction);

} // namespace state_finder
