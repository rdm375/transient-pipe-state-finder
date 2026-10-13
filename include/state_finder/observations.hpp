#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace state_finder {

// Roles describe how a measurement is used in the
// current inverse formulation.
//
// PrescribedInput:
//     Drives the forward problem.
//
// ObjectiveResidual:
//     Contributes an independent observation residual.
//
// RecordedOnly:
//     Retained for future formulations.
//
// The present solver supports prescribed inlet pressure
// and prescribed outlet mass flow.
enum class MeasurementRole {
    PrescribedInput,
    ObjectiveResidual,
    RecordedOnly
};

struct BoundaryMeasurements {
    // All histories include t = 0.
    std::vector<double> inlet_pressure;
    std::vector<double> inlet_flow;
    std::vector<double> outlet_pressure;
    std::vector<double> outlet_flow;
};

struct BoundaryMeasurementScales {
    double inlet_pressure = 1.0e4;
    double inlet_flow = 1.0;
    double outlet_pressure = 1.0e4;
    double outlet_flow = 1.0;
};

struct BoundaryMeasurementRoles {
    MeasurementRole inlet_pressure =
        MeasurementRole::PrescribedInput;

    MeasurementRole inlet_flow =
        MeasurementRole::ObjectiveResidual;

    MeasurementRole outlet_pressure =
        MeasurementRole::ObjectiveResidual;

    MeasurementRole outlet_flow =
        MeasurementRole::PrescribedInput;
};

struct BoundaryObservationRecord {
    BoundaryMeasurements measured;
    BoundaryMeasurementScales scales;
    BoundaryMeasurementRoles roles;

    bool include_initial = false;
};

inline void validate_boundary_observations(
    const BoundaryObservationRecord& record,
    std::size_t expected_samples)
{
    const auto& m = record.measured;
    const auto& s = record.scales;

    const auto validate_channel =
        [expected_samples](
            const std::vector<double>& values,
            double scale)
        {
            if (values.size() != expected_samples)
                throw std::invalid_argument(
                    "Boundary measurement dimension mismatch");

            if (!std::isfinite(scale) || scale <= 0.0)
                throw std::invalid_argument(
                    "Invalid boundary measurement scale");

            for (double value : values) {
                if (!std::isfinite(value))
                    throw std::invalid_argument(
                        "Nonfinite boundary measurement");
            }
        };

    validate_channel(
        m.inlet_pressure, s.inlet_pressure);

    validate_channel(
        m.inlet_flow, s.inlet_flow);

    validate_channel(
        m.outlet_pressure, s.outlet_pressure);

    validate_channel(
        m.outlet_flow, s.outlet_flow);
}

inline void validate_current_boundary_roles(
    const BoundaryMeasurementRoles& roles)
{
    if (roles.inlet_pressure !=
            MeasurementRole::PrescribedInput ||
        roles.inlet_flow !=
            MeasurementRole::ObjectiveResidual ||
        roles.outlet_pressure !=
            MeasurementRole::ObjectiveResidual ||
        roles.outlet_flow !=
            MeasurementRole::PrescribedInput)
    {
        throw std::invalid_argument(
            "Unsupported boundary measurement roles "
            "for current inverse formulation");
    }
}

// Compatibility interface used by the verified
// objective, discrete adjoint, and existing tests.
//
// This is intentionally retained during migration.
struct Observations {
    std::vector<double> inlet_flow;
    std::vector<double> outlet_pressure;

    double sigma_flow = 1.0;
    double sigma_pressure = 1.0e4;

    bool include_initial = false;
};

// Convert a validated four-channel record into the
// existing two-residual objective representation.
//
// Prescribed-input channels are not added to the
// objective because their residuals would be identically
// zero under the present exact-BC formulation.
inline Observations make_current_observations(
    const BoundaryObservationRecord& record,
    std::size_t expected_samples)
{
    validate_boundary_observations(
        record, expected_samples);

    validate_current_boundary_roles(record.roles);

    Observations observations;

    observations.inlet_flow =
        record.measured.inlet_flow;

    observations.outlet_pressure =
        record.measured.outlet_pressure;

    observations.sigma_flow =
        record.scales.inlet_flow;

    observations.sigma_pressure =
        record.scales.outlet_pressure;

    observations.include_initial =
        record.include_initial;

    return observations;
}

} // namespace state_finder
