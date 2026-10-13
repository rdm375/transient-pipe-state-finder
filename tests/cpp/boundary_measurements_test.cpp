#include "state_finder/observations.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
void require_throws(
    Function function,
    const char* message)
{
    bool threw = false;

    try {
        function();
    }
    catch (const std::invalid_argument&) {
        threw = true;
    }

    require(threw, message);
}

} // namespace

int main()
{
    try {
        using namespace state_finder;

        BoundaryObservationRecord record;

        record.measured.inlet_pressure =
            {8.0e6, 8.0e6, 8.0e6};

        record.measured.inlet_flow =
            {100.0, 100.2, 100.4};

        record.measured.outlet_pressure =
            {7.9e6, 7.89e6, 7.88e6};

        record.measured.outlet_flow =
            {100.0, 101.0, 102.0};

        record.include_initial = true;

        validate_boundary_observations(record, 3);

        const auto legacy =
            make_current_observations(record, 3);

        require(
            legacy.inlet_flow ==
                record.measured.inlet_flow,
            "Inlet flow migration failed");

        require(
            legacy.outlet_pressure ==
                record.measured.outlet_pressure,
            "Outlet pressure migration failed");

        require(
            legacy.sigma_flow ==
                record.scales.inlet_flow,
            "Flow scale migration failed");

        require(
            legacy.sigma_pressure ==
                record.scales.outlet_pressure,
            "Pressure scale migration failed");

        require(
            legacy.include_initial,
            "Initial observation flag lost");

        // Detect incomplete four-channel records.
        {
            auto bad = record;
            bad.measured.outlet_flow.pop_back();

            require_throws(
                [&] {
                    validate_boundary_observations(
                        bad, 3);
                },
                "Missing outlet-flow sample accepted");
        }

        // Detect nonfinite measurements.
        {
            auto bad = record;
            bad.measured.inlet_pressure[1] =
                std::nan("");

            require_throws(
                [&] {
                    validate_boundary_observations(
                        bad, 3);
                },
                "Nonfinite inlet pressure accepted");
        }

        // Detect invalid measurement scales.
        {
            auto bad = record;
            bad.scales.outlet_flow = 0.0;

            require_throws(
                [&] {
                    validate_boundary_observations(
                        bad, 3);
                },
                "Zero measurement scale accepted");
        }

        // Reject roles unsupported by the present
        // forward-model boundary formulation.
        {
            auto bad = record;

            bad.roles.inlet_pressure =
                MeasurementRole::ObjectiveResidual;

            require_throws(
                [&] {
                    make_current_observations(
                        bad, 3);
                },
                "Unsupported measurement roles accepted");
        }

        std::cout
            << "PASS four-channel boundary measurements\n"
            << "PASS compatibility conversion\n"
            << "PASS measurement validation\n"
            << "PASS role validation\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr
            << "FAIL: " << e.what() << '\n';

        return EXIT_FAILURE;
    }
}
