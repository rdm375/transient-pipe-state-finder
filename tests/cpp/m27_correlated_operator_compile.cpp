#include "state_finder/correlated_joint_gauss_newton.hpp"

#include <type_traits>

int main()
{
    using namespace state_finder;

    static_assert(
        std::is_same_v<
            decltype(
                make_correlated_joint_gauss_newton_factory(
                    ForwardConfiguration{},
                    BoundaryObservationRecord{},
                    JointInverseVariables{},
                    TemporalMeasurementCovariance(
                        1, 10.0, 120.0)
                )
            ),
            GaussNewtonOperatorFactory
        >
    );

    return 0;
}
