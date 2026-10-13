#include "state_finder/joint_tangent_linear.hpp"
#include "state_finder/four_channel_objective.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace state_finder;

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

double relative_error(double a, double b)
{
    return std::abs(a-b)/std::max({1.0,std::abs(a),std::abs(b)});
}
} // namespace

int main()
{
    try {
        constexpr int n=6, steps=5;
        const double pi=std::acos(-1.0);
        ForwardConfiguration config{
            n, steps,
            pipe_sim::TransientParameters{
                1000.0,15.0,0.65,1.0,pi/4.0,
                288.15,0.9,500.0,1.1e-5,4.5e-5},
            pipe_sim::DemandRamp{8.0e6,100.0,100.0,1.0},
            pipe_sim::NewtonOptions{1.0e-9,1.0e-12,30,false}
        };
        JointInverseVariables baseline;
        baseline.initial.pressure.assign(n+1,8.0e6);
        baseline.initial.flow.assign(n,100.0);
        baseline.initial.inlet_flow=100.0;
        baseline.boundary.inlet_pressure.assign(steps+1,8.0e6);
        baseline.boundary.outlet_flow.assign(steps+1,100.0);
        baseline.initial.pressure[2]+=1400.0;
        baseline.initial.flow[1]+=0.2;
        baseline.initial.inlet_flow+=0.1;
        baseline.boundary.inlet_pressure[0]+=400.0;
        baseline.boundary.inlet_pressure[2]-=500.0;
        baseline.boundary.outlet_flow[1]+=0.2;
        baseline.boundary.outlet_flow[4]-=0.1;
        baseline.initial.pressure[0]=baseline.boundary.inlet_pressure[0];

        JointVariableScales scales;
        const auto dimension=joint_variable_dimension(n,steps);
        std::vector<double> direction(dimension);
        for (std::size_t i=0;i<dimension;++i)
            direction[i]=0.17*std::sin(0.43*(i+1));
        const auto trajectory=record_joint_trajectory(config,baseline);
        const auto tangent=evaluate_joint_tangent_linear(
            config,trajectory,direction,scales);
        const auto tangent_direct=evaluate_joint_tangent_linear(
            config,baseline,direction,scales);
        require(tangent.final_state.size()==static_cast<std::size_t>(2*n+2),
                "Final tangent state size mismatch");
        for (std::size_t i=0;i<tangent.final_state.size();++i)
            require(relative_error(tangent.final_state[i],tangent_direct.final_state[i])<1e-13,
                    "Shared/direct trajectory tangent mismatch");
        require(relative_error(tangent.final_state[0],tangent.response.inlet_pressure.back())<1e-10,
                "Final inlet-pressure compatibility mismatch");

        BoundaryObservationRecord observations;
        observations.measured=trajectory.predicted;
        double worst=0.0;
        constexpr double h=1e-3;
        for (bool include_initial : {true,false}) {
            observations.include_initial=include_initial;
            const auto jv=normalized_joint_tangent_residuals(tangent,observations);
            auto plus=direction, minus=direction;
            for (std::size_t i=0;i<dimension;++i) {
                plus[i]*=h;
                minus[i]*=-h;
            }
            const auto vp=unpack_joint_variables(plus,baseline,scales);
            const auto vm=unpack_joint_variables(minus,baseline,scales);
            const auto tp=record_joint_trajectory(config,vp);
            const auto tm=record_joint_trajectory(config,vm);
            const auto rp=evaluate_four_channel_objective(
                observations,FourChannelPrediction{tp.predicted}).residuals;
            const auto rm=evaluate_four_channel_objective(
                observations,FourChannelPrediction{tm.predicted}).residuals;
            require(jv.size()==rp.size() && jv.size()==rm.size(),
                    "Normalized residual ordering mismatch");
            for (std::size_t i=0;i<jv.size();++i) {
                const double fd=(rp[i]-rm[i])/(2*h);
                worst=std::max(worst,relative_error(jv[i],fd));
            }
        }
        std::cout << "joint_direction_dimension=" << dimension << '\n'
                  << "max_relative_fd_error=" << worst << '\n'
                  << "max_linear_backward_error="
                  << tangent.max_linear_backward_error << '\n';
        require(worst<2e-4,"Joint tangent vs finite difference mismatch");
        std::cout << "PASS joint tangent-linear operator\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
