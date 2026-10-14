// M2.4-F2D nonlinear oracle: persistent stdin/stdout protocol.
// Each request is 41 comma-separated scaled initial-state coordinates.
// Response: objective pressure, flow, inventory (signed at F2B indices),
// squared normalized measurement norm, minimum pressure, conservation defect,
// followed by 41 derivatives for each of the four quantities.
#include "state_finder/joint_trajectory.hpp"
#include "state_finder/joint_tangent_linear.hpp"
#include "state_finder/joint_scaled_problem.hpp"
#include "state_finder/steady_state_initializer.hpp"
#include "pipe_sim/integrate.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace state_finder;
namespace {
constexpr int n=20, steps=60, nx=41;
constexpr double dt=10.;
std::vector<double> pressures(std::span<const double> s) {
    std::vector<double> p(n+1); p[0]=s[0];
    for(int i=1;i<=n;++i) p[i]=s[2*i+1];
    return p;
}
double inventory(std::span<const double> s,const ForwardConfiguration& c) {
    return pipe_sim::linepack_cz(n,pressures(s),c.parameters);
}
std::array<double,4> residual(const BoundaryMeasurements& a,
                              const BoundaryMeasurements& b,int k) {
    return {(b.inlet_pressure[k]-a.inlet_pressure[k])/1e4,
            (b.inlet_flow[k]-a.inlet_flow[k]),
            (b.outlet_pressure[k]-a.outlet_pressure[k])/1e4,
            (b.outlet_flow[k]-a.outlet_flow[k])};
}
std::array<double,4> tangent_residual(const BoundaryMeasurements& b,int k) {
    return {b.inlet_pressure[k]/1e4,b.inlet_flow[k],
            b.outlet_pressure[k]/1e4,b.outlet_flow[k]};
}
std::array<double,3> objectives(std::span<const double> a,
                                std::span<const double> b,
                                const ForwardConfiguration& c) {
    return {b[2*n+1]-a[2*n+1], // pressure node 20
            b[2]-a[2],         // flow face 1
            inventory(b,c)-inventory(a,c)};
}
std::array<double,3> objective_derivatives(std::span<const double> t,
                                           const ForwardConfiguration& c) {
    return {t[2*n+1],t[2],inventory(t,c)};
}
}
int main() {
 try {
    const double diameter=.6, flow=100., pin=8e6;
    const double area=std::acos(-1.)*diameter*diameter/4.;
    ForwardConfiguration c{n,steps,
        pipe_sim::TransientParameters{50000./n,dt,.65,diameter,area,288.15,.9,500.,1.1e-5,4.5e-5},
        pipe_sim::DemandRamp{pin,flow,flow,1.},
        pipe_sim::NewtonOptions{1e-9,1e-12,30,false}};
    JointInverseVariables truth;
    truth.initial=initialize_steady_state(c,flow);
    truth.boundary.inlet_pressure.assign(steps+1,pin);
    truth.boundary.outlet_flow.assign(steps+1,flow);
    for(int k=0;k<=steps;++k) {
        const double f=std::clamp((k*dt-120.)/120.,0.,1.);
        truth.boundary.outlet_flow[k]=flow+10.*f*f*(3.-2.*f);
    }
    const auto reference=record_joint_trajectory(c,truth);
    const auto dimension=joint_variable_dimension(n,steps);
    const JointVariableScales scales{};
    std::cout<<"READY 41 3\n"<<std::flush;
    std::string line;
    while(std::getline(std::cin,line)) {
      if(line=="QUIT") break;
      try {
        std::istringstream in(line);
        std::vector<double> v(dimension,0.);
        for(int j=0;j<nx;++j) {
            std::string field;
            if(!std::getline(in,field,',')) throw std::runtime_error("Missing coordinate");
            v[j]=std::stod(field);
            if(!std::isfinite(v[j])) throw std::runtime_error("Nonfinite coordinate");
        }
        std::string extra;
        if(std::getline(in,extra,',')) throw std::runtime_error("Extra coordinate");
        auto perturbed=unpack_joint_variables(v,truth,scales);
        perturbed.boundary=truth.boundary;
        const auto trajectory=record_joint_trajectory(c,perturbed);
        const auto values=objectives(reference.state(steps),trajectory.state(steps),c);
        std::array<std::array<double,4>,steps+1> r{};
        double norm2=0.,min_pressure=std::numeric_limits<double>::infinity();
        double max_defect=0.,previous_inventory=0.;
        for(int k=0;k<=steps;++k) {
            r[k]=residual(reference.predicted,trajectory.predicted,k);
            for(double x:r[k]) norm2+=x*x;
            for(double p:pressures(trajectory.state(k))) min_pressure=std::min(min_pressure,p);
            const double delta=inventory(trajectory.state(k),c)-inventory(reference.state(k),c);
            if(k) {
                const double prev=trajectory.state(k-1)[1]-reference.state(k-1)[1];
                const double now=trajectory.state(k)[1]-reference.state(k)[1];
                const double defect=delta-previous_inventory-dt*((1.-c.parameters.theta)*prev+c.parameters.theta*now);
                max_defect=std::max(max_defect,std::abs(defect));
            }
            previous_inventory=delta;
        }
        std::array<std::vector<double>,4> grads;
        for(auto& g:grads) g.resize(nx);
        for(int j=0;j<nx;++j) {
            std::vector<double> e(dimension,0.); e[j]=1.;
            const auto t=evaluate_joint_tangent_linear(c,trajectory,e,scales);
            const auto dg=objective_derivatives(t.state(steps,n),c);
            for(int q=0;q<3;++q) grads[q][j]=dg[q];
            double g=0.;
            for(int k=0;k<=steps;++k) {
                const auto dr=tangent_residual(t.response,k);
                for(int h=0;h<4;++h) g+=2.*r[k][h]*dr[h];
            }
            grads[3][j]=g;
        }
        std::cout<<std::setprecision(17)<<"OK";
        for(double x:values) std::cout<<','<<x;
        std::cout<<','<<norm2<<','<<min_pressure<<','<<max_defect;
        for(const auto& g:grads) for(double x:g) std::cout<<','<<x;
        std::cout<<'\n'<<std::flush;
      } catch(const std::exception& e) {
        std::cout<<"ERROR,"<<e.what()<<'\n'<<std::flush;
      }
    }
    return 0;
 } catch(const std::exception& e) {
    std::cerr<<"F2D oracle fatal: "<<e.what()<<'\n';return 1;
 }
}
