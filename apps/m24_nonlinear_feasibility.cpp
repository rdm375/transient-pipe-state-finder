#include "state_finder/joint_trajectory.hpp"
#include "state_finder/joint_scaled_problem.hpp"
#include "state_finder/steady_state_initializer.hpp"
#include "pipe_sim/integrate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace state_finder;
namespace {
constexpr int n=20, steps=60, nx=2*n+1;
constexpr double dt=10.0, epsilon=1.0;
using Key=std::pair<double,std::string>;
struct Direction {
    std::array<double,nx> values{};
    std::array<bool,nx> present{};
    int index=-1;
};
std::map<Key,Direction> read_directions(const std::filesystem::path& path) {
    std::ifstream in(path);
    if(!in) throw std::runtime_error("Cannot open "+path.string());
    std::string line;
    if(!std::getline(in,line)) throw std::runtime_error("Empty direction CSV");
    if(line!="measurement_bound,quantity,time_s,worst_index,initial_coordinate,scaled_perturbation")
        throw std::runtime_error("Unexpected direction CSV header: "+line);
    std::map<Key,Direction> result;
    while(std::getline(in,line)) {
        if(line.empty()) continue;
        std::istringstream ss(line);
        std::array<std::string,6> c;
        for(auto& item:c) if(!std::getline(ss,item,',')) throw std::runtime_error("Short CSV row");
        std::string extra;
        if(std::getline(ss,extra,',')) throw std::runtime_error("Long CSV row");
        const double eps=std::stod(c[0]), time=std::stod(c[2]);
        if(std::abs(eps-epsilon)>1e-12 || std::abs(time-600.0)>1e-9) continue;
        const auto& quantity=c[1];
        if(quantity!="pressure" && quantity!="flow" && quantity!="inventory") continue;
        const int index=std::stoi(c[3]), coordinate=std::stoi(c[4]);
        const double value=std::stod(c[5]);
        if(coordinate<0 || coordinate>=nx || !std::isfinite(value))
            throw std::runtime_error("Invalid direction coordinate");
        auto& d=result[{eps,quantity}];
        if(d.present[coordinate]) throw std::runtime_error("Duplicate direction coordinate");
        if(d.index>=0 && d.index!=index) throw std::runtime_error("Inconsistent target index");
        d.index=index; d.values[coordinate]=value; d.present[coordinate]=true;
    }
    if(result.size()!=3) throw std::runtime_error("Expected three epsilon=1 directions");
    for(const auto& [key,d]:result)
        if(!std::all_of(d.present.begin(),d.present.end(),[](bool x){return x;}))
            throw std::runtime_error("Incomplete direction: "+key.second);
    return result;
}
std::vector<double> pressures(std::span<const double> state) {
    std::vector<double> p(n+1); p[0]=state[0];
    for(int i=1;i<=n;++i) p[i]=state[2*i+1];
    return p;
}
double linepack(std::span<const double> state,const ForwardConfiguration& config) {
    const auto p=pressures(state);
    return pipe_sim::linepack_cz(n,p,config.parameters);
}
double norm_measurements(const JointTrajectory& a,const JointTrajectory& b) {
    double sum=0.0;
    for(int k=0;k<=steps;++k) {
        const std::array<double,4> r{
            (b.predicted.inlet_pressure[k]-a.predicted.inlet_pressure[k])/1e4,
            (b.predicted.inlet_flow[k]-a.predicted.inlet_flow[k])/1.0,
            (b.predicted.outlet_pressure[k]-a.predicted.outlet_pressure[k])/1e4,
            (b.predicted.outlet_flow[k]-a.predicted.outlet_flow[k])/1.0
        };
        for(double x:r) sum+=x*x;
    }
    return std::sqrt(sum);
}
double targeted(std::span<const double> a,std::span<const double> b,
                const ForwardConfiguration& config,const std::string& quantity,int index) {
    if(quantity=="pressure") {
        if(index<0 || index>n) throw std::runtime_error("Invalid pressure index");
        const int j=index==0?0:2*index+1;
        return b[j]-a[j];
    }
    if(quantity=="flow") {
        if(index<0 || index>n+1) throw std::runtime_error("Invalid flow index");
        if(index==n+1) return 0.0; // Prescribed outlet boundary history is fixed.
        const int j=index==0?1:2*index;
        return b[j]-a[j];
    }
    if(quantity=="inventory") {
        if(index!=0) throw std::runtime_error("Invalid inventory index");
        return linepack(b,config)-linepack(a,config);
    }
    throw std::runtime_error("Unknown quantity");
}
struct Diagnostics {
    double min_pressure=std::numeric_limits<double>::infinity();
    double max_conservation_defect=0.0;
    double max_targeted_abs=0.0;
    double terminal_targeted=0.0;
};
Diagnostics diagnostics(const JointTrajectory& a,const JointTrajectory& b,
                        const ForwardConfiguration& config,const std::string& quantity,int index) {
    Diagnostics out;
    double previous_inventory_delta=0.0;
    for(int k=0;k<=steps;++k) {
        const auto ref=a.state(k), state=b.state(k);
        const double deviation=targeted(ref,state,config,quantity,index);
        out.max_targeted_abs=std::max(out.max_targeted_abs,std::abs(deviation));
        if(k==steps) out.terminal_targeted=deviation;
        for(double p:pressures(state)) out.min_pressure=std::min(out.min_pressure,p);
        const double inventory_delta=linepack(state,config)-linepack(ref,config);
        if(k>0) {
            const double previous_inlet_delta=b.state(k-1)[1]-a.state(k-1)[1];
            const double current_inlet_delta=state[1]-ref[1];
            const double weighted=(1.0-config.parameters.theta)*previous_inlet_delta
                                  +config.parameters.theta*current_inlet_delta;
            const double defect=inventory_delta-previous_inventory_delta-dt*weighted;
            out.max_conservation_defect=std::max(out.max_conservation_defect,std::abs(defect));
        }
        previous_inventory_delta=inventory_delta;
    }
    return out;
}
}
int main() {
    try {
        constexpr double length=50000.0, diameter=0.6, inlet_pressure=8e6, flow=100.0;
        const double area=std::acos(-1.0)*diameter*diameter/4.0;
        ForwardConfiguration config{
            n,steps,
            pipe_sim::TransientParameters{length/n,dt,0.65,diameter,area,288.15,0.9,500.0,1.1e-5,4.5e-5},
            pipe_sim::DemandRamp{inlet_pressure,flow,flow,1.0},
            pipe_sim::NewtonOptions{1e-9,1e-12,30,false}
        };
        JointInverseVariables truth;
        truth.initial=initialize_steady_state(config,flow);
        truth.boundary.inlet_pressure.assign(steps+1,inlet_pressure);
        truth.boundary.outlet_flow.assign(steps+1,flow);
        for(int k=0;k<=steps;++k) {
            const double f=std::clamp((k*dt-120.0)/120.0,0.0,1.0);
            truth.boundary.outlet_flow[k]=flow+10.0*f*f*(3.0-2.0*f);
        }
        const auto reference=record_joint_trajectory(config,truth);
        const std::filesystem::path dir="benchmarks/results-m24";
        const auto directions=read_directions(dir/"adversarial_initial_states.csv");
        std::filesystem::create_directories(dir);
        std::ofstream out(dir/"adversarial_feasible_amplitudes.csv");
        if(!out) throw std::runtime_error("Cannot create feasibility output");
        out<<std::setprecision(17);
        out<<"quantity,measurement_bound,worst_index,initial_norm,alpha_feasible,measurement_norm,full_amplitude_measurement_norm,monotonicity_pass,bisection_iterations,terminal_targeted_signed,terminal_targeted_abs,trajectory_targeted_linf,min_pressure_Pa,max_conservation_defect_kg\n";
        const auto dimension=joint_variable_dimension(n,steps);
        const JointVariableScales scales{};
        for(const auto& [key,d]:directions) {
            const auto& quantity=key.second;
            std::vector<double> direction(dimension,0.0);
            double norm_sq=0.0;
            for(int j=0;j<nx;++j) { direction[j]=d.values[j]; norm_sq+=d.values[j]*d.values[j]; }
            const double initial_norm=std::sqrt(norm_sq);
            if(initial_norm>1.0+1e-6) throw std::runtime_error("Initial-state uncertainty radius violated");
            auto simulate=[&](double alpha) {
                auto scaled=direction;
                for(double& v:scaled) v*=alpha;
                auto perturbed=unpack_joint_variables(scaled,truth,scales);
                perturbed.boundary=truth.boundary;
                return record_joint_trajectory(config,perturbed);
            };
            auto measure=[&](double alpha) {
                const auto nonlinear=simulate(alpha);
                const double value=norm_measurements(reference,nonlinear);
                if(!std::isfinite(value)) throw std::runtime_error("Nonfinite measurement norm");
                return value;
            };
            constexpr int samples=100;
            double previous=measure(0.0);
            bool monotone=true;
            for(int i=1;i<=samples;++i) {
                const double value=measure(static_cast<double>(i)/samples);
                if(value+1e-8<previous) monotone=false;
                previous=value;
            }
            if(!monotone) throw std::runtime_error("Sampled nonlinear measurement norm is nonmonotone");
            const double full_norm=previous;
            double alpha=1.0;
            int iterations=0;
            if(full_norm>epsilon) {
                double lo=0.0,hi=1.0;
                while(hi-lo>1e-10 && iterations<80) {
                    const double mid=(lo+hi)/2.0;
                    if(measure(mid)<=epsilon) lo=mid;
                    else hi=mid;
                    ++iterations;
                }
                alpha=lo;
            }
            const auto nonlinear=simulate(alpha);
            const double final_norm=norm_measurements(reference,nonlinear);
            if(final_norm>epsilon+1e-8) throw std::runtime_error("Feasible amplitude violates measurement bound");
            const auto diag=diagnostics(reference,nonlinear,config,quantity,d.index);
            if(!(diag.min_pressure>0.0) || diag.max_conservation_defect>1e-6)
                throw std::runtime_error("Physical admissibility or conservation check failed");
            out<<quantity<<','<<epsilon<<','<<d.index<<','<<initial_norm<<','<<alpha<<','
               <<final_norm<<','<<full_norm<<','<<monotone<<','<<iterations<<','
               <<diag.terminal_targeted<<','<<std::abs(diag.terminal_targeted)<<','
               <<diag.max_targeted_abs<<','<<diag.min_pressure<<','
               <<diag.max_conservation_defect<<'\n';
            std::cout<<std::setprecision(12)<<quantity<<" index="<<d.index
                     <<" alpha="<<alpha<<" ||r||="<<final_norm
                     <<" terminal="<<diag.terminal_targeted
                     <<" conservation="<<diag.max_conservation_defect<<'\n';
        }
        if(!out) throw std::runtime_error("CSV write failed");
        std::cout<<"M2.4-F2C-2 PASS\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"M2.4-F2C-2 FAIL: "<<e.what()<<'\n';
        return 1;
    }
}
