#include "pipe_sim/physics.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

using Real = boost::multiprecision::cpp_dec_float_50;

struct Reference {
    Real source;
    Real derivative_flow;
    Real derivative_pressure;
};

Reference reference(
    double mdot_double,
    double pressure_left_double,
    double pressure_right_double)
{
    const Real mdot(mdot_double);
    const Real pl(pressure_left_double);
    const Real pr(pressure_right_double);

    const Real diameter("1.0");
    const Real area =
        acos(Real(-1))/Real(4);

    const Real temperature("288.15");
    const Real z("0.9");
    const Real gas_constant("500.0");
    const Real viscosity("0.000011");
    const Real roughness("0.000045");

    const Real crho =
        Real(1)/(z*gas_constant*temperature);

    const Real rho_face =
        (pl+pr)*crho/Real(2);

    const Real re =
        abs(mdot)*diameter/(area*viscosity);

    const Real re09 = pow(re, Real("0.9"));

    const Real arg =
        roughness/(Real("3.7")*diameter)
        + Real("5.74")/re09;

    const Real logarithm = log10(arg);

    const Real friction =
        Real("0.25")/(logarithm*logarithm);

    const Real darg_dre =
        -Real("0.9")*Real("5.74")/
        (re*re09);

    const Real dfriction_dre =
        -Real("0.5")*darg_dre/
        (arg*log(Real(10))*
         logarithm*logarithm*logarithm);

    const Real dre_dm =
        (mdot > 0 ? Real(1) : Real(-1))*
        diameter/(area*viscosity);

    const Real source =
        friction*mdot*abs(mdot)/
        (Real(2)*diameter*area*area*rho_face);

    const Real derivative_flow =
        (dfriction_dre*dre_dm*mdot*abs(mdot)
         + Real(2)*friction*abs(mdot))/
        (Real(2)*diameter*area*area*rho_face);

    const Real derivative_pressure =
        -source*crho/(Real(2)*rho_face);

    return {
        source,
        derivative_flow,
        derivative_pressure
    };
}

void check(double mdot, double pl, double pr)
{
    const double diameter = 1.0;
    const double area = std::acos(-1.0)/4.0;
    const double viscosity = 1.1e-5;
    const double roughness = 4.5e-5;

    const double crho =
        pipe_sim::density_derivative_cz(
            288.15, 0.9, 500.0);

    const double rho_face =
        0.5*(
            pipe_sim::density_cz(pl, 288.15, 0.9, 500.0)
            + pipe_sim::density_cz(pr, 288.15, 0.9, 500.0));

    const double re =
        pipe_sim::reynolds_mass(
            mdot, diameter, area, viscosity);

    const auto friction =
        pipe_sim::friction_sj_value_derivative(
            re, roughness, diameter);

    const double dfdm =
        friction.derivative_re*
        pipe_sim::reynolds_derivative_mdot(
            mdot, diameter, area, viscosity);

    const double source =
        pipe_sim::friction_source(
            friction.value, diameter, area, rho_face, mdot);

    const double derivative_flow =
        pipe_sim::friction_source_derivative_mdot(
            friction.value, dfdm,
            diameter, area, rho_face, mdot);

    const double derivative_pressure =
        -source*crho/(2.0*rho_face);

    const auto ref = reference(mdot, pl, pr);

    auto verify = [&](const char* name,
                      double actual,
                      const Real& expected) {
        const Real discrepancy =
            abs(Real(actual)-expected);

        Real scale = abs(expected);
        if (scale < Real("1e-100"))
            scale = Real("1e-100");

        const Real relative = discrepancy/scale;

        std::cout << std::setprecision(8)
                  << "flow=" << mdot
                  << " quantity=" << name
                  << " relative_error="
                  << std::setprecision(16)
                  << relative << '\n';

        if (relative > Real("2e-13"))
            throw std::runtime_error(
                "High-precision physics comparison failed");
    };

    verify("source", source, ref.source);
    verify("dS/dm", derivative_flow, ref.derivative_flow);
    verify("dS/dp", derivative_pressure, ref.derivative_pressure);
}

} // namespace

int main()
{
    try {
        for (double mdot : {
            -150.0, -100.0, -10.0, -1.0,
            1.0, 10.0, 100.0, 150.0
        }) {
            check(mdot, 8.0e6, 7.8e6);
        }

        std::cout << "PASS high-precision physics derivatives\n";
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
