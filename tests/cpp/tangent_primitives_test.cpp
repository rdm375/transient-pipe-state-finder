#include "state_finder/old_state_jvp.hpp"
#include "state_finder/old_state_vjp.hpp"
#include "state_finder/transpose_solver.hpp"

#include "pipe_sim/transient.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

double dot(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    require(a.size() == b.size(), "Dot dimension mismatch");

    double sum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
        sum += a[i]*b[i];

    return sum;
}

void run_case(int n, double theta)
{
    const int width = 2*n + 2;
    constexpr int kl = 2;
    constexpr int ku = 1;
    constexpr int ldab = 6;

    const double area = std::acos(-1.0)/4.0;

    const pipe_sim::TransientParameters par{
        1000.0, 15.0, theta,
        1.0, area, 288.15, 0.9, 500.0,
        1.1e-5, 4.5e-5
    };

    std::vector<double> state(width);
    std::vector<double> v(width);
    std::vector<double> w(width);

    state[0] = 8.0e6;
    state[1] = 100.0;

    for (int i = 0; i < n; ++i) {
        state[2*i+2] = 100.0 + 0.1*i;
        state[2*i+3] = 8.0e6 - 1500.0*(i+1);
    }

    for (int i = 0; i < width; ++i) {
        v[i] = std::sin(0.37*(i+1));
        w[i] = std::cos(0.29*(i+1));
    }

    // Test B*v against B^T*w.
    std::vector<double> Bv(width);
    std::vector<double> BTw(width);

    state_finder::old_state_jvp(
        n, state, par, v, Bv);

    state_finder::old_state_vjp(
        n, state, par, w, BTw);

    const double lhs = dot(w, Bv);
    const double rhs = dot(v, BTw);

    const double transpose_error =
        std::abs(lhs-rhs) /
        std::max({1.0, std::abs(lhs), std::abs(rhs)});

    require(
        transpose_error < 1.0e-12,
        "JVP/VJP transpose identity failed");

    // Test the nontransposed LAPACK banded solve.
    std::vector<double> ab(ldab*width);

    pipe_sim::assemble_jacobian_banded(
        n, state, par, ab, ldab);

    std::vector<double> exact(width);

    for (int i = 0; i < width; ++i)
        exact[i] = std::sin(0.17*(i+1));

    std::vector<double> b(width, 0.0);

    const int kv = kl + ku;

    for (int j = 0; j < width; ++j) {
        const int first = std::max(0, j-ku);
        const int last = std::min(width-1, j+kl);

        for (int i = first; i <= last; ++i) {
            b[i] +=
                ab[j*ldab + kv + i-j]*exact[j];
        }
    }

    const auto solved = state_finder::solve_banded(
        width, ab, ldab, kl, ku, b);

    double solution_error = 0.0;

    for (int i = 0; i < width; ++i) {
        solution_error = std::max(
            solution_error,
            std::abs(solved.solution[i]-exact[i]));
    }

    std::cout
        << "n=" << n
        << " theta=" << theta
        << " transpose_error=" << transpose_error
        << " backward_error="
        << solved.relative_backward_error
        << " solution_error=" << solution_error
        << '\n';

    require(
        solved.relative_backward_error < 1.0e-12,
        "Banded solve backward error too large");

    require(
        solution_error < 1.0e-8,
        "Banded solve solution error too large");
}

} // namespace

int main()
{
    try {
        std::cout << std::setprecision(17);

        for (int n : {2, 10, 100}) {
            run_case(n, 0.65);
        }

        std::cout
            << "PASS tangent-linear primitives\n";

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
