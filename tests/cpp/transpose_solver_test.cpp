#include "state_finder/transpose_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    try {
        constexpr int n = 8;
        constexpr int kl = 2;
        constexpr int ku = 1;
        constexpr int ldab = 6;
        constexpr int kv = kl + ku;

        std::vector<double> ab(ldab*n, 0.0);
        std::vector<double> expected(n);
        std::vector<double> rhs(n, 0.0);

        const auto set = [&](int i, int j, double value) {
            ab[j*ldab + kv+i-j] = value;
        };

        for (int i = 0; i < n; ++i) {
            set(i, i, 4.0 + 0.1*i);

            if (i+1 < n) {
                set(i+1, i, -0.7);
                set(i, i+1, -0.3);
            }

            if (i+2 < n)
                set(i+2, i, 0.15);

            expected[i] = 1.0 + 0.2*i;
        }

        // Form rhs = A^T * expected independently.
        for (int j = 0; j < n; ++j) {
            for (int i = std::max(0,j-ku);
                 i <= std::min(n-1,j+kl); ++i) {

                rhs[j] +=
                    ab[j*ldab + kv+i-j]*expected[i];
            }
        }

        const auto result =
            state_finder::solve_transposed_banded(
                n, ab, ldab, kl, ku, rhs);

        double max_error = 0.0;

        for (int i = 0; i < n; ++i) {
            max_error = std::max(
                max_error,
                std::abs(result.solution[i]-expected[i]));
        }

        if (max_error > 2.0e-14 ||
            result.relative_backward_error > 2.0e-14) {
            throw std::runtime_error(
                "Transpose solver numerical regression failed");
        }

        std::cout
            << "PASS transpose solver\n"
            << "maximum solution error: " << max_error << '\n'
            << "relative backward error: "
            << result.relative_backward_error << '\n';

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
