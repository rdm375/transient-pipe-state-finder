cmake_minimum_required(VERSION 3.20)

project(
    TransientPipeStateFinder
    VERSION 0.1.0
    LANGUAGES CXX
)

include(CTest)

set(
    PIPE_SIM_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../isothermal-pipe-sim"
    CACHE PATH
    "Path to isothermal-pipe-sim"
)

if(NOT EXISTS "${PIPE_SIM_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "Forward simulator not found: ${PIPE_SIM_SOURCE_DIR}"
    )
endif()

# Reuse the existing C++ forward solver.
add_subdirectory(
    "${PIPE_SIM_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/pipe_sim"
)

find_package(LAPACK REQUIRED)

add_executable(
    pipe_state_finder_smoke
    tests/cpp/forward_smoke.cpp
)

target_link_libraries(
    pipe_state_finder_smoke
    PRIVATE
        pipe_sim_cpp_core
        LAPACK::LAPACK
)

target_compile_features(
    pipe_state_finder_smoke
    PRIVATE cxx_std_20
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_forward_smoke
        COMMAND pipe_state_finder_smoke
    )
endif()


# State-finder transient numerical parity
add_executable(
    pipe_state_finder_parity
    tests/cpp/transient_parity.cpp
)

target_link_libraries(
    pipe_state_finder_parity
    PRIVATE pipe_sim_cpp_core
)

target_compile_features(
    pipe_state_finder_parity
    PRIVATE cxx_std_20
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_transient_parity
        COMMAND pipe_state_finder_parity
    )
endif()


# State-finder objective library
add_library(state_finder_core STATIC
    src/cpp/forward_model.cpp
    src/cpp/objective.cpp
)

target_compile_features(
    state_finder_core
    PUBLIC cxx_std_20
)

target_include_directories(
    state_finder_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(
    state_finder_core
    PUBLIC pipe_sim_cpp_core
)

add_executable(
    state_finder_objective_test
    tests/cpp/objective_test.cpp
)

target_link_libraries(
    state_finder_objective_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_objective
        COMMAND state_finder_objective_test
    )
endif()


# State-finder full trajectory recording
target_sources(
    state_finder_core
    PRIVATE src/cpp/trajectory.cpp
)

add_executable(
    state_finder_trajectory_parity
    tests/cpp/trajectory_parity.cpp
)

target_link_libraries(
    state_finder_trajectory_parity
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_trajectory_parity
        COMMAND state_finder_trajectory_parity
    )
endif()


# State-finder derivative infrastructure
target_sources(
    state_finder_core
    PRIVATE
        src/cpp/finite_difference.cpp
        src/cpp/transpose_solver.cpp
)

target_link_libraries(
    state_finder_core
    PUBLIC LAPACK::LAPACK
)

add_executable(
    state_finder_transpose_solver_test
    tests/cpp/transpose_solver_test.cpp
)

target_link_libraries(
    state_finder_transpose_solver_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_transpose_solver
        COMMAND state_finder_transpose_solver_test
    )
endif()


# Analytic old-state residual VJP
target_sources(
    state_finder_core
    PRIVATE src/cpp/old_state_vjp.cpp
)

add_executable(
    state_finder_old_state_vjp_test
    tests/cpp/old_state_vjp_test.cpp
)

target_link_libraries(
    state_finder_old_state_vjp_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_old_state_vjp
        COMMAND state_finder_old_state_vjp_test
    )
endif()


# Complete discrete adjoint
target_sources(
    state_finder_core
    PRIVATE src/cpp/adjoint.cpp
)


# End-to-end adjoint gradient verification
add_executable(
    state_finder_adjoint_gradient_test
    tests/cpp/adjoint_gradient_test.cpp
)

target_link_libraries(
    state_finder_adjoint_gradient_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_adjoint_gradient
        COMMAND state_finder_adjoint_gradient_test
    )
endif()


# M1.5 numerical verification
add_executable(
    state_finder_high_precision_physics_test
    tests/cpp/high_precision_physics_test.cpp
)

target_link_libraries(
    state_finder_high_precision_physics_test
    PRIVATE state_finder_core
)

add_executable(
    state_finder_adjoint_components_test
    tests/cpp/adjoint_components_test.cpp
)

target_link_libraries(
    state_finder_adjoint_components_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_high_precision_physics
        COMMAND state_finder_high_precision_physics_test
    )

    add_test(
        NAME state_finder_adjoint_components
        COMMAND state_finder_adjoint_components_test
    )
endif()


# M1.5b long-trajectory benchmark
# Deliberately excluded from the regular CTest suite.
add_executable(
    state_finder_adjoint_long_trajectory
    tests/cpp/adjoint_long_trajectory.cpp
)

target_link_libraries(
    state_finder_adjoint_long_trajectory
    PRIVATE state_finder_core
)


# M1.5b — Observation-window sensitivity diagnostics
add_executable(
    state_finder_observation_windows
    tests/cpp/observation_windows.cpp
)

target_link_libraries(
    state_finder_observation_windows
    PRIVATE state_finder_core
)


# M1.6a — Reusable L-BFGS optimizer
target_sources(
    state_finder_core
    PRIVATE src/cpp/optimizer.cpp
)

add_executable(
    state_finder_optimizer_quadratic_test
    tests/cpp/optimizer_quadratic_test.cpp
)

target_link_libraries(
    state_finder_optimizer_quadratic_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_optimizer_quadratic
        COMMAND state_finder_optimizer_quadratic_test
    )
endif()


# M1.6a — Pipeline reconstruction
# Excluded from regular CTest: nonlinear optimization benchmark.
add_executable(
    state_finder_inverse_state_reconstruction
    tests/cpp/inverse_state_reconstruction.cpp
)

target_link_libraries(
    state_finder_inverse_state_reconstruction
    PRIVATE state_finder_core
)


# M1.6a — Four-channel boundary measurements
add_executable(
    state_finder_boundary_measurements_test
    tests/cpp/boundary_measurements_test.cpp
)

target_link_libraries(
    state_finder_boundary_measurements_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_boundary_measurements
        COMMAND state_finder_boundary_measurements_test
    )
endif()


# M1.6b — Initial-state numerical observability
# Excluded from regular CTest: 402 forward integrations.
add_executable(
    state_finder_initial_state_observability
    tests/cpp/initial_state_observability.cpp
)

target_link_libraries(
    state_finder_initial_state_observability
    PRIVATE state_finder_core
)


# M1.6b — Analytic tangent-linear primitives
target_sources(
    state_finder_core
    PRIVATE src/cpp/old_state_jvp.cpp
)

add_executable(
    state_finder_tangent_primitives_test
    tests/cpp/tangent_primitives_test.cpp
)

target_link_libraries(
    state_finder_tangent_primitives_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_tangent_primitives
        COMMAND state_finder_tangent_primitives_test
    )
endif()


# M1.6b — Single-direction discrete tangent-linear propagation
target_sources(
    state_finder_core
    PRIVATE src/cpp/tangent_linear.cpp
)

add_executable(
    state_finder_tangent_linear_test
    tests/cpp/tangent_linear_test.cpp
)

target_link_libraries(
    state_finder_tangent_linear_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_tangent_linear
        COMMAND state_finder_tangent_linear_test
    )
endif()


# M1.6b — Multiple-RHS tangent sensitivity matrix
target_sources(
    state_finder_core
    PRIVATE
        src/cpp/banded_multi_rhs.cpp
        src/cpp/tangent_matrix.cpp
)

add_executable(
    state_finder_tangent_matrix_test
    tests/cpp/tangent_matrix_test.cpp
)

target_link_libraries(
    state_finder_tangent_matrix_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_tangent_matrix
        COMMAND state_finder_tangent_matrix_test
    )
endif()


# M1.6b — Analytic initial-state observability
# Deliberately excluded from regular CTest.
add_executable(
    state_finder_analytic_observability
    tests/cpp/analytic_initial_state_observability.cpp
)

target_link_libraries(
    state_finder_analytic_observability
    PRIVATE state_finder_core
)

add_executable(
    state_finder_quadratic_regularizer_test
    tests/cpp/quadratic_regularizer_test.cpp
)

target_link_libraries(
    state_finder_quadratic_regularizer_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_quadratic_regularizer_test
        COMMAND state_finder_quadratic_regularizer_test
    )
endif()
add_executable(
    state_finder_regularized_objective_test
    tests/cpp/regularized_objective_test.cpp
)

target_link_libraries(
    state_finder_regularized_objective_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_regularized_objective_test
        COMMAND state_finder_regularized_objective_test
    )
endif()

# M1.6c — Nonlinear regularized initial-state reconstruction.

add_executable(
    state_finder_regularized_reconstruction
    tests/cpp/regularized_state_reconstruction.cpp
)

target_link_libraries(
    state_finder_regularized_reconstruction
    PRIVATE state_finder_core
)

# M1.7e — Exact discrete modal operator diagnostic.
# Excluded from regular CTest.
add_executable(
    state_finder_discrete_modal_operator
    analysis/discrete_modal_operator.cpp
)

target_link_libraries(
    state_finder_discrete_modal_operator
    PRIVATE state_finder_core
)

# M1.7h — Direct one-direction predictive response.
add_executable(
    state_finder_direct_predictive_response
    analysis/direct_predictive_response.cpp
)

target_link_libraries(
    state_finder_direct_predictive_response
    PRIVATE state_finder_core
)

# M1.8 — Matrix-free tangent/adjoint identity.
add_executable(
    state_finder_matrix_free_adjoint_test
    analysis/matrix_free_adjoint_test.cpp
)

target_link_libraries(
    state_finder_matrix_free_adjoint_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_matrix_free_adjoint
        COMMAND state_finder_matrix_free_adjoint_test
    )
endif()

# M1.8b — Matrix-free prior/noise-scaled normal operator.
add_executable(
    state_finder_matrix_free_normal_operator
    analysis/matrix_free_normal_operator.cpp
)

target_link_libraries(
    state_finder_matrix_free_normal_operator
    PRIVATE state_finder_core
)

# M1.8c-1 — Direct observation-forced discrete adjoint.
target_sources(
    state_finder_core
    PRIVATE src/cpp/observation_adjoint.cpp
)

add_executable(
    state_finder_observation_adjoint_test
    tests/cpp/observation_adjoint_test.cpp
)

target_link_libraries(
    state_finder_observation_adjoint_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_observation_adjoint
        COMMAND state_finder_observation_adjoint_test
    )
endif()

# M1.8c — Independent transpose identity and
# shared-trajectory tangent equivalence.
add_executable(
    state_finder_shared_operators_test
    analysis/verify_shared_operators.cpp
)

target_link_libraries(
    state_finder_shared_operators_test
    PRIVATE state_finder_core
)

# M1.8c-3 — Shared-trajectory, in-process Lanczos.
add_executable(
    state_finder_inprocess_lanczos
    analysis/inprocess_lanczos.cpp
)

target_link_libraries(
    state_finder_inprocess_lanczos
    PRIVATE
        state_finder_core
        LAPACK::LAPACK
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_shared_operators
        COMMAND
            state_finder_shared_operators_test
            80
            "${CMAKE_CURRENT_BINARY_DIR}/shared_operator_test.txt"
    )
endif()
