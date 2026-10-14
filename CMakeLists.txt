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

# M2.0a — Matrix-free Gauss–Newton linear solver.
add_executable(
    state_finder_matrix_free_cg_test
    tests/cpp/matrix_free_cg_test.cpp
)

target_link_libraries(
    state_finder_matrix_free_cg_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_matrix_free_cg
        COMMAND state_finder_matrix_free_cg_test
    )
endif()

# M2.0b — Matrix-free pipeline Gauss-Newton operator.
add_executable(
    state_finder_gauss_newton_operator_test
    tests/cpp/gauss_newton_operator_test.cpp
)

target_link_libraries(
    state_finder_gauss_newton_operator_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_gauss_newton_operator
        COMMAND state_finder_gauss_newton_operator_test
    )
endif()

# M2.0e: exact regularization-Hessian preconditioner test.
add_executable(
    state_finder_regularization_preconditioner_test
    tests/cpp/regularization_preconditioner_test.cpp
)

target_include_directories(
    state_finder_regularization_preconditioner_test
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_test(
    NAME state_finder_regularization_preconditioner_test
    COMMAND state_finder_regularization_preconditioner_test
)

# M2.2a — Full information spectrum for the
# M2.1 synthetic initial-state reconstruction.
# Deliberately excluded from regular CTest.
add_executable(
    state_finder_m22_information_spectrum
    analysis/m22_information_spectrum.cpp
)

target_link_libraries(
    state_finder_m22_information_spectrum
    PRIVATE
        state_finder_core
        LAPACK::LAPACK
)


# M2.2c — Full four-channel measurement objective.
target_sources(
    state_finder_core
    PRIVATE src/cpp/four_channel_objective.cpp
)

add_executable(
    state_finder_four_channel_objective_test
    tests/cpp/four_channel_objective_test.cpp
)

target_link_libraries(
    state_finder_four_channel_objective_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_four_channel_objective
        COMMAND state_finder_four_channel_objective_test
    )
endif()


# M2.2c — Adjustable boundary-history integration.
target_sources(
    state_finder_core
    PRIVATE src/cpp/boundary_history_integrator.cpp
)

add_executable(
    state_finder_boundary_history_integrator_test
    tests/cpp/boundary_history_integrator_test.cpp
)

target_link_libraries(
    state_finder_boundary_history_integrator_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_boundary_history_integrator
        COMMAND state_finder_boundary_history_integrator_test
    )
endif()


# M2.2c — Boundary-history trajectory parity.
add_executable(
    state_finder_boundary_history_parity_test
    tests/cpp/boundary_history_parity_test.cpp
)

target_link_libraries(
    state_finder_boundary_history_parity_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_boundary_history_parity
        COMMAND state_finder_boundary_history_parity_test
    )
endif()


# M2.2c — Joint four-measurement inverse objective.
target_sources(
    state_finder_core
    PRIVATE src/cpp/joint_inverse_objective.cpp
)

add_executable(
    state_finder_joint_inverse_objective_test
    tests/cpp/joint_inverse_objective_test.cpp
)

target_link_libraries(
    state_finder_joint_inverse_objective_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_joint_inverse_objective
        COMMAND state_finder_joint_inverse_objective_test
    )
endif()


# M2.2c — Scaled joint four-measurement inverse problem.
target_sources(
    state_finder_core
    PRIVATE src/cpp/joint_scaled_problem.cpp
)

add_executable(
    state_finder_joint_scaled_problem_test
    tests/cpp/joint_scaled_problem_test.cpp
)

target_link_libraries(
    state_finder_joint_scaled_problem_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_joint_scaled_problem
        COMMAND state_finder_joint_scaled_problem_test
    )
endif()


# M2.2c — Joint reconstruction benchmark.
add_executable(
    state_finder_m22c_joint_reconstruction
    benchmarks/m22c_joint_reconstruction.cpp
)

target_link_libraries(
    state_finder_m22c_joint_reconstruction
    PRIVATE state_finder_core
)


# M2.2c-6A — Boundary residual derivatives.
target_sources(
    state_finder_core
    PRIVATE src/cpp/boundary_derivatives.cpp
)

add_executable(
    state_finder_boundary_derivatives_test
    tests/cpp/boundary_derivatives_test.cpp
)

target_link_libraries(
    state_finder_boundary_derivatives_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_boundary_derivatives
        COMMAND state_finder_boundary_derivatives_test
    )
endif()


# M2.2c-6B — Joint discrete adjoint.
target_sources(
    state_finder_core
    PRIVATE
        src/cpp/joint_trajectory.cpp
        src/cpp/joint_adjoint.cpp
)


# M2.2c-6B — Joint adjoint validation.
add_executable(
    state_finder_joint_adjoint_test
    tests/cpp/joint_adjoint_test.cpp
)

target_link_libraries(
    state_finder_joint_adjoint_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_joint_adjoint
        COMMAND state_finder_joint_adjoint_test
    )
endif()


# M2.2c-6C — Analytic joint regularization gradient.
target_sources(
    state_finder_core
    PRIVATE src/cpp/joint_regularization_gradient.cpp
)


# M2.2c-6C — Joint optimizer gradient test.
add_executable(
    state_finder_joint_optimizer_gradient_test
    tests/cpp/joint_optimizer_gradient_test.cpp
)

target_link_libraries(
    state_finder_joint_optimizer_gradient_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_joint_optimizer_gradient
        COMMAND state_finder_joint_optimizer_gradient_test
    )
endif()


# M2.2c-8B — Four-channel joint tangent-linear operator.
target_sources(state_finder_core PRIVATE src/cpp/joint_tangent_linear.cpp)

add_executable(state_finder_joint_tangent_linear_test
    tests/cpp/joint_tangent_linear_test.cpp)
target_link_libraries(state_finder_joint_tangent_linear_test
    PRIVATE state_finder_core)
if(BUILD_TESTING)
    add_test(NAME state_finder_joint_tangent_linear
             COMMAND state_finder_joint_tangent_linear_test)
endif()

# M2.2c-7 — Joint convergence study.
add_executable(
    state_finder_m22c_convergence_study
    benchmarks/m22c_convergence_study.cpp
)

target_link_libraries(
    state_finder_m22c_convergence_study
    PRIVATE state_finder_core
)


# M2.2c-8C — Arbitrary four-channel observation adjoint.
add_executable(
    state_finder_joint_observation_adjoint_test
    tests/cpp/joint_observation_adjoint_test.cpp
)

target_link_libraries(
    state_finder_joint_observation_adjoint_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_joint_observation_adjoint
        COMMAND state_finder_joint_observation_adjoint_test
    )
endif()


# M2.2c-8D — Joint matrix-free Gauss-Newton operator.
add_executable(
    state_finder_joint_gauss_newton_operator_test
    tests/cpp/joint_gauss_newton_operator_test.cpp
)

target_link_libraries(
    state_finder_joint_gauss_newton_operator_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_joint_gauss_newton_operator
        COMMAND state_finder_joint_gauss_newton_operator_test
    )
endif()


# M2.2c-8E — End-to-end joint nonlinear reconstruction.
add_executable(
    state_finder_joint_nonlinear_reconstruction_test
    tests/cpp/joint_nonlinear_reconstruction_test.cpp
)

target_link_libraries(
    state_finder_joint_nonlinear_reconstruction_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_joint_nonlinear_reconstruction
        COMMAND state_finder_joint_nonlinear_reconstruction_test
    )
endif()

# M2.2c-8F — Joint regularization preconditioner.
add_executable(
    state_finder_joint_regularization_preconditioner_test
    tests/cpp/joint_regularization_preconditioner_test.cpp
)

target_link_libraries(
    state_finder_joint_regularization_preconditioner_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_joint_regularization_preconditioner
        COMMAND state_finder_joint_regularization_preconditioner_test
    )
endif()

# M2.3-A — Engineering reference trajectory.
add_executable(
    state_finder_m23a_reference
    apps/m23a_reference_trajectory.cpp
)

target_link_libraries(
    state_finder_m23a_reference
    PRIVATE state_finder_core
)


# Reusable analytical steady-state initializer.
target_sources(
    state_finder_core
    PRIVATE src/cpp/steady_state_initializer.cpp
)

add_executable(
    state_finder_steady_state_initializer_test
    tests/cpp/steady_state_initializer_test.cpp
)

target_link_libraries(
    state_finder_steady_state_initializer_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_steady_state_initializer
        COMMAND state_finder_steady_state_initializer_test
    )
endif()

# M2.3-C: Reproducible synthetic boundary measurements.
add_executable(
    state_finder_m23c_measurements
    apps/m23c_generate_measurements.cpp
)

target_compile_features(
    state_finder_m23c_measurements
    PRIVATE cxx_std_17
)


# M2.3-E — Four-channel engineering reconstruction.
add_executable(
    state_finder_m23e_reconstruction
    apps/m23e_four_channel_reconstruction.cpp
)

target_link_libraries(
    state_finder_m23e_reconstruction
    PRIVATE state_finder_core
)

# M2.4: Joint four-channel observability analysis.
add_executable(
    state_finder_m24_observability
    apps/m24_observability.cpp
)

target_link_libraries(
    state_finder_m24_observability
    PRIVATE state_finder_core
)

# M2.4: Physical decay of weakly observable initial-state modes.
add_executable(
    state_finder_m24_mode_decay
    apps/m24_mode_decay.cpp
)

target_link_libraries(
    state_finder_m24_mode_decay
    PRIVATE state_finder_core
)

# M2.4-F1: Full-history tangent-linear regression.
add_executable(
    state_finder_m24_tangent_history_test
    tests/m24_tangent_history.cpp
)

target_link_libraries(
    state_finder_m24_tangent_history_test
    PRIVATE state_finder_core
)

add_test(
    NAME state_finder_m24_tangent_history
    COMMAND state_finder_m24_tangent_history_test
)

# M2.4-F1: Time-dependent operational sensitivity matrices.
add_executable(
    state_finder_m24_operational_sensitivity
    apps/m24_operational_sensitivity.cpp
)

target_link_libraries(
    state_finder_m24_operational_sensitivity
    PRIVATE state_finder_core
)

# M2.4-F2C: Nonlinear validation of adversarial initial states.
add_executable(
    state_finder_m24_adversarial_validation
    apps/m24_adversarial_validation.cpp
)

target_link_libraries(
    state_finder_m24_adversarial_validation
    PRIVATE state_finder_core
)

# M2.4-F2C-2: Nonlinear measurement-feasible adversarial rays.
# Excluded from regular CTest because it performs hundreds
# of nonlinear forward integrations.
add_executable(
    state_finder_m24_nonlinear_feasibility
    apps/m24_nonlinear_feasibility.cpp
)

target_link_libraries(
    state_finder_m24_nonlinear_feasibility
    PRIVATE state_finder_core
)


# M2.4-F2D: Nonlinear operational uncertainty oracle.
add_executable(
    state_finder_m24_f2d_oracle
    apps/m24_f2d_nonlinear_oracle.cpp
)
target_link_libraries(
    state_finder_m24_f2d_oracle
    PRIVATE state_finder_core
)

# M2.5-B1: Joint initial-state and boundary-history
# operational sensitivities.
add_executable(
    state_finder_m25_joint_operational_sensitivity
    apps/m25_joint_operational_sensitivity.cpp
)

target_link_libraries(
    state_finder_m25_joint_operational_sensitivity
    PRIVATE state_finder_core
)


add_executable(
    state_finder_m25_joint_nonlinear_replay
    apps/m25_joint_nonlinear_replay.cpp
)

target_link_libraries(
    state_finder_m25_joint_nonlinear_replay
    PRIVATE state_finder_core
)


# M27 correlated measurement covariance regression test.
add_executable(
    state_finder_m27_temporal_covariance_test
    tests/temporal_measurement_covariance.cpp
)

target_include_directories(
    state_finder_m27_temporal_covariance_test
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_compile_features(
    state_finder_m27_temporal_covariance_test
    PRIVATE cxx_std_20
)

add_test(
    NAME state_finder_m27_temporal_covariance
    COMMAND state_finder_m27_temporal_covariance_test
)


# M27 covariance-weighted Gauss-Newton interface compilation.
add_executable(
    state_finder_m27_correlated_operator_compile
    tests/cpp/m27_correlated_operator_compile.cpp
)

target_compile_features(
    state_finder_m27_correlated_operator_compile
    PRIVATE cxx_std_20
)

target_link_libraries(
    state_finder_m27_correlated_operator_compile
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_m27_correlated_operator_compile
        COMMAND state_finder_m27_correlated_operator_compile
    )
endif()


# M27 covariance-weighted tangent/adjoint and GN validation.
add_executable(
    state_finder_m27_correlated_operator_test
    tests/cpp/m27_correlated_operator_test.cpp
)

target_compile_features(
    state_finder_m27_correlated_operator_test
    PRIVATE cxx_std_20
)

target_link_libraries(
    state_finder_m27_correlated_operator_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_m27_correlated_operator
        COMMAND state_finder_m27_correlated_operator_test
    )
endif()


# M27 covariance-weighted nonlinear objective gradient.
add_executable(
    state_finder_m27_correlated_objective_gradient_test
    tests/cpp/m27_correlated_objective_gradient_test.cpp
)

target_compile_features(
    state_finder_m27_correlated_objective_gradient_test
    PRIVATE cxx_std_20
)

target_link_libraries(
    state_finder_m27_correlated_objective_gradient_test
    PRIVATE state_finder_core
)

if(BUILD_TESTING)
    add_test(
        NAME state_finder_m27_correlated_objective_gradient
        COMMAND state_finder_m27_correlated_objective_gradient_test
    )
endif()


# M27 native covariance-weighted nonlinear reconstruction.
add_executable(
    state_finder_m27_correlated_reconstruction
    apps/m27_correlated_nonlinear_reconstruction.cpp
)

target_compile_features(
    state_finder_m27_correlated_reconstruction
    PRIVATE cxx_std_20
)

target_link_libraries(
    state_finder_m27_correlated_reconstruction
    PRIVATE state_finder_core
)

# M27-A6: Nonlinear terminal inventory directional sensitivity.
add_executable(
    state_finder_m27_inventory_directional_sensitivity
    apps/m27_inventory_directional_sensitivity.cpp
)

target_compile_features(
    state_finder_m27_inventory_directional_sensitivity
    PRIVATE cxx_std_20
)

target_link_libraries(
    state_finder_m27_inventory_directional_sensitivity
    PRIVATE state_finder_core
)

# M27-B1: pressure calibration observation adapter.
add_executable(
    state_finder_m27_pressure_calibration_test
    tests/cpp/m27_pressure_calibration_test.cpp
)

target_link_libraries(
    state_finder_m27_pressure_calibration_test
    PRIVATE state_finder_core
)

add_test(
    NAME state_finder_m27_pressure_calibration_test
    COMMAND state_finder_m27_pressure_calibration_test
)

# M27-B2: calibrated nonlinear objective and GN operator.
add_executable(
    state_finder_m27_calibrated_joint_test
    tests/cpp/m27_calibrated_joint_test.cpp
)

target_link_libraries(
    state_finder_m27_calibrated_joint_test
    PRIVATE state_finder_core
)

add_test(
    NAME state_finder_m27_calibrated_joint_test
    COMMAND state_finder_m27_calibrated_joint_test
)


# M27-B3: calibrated 165-variable nonlinear reconstruction.
add_executable(
    state_finder_m27_calibrated_reconstruction
    apps/m27_calibrated_nonlinear_reconstruction.cpp
)

target_link_libraries(
    state_finder_m27_calibrated_reconstruction
    PRIVATE state_finder_core
)
