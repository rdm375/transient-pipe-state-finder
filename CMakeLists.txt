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
