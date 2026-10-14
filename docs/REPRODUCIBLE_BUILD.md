# Reproducible StateFinder Build

## Overview

StateFinder depends on the C++ transient-pipe forward simulator.

Repository:

https://github.com/rdm375/transient-pipe-flow

Verified revision:

`670ac493e4461e678ca66f2254dc3d402df3f312`

The dependency revision is recorded in
`cmake/ForwardSolverRevision.cmake`.

## Verified development environment

| Component | Version |
|---|---|
| GCC | 15.2.0 |
| CMake | 4.2.3 |
| BLAS | 3.12.1 |
| LAPACK | 3.12.1 |
| Registered CTest tests | 45 |
| Passing tests | 45 |
| Failing tests | 0 |

The tests completed in approximately 10.36 seconds.

This establishes a verified local baseline, not bitwise
reproducibility across all supported platforms.

## System dependencies

On Ubuntu, install:

    sudo apt-get update

    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        python3 \
        libblas-dev \
        liblapack-dev

A C++20-capable compiler is required.

## Checkout the forward solver

From the StateFinder repository root:

    ./scripts/checkout-forward-solver.sh

This creates:

    external/transient-pipe-flow

The script reads the pinned revision from the dependency
manifest and refuses to overwrite an existing directory.

## Configure

    cmake -S . -B build-reproducible \
        -DPIPE_SIM_SOURCE_DIR="$PWD/external/transient-pipe-flow" \
        -DBUILD_TESTING=ON \
        -DCMAKE_BUILD_TYPE=Release

## Build

    cmake --build build-reproducible --parallel 2

## Test

    ctest --test-dir build-reproducible \
        --output-on-failure

## Verify the dependency

    git -C external/transient-pipe-flow rev-parse HEAD

The result must equal:

    670ac493e4461e678ca66f2254dc3d402df3f312

Verify the CMake configuration:

    grep '^PIPE_SIM_SOURCE_DIR:' \
        build-reproducible/CMakeCache.txt

## Existing development worktrees

An existing forward-solver worktree may be supplied directly:

    cmake -S . -B build-reproducible \
        -DPIPE_SIM_SOURCE_DIR="$HOME/Sync/isothermal-pipe-sim-statefinder" \
        -DBUILD_TESTING=ON

This avoids disturbing active forward-solver development.

## Scientific reproducibility

Passing the registered test suite establishes that the
software builds and passes its current automated checks.

It does not establish bitwise-identical results across
compilers, processors, operating systems, or BLAS libraries.

Scientific validation additionally requires documented:

- Physical parameters and boundary conditions.
- Spatial and temporal discretizations.
- Solver tolerances.
- Conservation defects.
- Tangent-linear and adjoint verification.
- Independent finite-difference comparisons.
- Numerical error thresholds.
- Reproducible experiment configurations.

## Continuous integration

GitHub Actions executes the CPU build and registered
CTest suite on Ubuntu 24.04.

The workflow is:

    .github/workflows/ci.yml

The first successful CI run establishes the baseline
on a GitHub-hosted environment.
