# Development roadmap

The following tracks are distinct. Completing a research milestone does not imply production-release readiness.

| Track | Deliverable | Acceptance evidence |
|---|---|---|
| Documentation | Current README and mathematical foundations | Source-linked definitions and checked numerical claims |
| Reproducibility | Fresh-clone instructions and pinned forward-simulator revision | Reproducible clean build on supported Linux |
| Dependency management | Explicit simulator revision and consistent Python environment | Automated dependency checks |
| CI | CMake build, CTest, representative Python tests | Green GitHub Actions run on feature branch and main |
| Scientific report | Consolidated formulation, methods, benchmark tables, limitations | Reviewed results traced to committed data |
| Public API | Versioned configuration, typed outputs, explicit failure statuses | Integration tests and example client |
| Release management | Release checklist, changelog, release notes, semantic tags | All v0.1.0 acceptance gates passed |

## Immediate sequence

1. Review and commit the documentation foundation.
2. Determine and pin the compatible `isothermal-pipe-sim` commit.
3. Establish a minimal CI build and test matrix using that revision.
4. Produce a quantitative scientific report from committed benchmark artifacts.
5. Design the narrow C++ public API separately from experimental drivers.
6. Complete release acceptance gates before any `v0.1.0` tag.

**Scientific language policy:** mathematical exposition targets senior undergraduate mathematics students. Distinguish assumptions, derived identities, computational observations, and unproved claims.
