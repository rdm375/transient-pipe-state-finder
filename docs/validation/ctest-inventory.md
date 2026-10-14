# StateFinder Validation Inventory

Generated: 2026-10-14T15:40:47.600297+00:00

## Source revisions

StateFinder: `8cf54de3bdea0ce5f4d145f592d31dd2fade3fbe`

Forward solver: `670ac493e4461e678ca66f2254dc3d402df3f312`

## Registered tests

~~~text
Test project /nvme/Sync/transient-pipe-state-finder/build-reproducible
  Test  #1: state_finder_forward_smoke
  Test  #2: state_finder_transient_parity
  Test  #3: state_finder_objective
  Test  #4: state_finder_trajectory_parity
  Test  #5: state_finder_transpose_solver
  Test  #6: state_finder_old_state_vjp
  Test  #7: state_finder_adjoint_gradient
  Test  #8: state_finder_high_precision_physics
  Test  #9: state_finder_adjoint_components
  Test #10: state_finder_optimizer_quadratic
  Test #11: state_finder_boundary_measurements
  Test #12: state_finder_tangent_primitives
  Test #13: state_finder_tangent_linear
  Test #14: state_finder_tangent_matrix
  Test #15: state_finder_quadratic_regularizer_test
  Test #16: state_finder_regularized_objective_test
  Test #17: state_finder_matrix_free_adjoint
  Test #18: state_finder_observation_adjoint
  Test #19: state_finder_shared_operators
  Test #20: state_finder_matrix_free_cg
  Test #21: state_finder_gauss_newton_operator
  Test #22: state_finder_regularization_preconditioner_test
  Test #23: state_finder_four_channel_objective
  Test #24: state_finder_boundary_history_integrator
  Test #25: state_finder_boundary_history_parity
  Test #26: state_finder_joint_inverse_objective
  Test #27: state_finder_joint_scaled_problem
  Test #28: state_finder_boundary_derivatives
  Test #29: state_finder_joint_adjoint
  Test #30: state_finder_joint_optimizer_gradient
  Test #31: state_finder_joint_tangent_linear
  Test #32: state_finder_joint_observation_adjoint
  Test #33: state_finder_joint_gauss_newton_operator
  Test #34: state_finder_joint_nonlinear_reconstruction
  Test #35: state_finder_joint_regularization_preconditioner
  Test #36: state_finder_steady_state_initializer
  Test #37: state_finder_m24_tangent_history
  Test #38: cpp_short_transient
  Test #39: cpp_long_transient
  Test #40: cpp_spatial_resolution_200
  Test #41: cpp_spatial_resolution_400
  Test #42: cpp_spatial_convergence
  Test #43: cpp_zero_steps
  Test #44: cpp_numerical_reference_80
  Test #45: cpp_numerical_reference_8000

Total Tests: 45
~~~

## CTest execution

~~~text
Test project /nvme/Sync/transient-pipe-state-finder/build-reproducible
      Start  1: state_finder_forward_smoke
 1/45 Test  #1: state_finder_forward_smoke .........................   Passed    0.00 sec
      Start  2: state_finder_transient_parity
 2/45 Test  #2: state_finder_transient_parity ......................   Passed    0.56 sec
      Start  3: state_finder_objective
 3/45 Test  #3: state_finder_objective .............................   Passed    0.01 sec
      Start  4: state_finder_trajectory_parity
 4/45 Test  #4: state_finder_trajectory_parity .....................   Passed    1.17 sec
      Start  5: state_finder_transpose_solver
 5/45 Test  #5: state_finder_transpose_solver ......................   Passed    0.00 sec
      Start  6: state_finder_old_state_vjp
 6/45 Test  #6: state_finder_old_state_vjp .........................   Passed    0.00 sec
      Start  7: state_finder_adjoint_gradient
 7/45 Test  #7: state_finder_adjoint_gradient ......................   Passed    0.08 sec
      Start  8: state_finder_high_precision_physics
 8/45 Test  #8: state_finder_high_precision_physics ................   Passed    0.02 sec
      Start  9: state_finder_adjoint_components
 9/45 Test  #9: state_finder_adjoint_components ....................   Passed    0.14 sec
      Start 10: state_finder_optimizer_quadratic
10/45 Test #10: state_finder_optimizer_quadratic ...................   Passed    0.00 sec
      Start 11: state_finder_boundary_measurements
11/45 Test #11: state_finder_boundary_measurements .................   Passed    0.00 sec
      Start 12: state_finder_tangent_primitives
12/45 Test #12: state_finder_tangent_primitives ....................   Passed    0.00 sec
      Start 13: state_finder_tangent_linear
13/45 Test #13: state_finder_tangent_linear ........................   Passed    0.37 sec
      Start 14: state_finder_tangent_matrix
14/45 Test #14: state_finder_tangent_matrix ........................   Passed    1.44 sec
      Start 15: state_finder_quadratic_regularizer_test
15/45 Test #15: state_finder_quadratic_regularizer_test ............   Passed    0.01 sec
      Start 16: state_finder_regularized_objective_test
16/45 Test #16: state_finder_regularized_objective_test ............   Passed    0.00 sec
      Start 17: state_finder_matrix_free_adjoint
17/45 Test #17: state_finder_matrix_free_adjoint ...................   Passed    0.79 sec
      Start 18: state_finder_observation_adjoint
18/45 Test #18: state_finder_observation_adjoint ...................   Passed    1.69 sec
      Start 19: state_finder_shared_operators
19/45 Test #19: state_finder_shared_operators ......................   Passed    0.66 sec
      Start 20: state_finder_matrix_free_cg
20/45 Test #20: state_finder_matrix_free_cg ........................   Passed    0.00 sec
      Start 21: state_finder_gauss_newton_operator
21/45 Test #21: state_finder_gauss_newton_operator .................   Passed    0.08 sec
      Start 22: state_finder_regularization_preconditioner_test
22/45 Test #22: state_finder_regularization_preconditioner_test ....   Passed    0.01 sec
      Start 23: state_finder_four_channel_objective
23/45 Test #23: state_finder_four_channel_objective ................   Passed    0.00 sec
      Start 24: state_finder_boundary_history_integrator
24/45 Test #24: state_finder_boundary_history_integrator ...........   Passed    0.00 sec
      Start 25: state_finder_boundary_history_parity
25/45 Test #25: state_finder_boundary_history_parity ...............   Passed    0.01 sec
      Start 26: state_finder_joint_inverse_objective
26/45 Test #26: state_finder_joint_inverse_objective ...............   Passed    0.01 sec
      Start 27: state_finder_joint_scaled_problem
27/45 Test #27: state_finder_joint_scaled_problem ..................   Passed    0.02 sec
      Start 28: state_finder_boundary_derivatives
28/45 Test #28: state_finder_boundary_derivatives ..................   Passed    0.00 sec
      Start 29: state_finder_joint_adjoint
29/45 Test #29: state_finder_joint_adjoint .........................   Passed    0.01 sec
      Start 30: state_finder_joint_optimizer_gradient
30/45 Test #30: state_finder_joint_optimizer_gradient ..............   Passed    0.01 sec
      Start 31: state_finder_joint_tangent_linear
31/45 Test #31: state_finder_joint_tangent_linear ..................   Passed    0.00 sec
      Start 32: state_finder_joint_observation_adjoint
32/45 Test #32: state_finder_joint_observation_adjoint .............   Passed    0.00 sec
      Start 33: state_finder_joint_gauss_newton_operator
33/45 Test #33: state_finder_joint_gauss_newton_operator ...........   Passed    0.01 sec
      Start 34: state_finder_joint_nonlinear_reconstruction
34/45 Test #34: state_finder_joint_nonlinear_reconstruction ........   Passed    0.77 sec
      Start 35: state_finder_joint_regularization_preconditioner
35/45 Test #35: state_finder_joint_regularization_preconditioner ...   Passed    0.00 sec
      Start 36: state_finder_steady_state_initializer
36/45 Test #36: state_finder_steady_state_initializer ..............   Passed    0.00 sec
      Start 37: state_finder_m24_tangent_history
37/45 Test #37: state_finder_m24_tangent_history ...................   Passed    0.00 sec
      Start 38: cpp_short_transient
38/45 Test #38: cpp_short_transient ................................   Passed    0.04 sec
      Start 39: cpp_long_transient
39/45 Test #39: cpp_long_transient .................................   Passed    0.51 sec
      Start 40: cpp_spatial_resolution_200
40/45 Test #40: cpp_spatial_resolution_200 .........................   Passed    0.06 sec
      Start 41: cpp_spatial_resolution_400
41/45 Test #41: cpp_spatial_resolution_400 .........................   Passed    0.12 sec
      Start 42: cpp_spatial_convergence
42/45 Test #42: cpp_spatial_convergence ............................   Passed    0.28 sec
      Start 43: cpp_zero_steps
43/45 Test #43: cpp_zero_steps .....................................   Passed    0.00 sec
      Start 44: cpp_numerical_reference_80
44/45 Test #44: cpp_numerical_reference_80 .........................   Passed    0.09 sec
      Start 45: cpp_numerical_reference_8000
45/45 Test #45: cpp_numerical_reference_8000 .......................   Passed    0.58 sec

100% tests passed, 0 tests failed out of 45

Total Test time (real) =   9.63 sec
~~~
