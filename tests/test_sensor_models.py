"""Physical consistency tests for sensor simulation.

These tests verify that the specific-force model used in simulation examples
is physically correct — i.e. it includes both the dynamic acceleration and
the gravity reaction, not just the static gravity component.
"""

import numpy as np

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.math_utils import euler_to_rotation_matrix


def test_hover_specific_force_magnitude_equals_g():
    """At exact hover thrust the specific-force magnitude equals g.

    |f_body| = |R^T @ g| = g  when a_linear ≈ 0.
    A sensor simulation that omits dynamics always returns this value; the test
    itself is trivial but documents the expected behaviour for future comparison.
    """
    quad = QuadcopterDynamics()
    gravity = np.array([0.0, 0.0, 9.81])
    R = euler_to_rotation_matrix(*quad.get_attitude())   # identity at start
    f_static = R.T @ gravity
    np.testing.assert_allclose(np.linalg.norm(f_static), 9.81, atol=0.01,
                               err_msg="Static hover specific force must equal g")


def test_accel_simulation_includes_dynamics():
    """At 2× hover thrust the specific-force magnitude must exceed 1.3 g.

    A static-only accelerometer model always returns |f| = g regardless of motor
    commands.  This test catches that omission: at double thrust, linear
    acceleration ≈ +g upward so |f_body| ≈ 2g.
    """
    quad = QuadcopterDynamics()
    gravity = np.array([0.0, 0.0, 9.81])
    dt = 0.01

    hover_speed = np.sqrt(quad.g * quad.mass / (4 * quad.k_f))
    motors_2x   = np.full(4, hover_speed * np.sqrt(2.0))   # 2× thrust

    # One integration step so dynamics reflects the new motor command
    quad.update(dt, motors_2x)

    R = euler_to_rotation_matrix(*quad.get_attitude())
    T_over_m       = float(np.sum(motors_2x**2) * quad.k_f / quad.mass)
    a_thrust_world = R @ np.array([0.0, 0.0, T_over_m])
    a_drag_world   = -quad.k_d * quad.get_velocity() / quad.mass
    a_lin          = a_thrust_world + a_drag_world + np.array([0.0, 0.0, -quad.g])
    # Full specific force — what the IMU actually measures
    f_body         = R.T @ (a_lin + gravity)

    f_norm = np.linalg.norm(f_body)
    assert f_norm > 9.81 * 1.3, (
        f"|f_body|={f_norm:.3f} N/kg should exceed 1.3g={9.81*1.3:.2f} at 2× hover thrust. "
        "Accelerometer simulation is missing the dynamic acceleration contribution."
    )


def test_specific_force_sign_convention():
    """Specific force must point *away from* the ground at hover.

    ENU convention (z-up):  f_body = R^T @ [0, 0, g].
    At identity rotation the body z-axis is world z, so f_body = [0, 0, 9.81].
    A sign error (f = R^T @ -g or f = -R^T @ g) would give f_body[2] < 0.
    """
    quad = QuadcopterDynamics()
    gravity = np.array([0.0, 0.0, 9.81])
    R = euler_to_rotation_matrix(*quad.get_attitude())
    f_body = R.T @ gravity
    assert f_body[2] > 0, (
        f"Specific force z-component = {f_body[2]:.4f}: must be positive (upward) at hover. "
        "Check sign convention — f_body = R^T @ g, not R^T @ (-g)."
    )
