"""Nonlinear six-degree-of-freedom quadcopter rigid-body dynamics.

Frames use ENU world coordinates (z up) and a front-right-up body frame.
Quaternions are Hamilton ``[w, x, y, z]`` and rotate body vectors to world.
The numerical state remains a 13-vector for compatibility::

    [position(3), velocity(3), quaternion(4), body_rates(3)]

The plant models bounded first-order actuators, arbitrary rigid-body rotation,
anisotropic linear/quadratic air drag, rotor failures, and external wrenches.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..math_utils import (
    quat_derivative,
    quat_from_euler,
    quat_normalize,
    quat_to_euler,
    quat_to_rotation_matrix,
)
from ..state import VehicleState
from .config import QuadcopterConfig


class QuadcopterDynamics:
    """A validated 13-state quaternion quadcopter model.

    Legacy scalar constructor arguments are retained. New code should prefer a
    :class:`QuadcopterConfig`, which makes units and actuator limits explicit.
    """

    STATE_SIZE = 13

    def __init__(
        self,
        mass: float = 1.0,
        arm_length: float = 0.2,
        inertia: NDArray | None = None,
        k_f: float = 1.0e-6,
        k_m: float = 1.0e-7,
        k_d: float | NDArray = 0.1,
        g: float = 9.81,
        motor_time_constant: float = 0.0,
        disturbances: list | None = None,
        *,
        config: QuadcopterConfig | None = None,
        quadratic_drag: float | NDArray = 0.0,
        max_motor_speed: float = 4000.0,
    ) -> None:
        cfg = config or QuadcopterConfig(
            mass=mass,
            arm_length=arm_length,
            inertia=np.diag([0.01, 0.01, 0.018]) if inertia is None else inertia,
            thrust_coefficient=k_f,
            moment_coefficient=k_m,
            linear_drag=k_d,
            quadratic_drag=quadratic_drag,
            gravity=g,
            motor_time_constant=motor_time_constant,
            max_motor_speed=max_motor_speed,
        )
        self.config = cfg

        # Public aliases preserve the long-standing API used by examples/RL.
        self.mass = cfg.mass
        self.arm_length = cfg.arm_length
        self.I = cfg.inertia.copy()
        self.k_f = cfg.thrust_coefficient
        self.k_m = cfg.moment_coefficient
        self.k_d = float(cfg.linear_drag[0]) if np.allclose(
            cfg.linear_drag, cfg.linear_drag[0]
        ) else cfg.linear_drag.copy()
        self.g = cfg.gravity
        self.motor_time_constant = cfg.motor_time_constant
        self.min_motor_speed = cfg.min_motor_speed
        self.max_motor_speed = cfg.max_motor_speed
        self.disturbances = list(disturbances or [])

        self.state = np.zeros(self.STATE_SIZE)
        self.state[6] = 1.0
        self.motor_states = np.zeros(4)
        self.last_acceleration = np.zeros(3)
        self.last_angular_acceleration = np.zeros(3)
        self.last_wrench = np.zeros(4)
        self._sim_time = 0.0
        self._step_wind = np.zeros(3)
        self._rotor_efficiencies = np.ones(4)
        self._ground_effect_multiplier = 1.0

    def reset(
        self,
        position: NDArray | None = None,
        attitude: NDArray | None = None,
        *,
        state: VehicleState | None = None,
    ) -> None:
        """Reset the plant, actuators, clock, and disturbance episode state."""
        if state is not None and (position is not None or attitude is not None):
            raise ValueError("provide either state or position/attitude, not both")
        if state is None:
            self.state = np.zeros(self.STATE_SIZE)
            self.state[6] = 1.0
            if position is not None:
                value = np.asarray(position, dtype=float)
                if value.shape != (3,):
                    raise ValueError("position must have shape (3,)")
                self.state[:3] = value
            if attitude is not None:
                value = np.asarray(attitude, dtype=float)
                if value.shape != (3,):
                    raise ValueError("attitude must have shape (3,)")
                self.state[6:10] = quat_from_euler(*value)
            self.motor_states = np.zeros(4)
        else:
            self.state = state.as_vector()
            self.motor_states = state.motor_speeds.copy()
        self._sim_time = 0.0
        self.last_acceleration.fill(0.0)
        self.last_angular_acceleration.fill(0.0)
        self.last_wrench.fill(0.0)
        for disturbance in self.disturbances:
            disturbance.reset()
            disturbance.modify_dynamics(self, 0.0)
        self._prepare_environment(0.0, 0.0)

    # -- state accessors -------------------------------------------------

    def get_state(self) -> VehicleState:
        return VehicleState.from_vector(
            self.state, motor_speeds=self.motor_states, time=self._sim_time
        )

    def get_position(self) -> NDArray:
        return self.state[:3].copy()

    def get_velocity(self) -> NDArray:
        return self.state[3:6].copy()

    def get_attitude(self) -> NDArray:
        return quat_to_euler(self.state[6:10])

    def get_quaternion(self) -> NDArray:
        return self.state[6:10].copy()

    def get_angular_velocity(self) -> NDArray:
        return self.state[10:13].copy()

    def get_motor_speeds(self) -> NDArray:
        return self.motor_states.copy()

    def rotation_matrix(self) -> NDArray:
        return quat_to_rotation_matrix(self.state[6:10])

    def specific_force_body(self) -> NDArray:
        """Ideal accelerometer specific force at the current state [m/s²]."""
        gravity_up = np.array([0.0, 0.0, self.g])
        return self.rotation_matrix().T @ (self.last_acceleration + gravity_up)

    # -- forces and integration -----------------------------------------

    @property
    def _linear_drag(self) -> NDArray:
        value = np.asarray(self.k_d, dtype=float)
        return np.full(3, float(value)) if value.ndim == 0 else value

    def _prepare_environment(self, t: float, dt: float) -> None:
        self._step_wind = np.zeros(3)
        self._rotor_efficiencies = np.ones(4)
        self._ground_effect_multiplier = 1.0
        for disturbance in self.disturbances:
            advance = getattr(disturbance, "advance", None)
            if advance is not None:
                advance(t, dt, self.state)
            wind_velocity = getattr(disturbance, "wind_velocity", None)
            if wind_velocity is not None:
                self._step_wind += np.asarray(wind_velocity(t), dtype=float)
            rotor_efficiencies = getattr(disturbance, "rotor_efficiencies", None)
            if rotor_efficiencies is not None:
                self._rotor_efficiencies *= np.asarray(
                    rotor_efficiencies(t), dtype=float
                )
            thrust_multiplier = getattr(disturbance, "thrust_multiplier", None)
            if thrust_multiplier is not None:
                self._ground_effect_multiplier *= float(
                    thrust_multiplier(self.state[2])
                )

    def _wrench(self, motor_speeds: NDArray) -> NDArray:
        allocation = self.allocation_matrix(self._rotor_efficiencies)
        wrench = allocation @ np.square(motor_speeds)
        wrench[0] *= self._ground_effect_multiplier
        return wrench

    def _derivatives(
        self,
        state: NDArray,
        motor_speeds: NDArray,
        *,
        t: float | None = None,
        dt: float = 0.01,
    ) -> NDArray:
        wrench = self._wrench(motor_speeds)
        thrust = wrench[0]
        torque = wrench[1:4]
        velocity = state[3:6]
        quaternion = state[6:10]
        body_rates = state[10:13]
        rotation = quat_to_rotation_matrix(quaternion)

        air_velocity_body = rotation.T @ (velocity - self._step_wind)
        drag_body = (
            -self._linear_drag * air_velocity_body
            - self.config.quadratic_drag
            * np.abs(air_velocity_body)
            * air_velocity_body
        )
        thrust_world = rotation @ np.array([0.0, 0.0, thrust])
        drag_world = rotation @ drag_body
        gravity_world = np.array([0.0, 0.0, -self.mass * self.g])
        total_force = thrust_world + drag_world + gravity_world
        total_torque = torque.copy()

        eval_time = self._sim_time if t is None else t
        for disturbance in self.disturbances:
            # Wind disturbances are already represented by relative airspeed.
            if not hasattr(disturbance, "wind_velocity"):
                total_force += disturbance.external_force(eval_time, dt, state)
            total_torque += disturbance.external_torque(eval_time, dt, state)

        acceleration = total_force / self.mass
        angular_acceleration = np.linalg.solve(
            self.I,
            total_torque - np.cross(body_rates, self.I @ body_rates),
        )
        derivative = np.zeros(self.STATE_SIZE)
        derivative[:3] = velocity
        derivative[3:6] = acceleration
        derivative[6:10] = quat_derivative(quaternion, body_rates)
        derivative[10:13] = angular_acceleration
        return derivative

    def update(self, dt: float, motor_speeds: NDArray) -> NDArray:
        """Advance one fixed step with RK4 and an exact motor-lag update."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive number")
        commands = np.asarray(motor_speeds, dtype=float)
        if commands.shape != (4,) or not np.all(np.isfinite(commands)):
            raise ValueError("motor_speeds must be a finite shape-(4,) vector")
        commands = np.clip(commands, self.min_motor_speed, self.max_motor_speed)

        if self.motor_time_constant > 0.0:
            alpha = 1.0 - np.exp(-dt / self.motor_time_constant)
            self.motor_states += alpha * (commands - self.motor_states)
        else:
            self.motor_states = commands.copy()
        self.motor_states = np.clip(
            self.motor_states, self.min_motor_speed, self.max_motor_speed
        )

        for disturbance in self.disturbances:
            disturbance.modify_dynamics(self, self._sim_time)
        self._prepare_environment(self._sim_time, dt)

        t0 = self._sim_time
        motors = self.motor_states
        k1 = self._derivatives(self.state, motors, t=t0, dt=dt)
        k2 = self._derivatives(
            self.state + 0.5 * dt * k1, motors, t=t0 + 0.5 * dt, dt=dt
        )
        k3 = self._derivatives(
            self.state + 0.5 * dt * k2, motors, t=t0 + 0.5 * dt, dt=dt
        )
        k4 = self._derivatives(self.state + dt * k3, motors, t=t0 + dt, dt=dt)
        self.state += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.state[6:10] = quat_normalize(self.state[6:10])
        self._sim_time += dt

        final_derivative = self._derivatives(
            self.state, motors, t=self._sim_time, dt=dt
        )
        self.last_acceleration = final_derivative[3:6].copy()
        self.last_angular_acceleration = final_derivative[10:13].copy()
        wrench = self._wrench(motors)
        self.last_wrench = wrench.copy()
        if not np.all(np.isfinite(self.state)):
            raise FloatingPointError(
                f"quadcopter state became non-finite at t={self._sim_time:.6f}s"
            )
        return self.state.copy()

    # -- actuator model --------------------------------------------------

    def allocation_matrix(
        self, efficiencies: NDArray | None = None
    ) -> NDArray:
        """Map squared motor speeds to collective thrust and body torque."""
        positions = self.config.rotor_positions
        efficiency = (
            np.ones(4)
            if efficiencies is None
            else np.asarray(efficiencies, dtype=float)
        )
        if efficiency.shape != (4,):
            raise ValueError("efficiencies must have shape (4,)")
        force_gain = self.k_f * efficiency
        return np.vstack(
            [
                force_gain,
                positions[:, 1] * force_gain,
                -positions[:, 0] * force_gain,
                self.config.rotor_directions * self.k_m * efficiency,
            ]
        )
