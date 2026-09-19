"""High-level closed-loop orchestration for reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .control import GeometricController
from .dynamics import QuadcopterDynamics
from .estimation import ExtendedKalmanFilter
from .math_utils import quat_to_euler
from .sensors import GPSConfig, GPSSimulator, IMUConfig, IMUSimulator
from .state import TrajectorySetpoint, VehicleState
from .trajectory import TrajectoryData


@dataclass
class SimulationConfig:
    """Timing, sensor, and reproducibility settings."""

    dt: float = 0.01
    use_estimator: bool = True
    gps_rate: float = 10.0
    seed: int = 0
    # Gravity-as-measurement updates are valid only when translational
    # acceleration is negligible. IMU acceleration is still used for INS
    # propagation; opt in to this correction for quasi-static scenarios.
    correct_accelerometer: bool = False
    correct_magnetometer: bool = True

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.gps_rate <= 0.0:
            raise ValueError("gps_rate must be positive")


@dataclass
class SimulationResult:
    """Aligned truth, estimate, reference, actuation, and filter diagnostics."""

    time: NDArray
    position: NDArray
    velocity: NDArray
    quaternion: NDArray
    body_rates: NDArray
    estimated_position: NDArray
    estimated_velocity: NDArray
    estimated_quaternion: NDArray
    reference_position: NDArray
    reference_velocity: NDArray
    motor_speeds: NDArray
    commanded_wrench: NDArray
    allocation_saturated: NDArray
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def tracking_error(self) -> NDArray:
        return self.position - self.reference_position

    @property
    def estimation_error(self) -> NDArray:
        return self.estimated_position - self.position

    def summary(self) -> dict[str, float]:
        tracking_norm = np.linalg.norm(self.tracking_error, axis=1)
        estimation_norm = np.linalg.norm(self.estimation_error, axis=1)
        return {
            "duration_s": float(self.time[-1] - self.time[0]) if len(self.time) else 0.0,
            "tracking_rmse_m": float(np.sqrt(np.mean(tracking_norm**2))),
            "tracking_max_m": float(np.max(tracking_norm)),
            "estimation_rmse_m": float(np.sqrt(np.mean(estimation_norm**2))),
            "motor_peak_rad_s": float(np.max(self.motor_speeds)),
            "allocation_saturation_fraction": float(
                np.mean(self.allocation_saturated)
            ),
        }


class ClosedLoopSimulator:
    """Own the full reference → control → plant → sensors → estimate loop."""

    def __init__(
        self,
        vehicle: QuadcopterDynamics | None = None,
        controller=None,
        estimator: ExtendedKalmanFilter | None = None,
        imu: IMUSimulator | None = None,
        gps: GPSSimulator | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.vehicle = vehicle or QuadcopterDynamics(motor_time_constant=0.04)
        self.controller = controller or GeometricController(self.vehicle)
        self.estimator = estimator
        self.imu = imu or IMUSimulator(
            IMUConfig(
                accel_scale=(1.0, 1.0),
                gyro_scale=(1.0, 1.0),
                mag_scale=(1.0, 1.0),
            ),
            seed=self.config.seed,
        )
        self.gps = gps or GPSSimulator(
            GPSConfig(position_noise_std=0.5, update_rate=self.config.gps_rate),
            seed=self.config.seed + 1,
        )

    def _reset_estimator(self, initial: VehicleState) -> None:
        if not self.config.use_estimator:
            return
        if self.estimator is None:
            initial_filter_state = np.zeros(16)
            initial_filter_state[:3] = initial.position
            initial_filter_state[3:6] = initial.velocity
            initial_filter_state[6:10] = initial.quaternion
            self.estimator = ExtendedKalmanFilter(
                self.config.dt,
                initial_state=initial_filter_state,
                gravity=np.array([0.0, 0.0, self.vehicle.g]),
                mag_ref=self.imu.cfg.mag_field_ref,
            )
        else:
            self.estimator.x.fill(0.0)
            self.estimator.x[:3] = initial.position
            self.estimator.x[3:6] = initial.velocity
            self.estimator.x[6:10] = initial.quaternion
            self.estimator.P = np.eye(self.estimator.n) * 0.01
            self.estimator.P[0:3, 0:3] *= 0.01
            self.estimator.P[3:6, 3:6] *= 0.1
            self.estimator.P[6:10, 6:10] *= 0.001
            self.estimator.P[13:16, 13:16] = np.eye(3) * 0.0001
            self.estimator.diagnostics = type(self.estimator.diagnostics)()
            self.estimator._rejection_streak.clear()

    def run(
        self,
        reference: TrajectoryData,
        *,
        initial_state: VehicleState | None = None,
    ) -> SimulationResult:
        """Run one deterministic simulation aligned to ``reference.t``."""
        if len(reference.t) < 2:
            raise ValueError("reference must contain at least two samples")
        sample_dt = np.diff(reference.t)
        if not np.allclose(sample_dt, self.config.dt, rtol=2e-2, atol=1e-8):
            raise ValueError(
                "reference sample period must match SimulationConfig.dt; "
                f"got median {np.median(sample_dt):.6f}s vs {self.config.dt:.6f}s"
            )
        if initial_state is None:
            initial_state = VehicleState(
                position=reference.position[0],
                velocity=reference.velocity[0],
                quaternion=reference.orientation_quat[0],
            )
        self.vehicle.reset(state=initial_state)
        self.imu.reset(self.config.seed)
        self.gps.reset(self.config.seed + 1)
        hover_speed = np.sqrt(
            self.vehicle.mass * self.vehicle.g / (4.0 * self.vehicle.k_f)
        )
        self.vehicle.motor_states[:] = hover_speed
        self.controller.reset()
        self._reset_estimator(initial_state)

        count = len(reference.t)
        shape3 = (count, 3)
        position = np.zeros(shape3)
        velocity = np.zeros(shape3)
        quaternion = np.zeros((count, 4))
        body_rates = np.zeros(shape3)
        estimated_position = np.zeros(shape3)
        estimated_velocity = np.zeros(shape3)
        estimated_quaternion = np.zeros((count, 4))
        motor_speeds = np.zeros((count, 4))
        commanded_wrench = np.zeros((count, 4))
        allocation_saturated = np.zeros(count, dtype=bool)
        gps_period = max(1, int(round(1.0 / (self.config.gps_rate * self.config.dt))))

        for index in range(count):
            true_state = self.vehicle.get_state()
            measurement = self.imu.step(
                true_state,
                self.vehicle.last_acceleration,
                self.config.dt,
                duration=float(reference.t[-1]),
            )

            control_state = true_state
            if self.config.use_estimator:
                assert self.estimator is not None
                self.estimator.predict(
                    measurement.angular_velocity, measurement.acceleration
                )
                if self.config.correct_accelerometer:
                    self.estimator.correct_accel(measurement.acceleration)
                if self.config.correct_magnetometer:
                    self.estimator.correct_mag(measurement.magnetic_field)
                if index % gps_period == 0:
                    gps_position, gps_velocity, valid = self.gps.step(
                        true_state.position, true_state.velocity
                    )
                    if valid:
                        gps_covariance = np.diag(
                            [self.gps.config.position_noise_std**2] * 3
                            + [self.gps.config.velocity_noise_std**2] * 3
                        )
                        self.estimator.correct_gps(
                            gps_position, gps_velocity, gps_covariance
                        )
                corrected_rates = (
                    measurement.angular_velocity - self.estimator.x[13:16]
                )
                control_state = self.estimator.get_vehicle_state(
                    corrected_rates, time=true_state.time
                )

            yaw = float(quat_to_euler(reference.orientation_quat[index])[2])
            setpoint = TrajectorySetpoint(
                position=reference.position[index],
                velocity=reference.velocity[index],
                acceleration=reference.acceleration[index],
                yaw=yaw,
                yaw_rate=float(reference.angular_velocity[index, 2]),
            )
            if hasattr(self.controller, "compute_output"):
                output = self.controller.compute_output(
                    setpoint, self.config.dt, state=control_state
                )
                command = output.motor_speeds
                commanded_wrench[index] = np.concatenate(
                    [[output.thrust], output.torque]
                )
                allocation_saturated[index] = output.saturated
            else:
                previous = reference.position[max(0, index - 1)]
                command = self.controller.compute(
                    setpoint.position, setpoint.yaw, self.config.dt, previous
                )
                commanded_wrench[index] = (
                    self.vehicle.allocation_matrix() @ np.square(command)
                )

            position[index] = true_state.position
            velocity[index] = true_state.velocity
            quaternion[index] = true_state.quaternion
            body_rates[index] = true_state.body_rates
            estimated_position[index] = control_state.position
            estimated_velocity[index] = control_state.velocity
            estimated_quaternion[index] = control_state.quaternion
            motor_speeds[index] = command
            self.vehicle.update(self.config.dt, command)

        metadata: dict[str, object] = {
            "dt": self.config.dt,
            "controller": type(self.controller).__name__,
            "estimator": (
                type(self.estimator).__name__ if self.config.use_estimator else "truth"
            ),
            "seed": self.config.seed,
        }
        if self.estimator is not None:
            metadata["ekf_accepted"] = dict(self.estimator.diagnostics.accepted)
            metadata["ekf_rejected"] = dict(self.estimator.diagnostics.rejected)
        return SimulationResult(
            time=reference.t.copy(),
            position=position,
            velocity=velocity,
            quaternion=quaternion,
            body_rates=body_rates,
            estimated_position=estimated_position,
            estimated_velocity=estimated_velocity,
            estimated_quaternion=estimated_quaternion,
            reference_position=reference.position.copy(),
            reference_velocity=reference.velocity.copy(),
            motor_speeds=motor_speeds,
            commanded_wrench=commanded_wrench,
            allocation_saturated=allocation_saturated,
            metadata=metadata,
        )
