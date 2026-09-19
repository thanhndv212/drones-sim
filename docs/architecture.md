# Architecture

`drones_sim` uses one explicit data contract across the full flight stack:

```text
TrajectoryData → TrajectorySetpoint → Controller → ControlOutput
                                             │            │
                                             │            ▼
                                    VehicleState ← Dynamics
                                             │
                               IMU/GPS measurements
                                             │
                                             ▼
                                      Navigation EKF
                                             │
                                             ▼
                                      SimulationResult
                                      ├─ dashboard
                                      ├─ 3D playback
                                      └─ metrics/logging
```

## Conventions

- World frame: ENU, with positive z upward.
- Body frame: front-right-up.
- Quaternion: Hamilton `[w, x, y, z]`, rotating body vectors into world.
- Angular velocity and torque: body frame.
- Thrust: positive body z.
- Accelerometer: positive specific force, `R.T @ (a_world + gravity_up)`.

`VehicleState`, `TrajectorySetpoint`, and `ControlOutput` validate shape and
finiteness at module boundaries. The numerical plant still exposes its 13-state
array for backwards compatibility.

## Modeling

`QuadcopterConfig` is the authoritative physical configuration. The plant uses
quaternion Newton-Euler dynamics, RK4 integration, exact discrete first-order
motor lag, motor speed limits, per-rotor efficiency, and anisotropic linear plus
quadratic drag relative to wind. Stochastic disturbances advance once per outer
step, never once per RK4 stage.

## Control

`GeometricController` is the default high-level controller. Position and
velocity errors produce a desired world-frame force; force direction and yaw
define a desired rotation on SO(3). Attitude error is computed directly from
rotation matrices. `ControlAllocator` solves bounded weighted least squares in
squared-rotor-speed space and reports both the achieved wrench and saturation.

The original cascaded PID and hover LQR remain available for comparison and for
existing callers.

## Estimation

`ExtendedKalmanFilter` retains the 16-state navigation model and analytical
quaternion measurement Jacobians. All measurement corrections share one
Joseph-form update path with normalized innovation squared (NIS) rejection.
Joint GPS position/velocity updates avoid sequentially double-counting one fix.
Acceptance counts and the latest NIS are exposed through `EKFDiagnostics`.
Accelerometer-as-gravity correction is opt-in in the high-level simulator,
because it is only observable during quasi-static flight; specific force is
always used for inertial propagation.

## Simulation and visualization

`ClosedLoopSimulator` aligns reference, truth, estimate, commands, and allocator
status into a `SimulationResult`. Random generators are local and seeded, so one
sensor no longer changes another module's random stream. `plot_simulation`
creates a five-panel diagnostic dashboard; `DroneViewer.playback_result` consumes
the same result directly for interactive 3D playback.

The default Rerun backend consumes the same result and records a synchronized
3D flight scene, position/error/motor time series, control telemetry, saturation
events, metadata, and summary metrics. `visualize_rerun` supports a spawned local
viewer, gRPC streaming, and portable `.rrd` output. The backend-neutral
`visualize(result)` entry point selects Rerun unless `backend="matplotlib"` or
`backend="viser"` is requested. Viser is retained as an optional extra for its
custom mission-control widgets.

## Compatibility

The legacy array accessors, `QuadcopterController`, LQR interface, examples, and
RL environment remain supported. New code should use the typed contracts and
the high-level simulation runner.
