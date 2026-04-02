# Drone Simulation — Documentation Index

This repository implements a full-stack quadcopter simulation covering rigid-body dynamics,
sensor modeling, state estimation, trajectory planning, cascade control, and visualization.
The six documents below form the complete technical reference, written at wiki depth with
LaTeX equations, implementation status markers (✅ implemented, ⚠️ partial, 🗺️ planned),
and concrete pointers to source files.

## Suggested Reading Order

| Step | Document | What it covers |
|------|----------|----------------|
| 1 | [Modeling](modeling.md) | Rigid-body dynamics, rotation convention, allocation matrix, math utilities |
| 2 | [Sensors](sensors.md) | Noise stack, temperature drift, IMU data flow, signal processing |
| 3 | [State Estimation](state-estimation.md) | EKF, AHRS, AdaptiveEKF — full predict/update equations and Jacobians |
| 4 | [Planning](planning.md) | Trajectory generators, differential flatness, minimum-snap, RRT* |
| 5 | [Control](control.md) | Cascade PID, gain table, LQR/MPC/SMC/geometric SE(3) families |
| 6 | [Simulation](simulation.md) | Full closed-loop pipeline, visualization, metrics, planned extensions |
| 7 | [Testing](testing.md) | Validation strategy, Jacobian tests, regression gates, implementation roadmap |

## Module Map

```
src/drones_sim/
├── math_utils.py          ← rotation, quaternion, axis-angle utilities
├── trajectory.py          ← HoverAccelCruise, CircularOrbit, WaypointTrajectory
├── dynamics/
│   └── quadcopter.py      ← QuadcopterDynamics (Newton–Euler, Euler integration)
├── sensors/
│   ├── imu.py             ← IMUSimulator (accel + gyro with bias, noise, temp)
│   └── models.py          ← SensorNoiseModel, TemperatureModel
├── estimation/
│   ├── ekf.py             ← ExtendedKalmanFilter (10-state), AdaptiveEKF (9-state)
│   └── ahrs.py            ← AHRS (complementary cross-product feedback)
├── control/
│   ├── pid.py             ← PIDController (discrete, anti-windup)
│   └── cascaded.py        ← QuadcopterController (9 PID loops, allocation)
└── visualization/
    ├── viewer.py           ← DroneViewer (viser 3-D interactive)
    └── plots.py            ← plot_ekf_results (6-panel matplotlib)
```

## Examples

| File | What it demonstrates |
|------|----------------------|
| `examples/01_imu_ekf_basic.py` | 10-state EKF fusing IMU + magnetometer |
| `examples/02_ekf_adaptive.py` | AdaptiveEKF under temperature-dependent noise |
| `examples/03_waypoint_nav.py` | Cascaded control with WaypointTrajectory |
| `examples/04_viser_viewer.py` | 3-D interactive DroneViewer |
| `examples/05_full_pipeline.py` | Full closed loop: dynamics → sensors → EKF → controller |

## Tests

```
tests/
├── test_math_utils.py   ← quaternion identities, rotation matrix orthogonality
├── test_dynamics.py     ← free-fall acceleration, hover equilibrium
├── test_pid.py          ← P/I/D term isolation, anti-windup clamping
└── test_ekf.py          ← EKF covariance propagation, initialization
```

