# Simulation Pipeline

**Key files:**
- [src/drones_sim/dynamics/quadcopter.py](../src/drones_sim/dynamics/quadcopter.py)
- [src/drones_sim/sensors/imu.py](../src/drones_sim/sensors/imu.py)
- [src/drones_sim/estimation/ekf.py](../src/drones_sim/estimation/ekf.py)
- [src/drones_sim/control/cascaded.py](../src/drones_sim/control/cascaded.py)
- [src/drones_sim/visualization/plots.py](../src/drones_sim/visualization/plots.py),
  [viewer.py](../src/drones_sim/visualization/viewer.py)

---

## 1. Overview

Simulation is the integration layer that connects all subsystems into a validatable closed loop.
It is the primary vehicle for controller tuning, estimator validation, and architecture evaluation
before any hardware work.

---

## 2. Closed-Loop Architecture

```
                    ┌─────────────────────────────────┐
                    │          Simulation Loop         │
                    │                                  │
  waypoints/refs ──►│  1. Planning / target selection  │
                    │                                  │
                    │  2. QuadcopterDynamics.update()  │◄── motor_speeds (4,)
                    │       x_{k+1} = f(x_k, u_k)     │
                    │                                  │
                    │  3. IMUSimulator / SensorNoise   │
                    │       y = Hx + noise             │
                    │                                  │
                    │  4. EKF/AHRS estimator           │
                    │       x̂_{k|k} = predict+correct  │
                    │                                  │
                    │  5. QuadcopterController.compute()├──► motor_speeds
                    │       u = cascade(target, x̂)     │
                    │                                  │
                    │  6. Log / visualize              │
                    └─────────────────────────────────┘
```

### 2.1 Loop Step Details

| Step | Function | Inputs | Outputs |
|------|----------|--------|---------|
| 1. Reference | Waypoint list or pre-generated traj | Mission spec | `target_pos`, `target_yaw` |
| 2. Dynamics | `QuadcopterDynamics.update(dt, motors)` | `motor_speeds (4,)` | Updated `quad.state (12,)` |
| 3. Sensors | `IMUSimulator.simulate()` or inline `SensorNoiseModel.apply()` | `quad.state` | `accel, gyro, mag` |
| 4. Estimation | `ekf.predict(gyro)` + `ekf.correct_accel(accel)` + `ekf.correct_mag(mag)` | Sensor data | `estimated_state` |
| 5. Control | `controller.compute(target_pos, target_yaw, dt, prev_target)` | `estimated_state` | `motor_speeds (4,)` |
| 6. Logging | Append to arrays; plot or stream to viser | State, motors, estimate | Figures, 3D view |

### 2.2 Typical Fixed Step Size

The loop runs at fixed $\Delta t$ (commonly 0.01 s = 100 Hz). At each step:

1. Dynamics advances: $\mathbf{x}_{k+1} = \mathbf{x}_k + f(\mathbf{x}_k, \mathbf{u}_k)\, \Delta t$
2. Sensor corruption: $\mathbf{y}_k = H\mathbf{x}_{k+1} + \boldsymbol\eta_k$
3. EKF recursion: predict + correct
4. Controller: output $\mathbf{u}_{k+1}$ for next step

All state and measurement arrays are pre-allocated for efficiency.

---

## 3. Example Scripts

| Script | Modes | Demonstrated capability |
|--------|-------|------------------------|
| [examples/01_imu_ekf_basic.py](../examples/01_imu_ekf_basic.py) | Open-loop | `generate_hover_accel_cruise` → `IMUSimulator` → `ExtendedKalmanFilter`; plots position/velocity/quaternion |
| [examples/02_ekf_adaptive.py](../examples/02_ekf_adaptive.py) | Open-loop | Temperature-enabled IMU simulation → `AdaptiveEKF`; shows noise adaptation |
| [examples/03_waypoint_nav.py](../examples/03_waypoint_nav.py) | Closed-loop | Online waypoint targeting; cascaded PID; position+attitude plots |
| [examples/04_viser_viewer.py](../examples/04_viser_viewer.py) | Closed-loop + 3D | Same as 03 + interactive viser playback; true vs filtered path overlay |
| [examples/05_full_pipeline.py](../examples/05_full_pipeline.py) | Full pipeline | Dynamics + noisy sensors + EKF estimation + cascaded PID feedback loop |
| [examples/06_trajectory_following.py](../examples/06_trajectory_following.py) | Closed-loop + 3D | EKF-fused trajectory following on `generate_circular()`; interactive viser GUI with random trajectory button, EKF toggle, and 4-panel tracking-error plot |

### 3.1 Minimal Loop Pattern (from 05_full_pipeline.py)

```python
quad    = QuadcopterDynamics()
noise   = SensorNoiseModel(noise_std=0.05, bias_range=0.1)
ekf     = ExtendedKalmanFilter(dt=0.01)
ctrl    = QuadcopterController(quad)

motors = np.full(4, hover_speed)
for i in range(N):
    # 1. Advance plant
    quad.update(dt, motors)

    # 2. Corrupt sensors
    pos   = quad.get_position()
    vel   = quad.get_velocity()
    accel_true = ...        # from dynamics accel computation
    gyro  = noise.apply(quad.get_angular_velocity())
    accel = noise.apply(accel_true)
    mag   = noise.apply(mag_body)

    # 3 & 4. Estimate
    ekf.predict(gyro)
    ekf.correct_accel(accel)
    ekf.correct_mag(mag)
    state = ekf.get_state()

    # 5. Control on estimated state
    motors = ctrl.compute(waypoints[wp_idx], target_yaw, dt, prev_wp)
```

---

## 4. Visualization

### 4.1 `DroneViewer` (viser — 3D Interactive)

| Capability | Method |
|-----------|--------|
| Add trajectory line (true path) | `add_trajectory(positions, name, color)` |
| Add trajectory line (filtered/estimated) | `add_trajectory(filtered_pos, name='ekf', color=(255,0,0))` |
| Add waypoint spheres | `add_waypoints(waypoints)` |
| Add quadcopter body + rotors | `add_quadcopter_frame(name, arm_length)` |
| Time-slider playback | `play_trajectory(positions, quats, dt)` |

**Best for:** spatial intuition, verifying waypoint coverage, comparing true vs. estimated paths in 3D,
time-scrubbing through crash moments.

Requires: `pip install viser`. Starts a web server (default port 8080); open `http://localhost:8080`.

### 4.2 `plot_ekf_results()` (matplotlib — 2D Analysis)

| Panel | Content |
|-------|---------|
| Position | True vs. filtered $x, y, z$ over time |
| Velocity | True vs. filtered $v_x, v_y, v_z$ over time |
| Orientation | True vs. filtered $[q_w, q_x, q_y, q_z]$ over time |
| Accelerometer | Raw $a_x, a_y, a_z$ readings |
| Gyroscope | Raw $\omega_x, \omega_y, \omega_z$ readings |
| 3D trajectory | Overlay of true and filtered paths |

**Best for:** measuring estimator drift, tuning PID gains, evaluating convergence speed, reading
control-effort magnitude, statistical analysis of tracking errors.

---

## 5. Test Structure

### 5.1 Test Files

| File | Class/function | What is tested |
|------|---------------|----------------|
| [tests/test_math_utils.py](../tests/test_math_utils.py) | `TestQuatOps` | Normalize, multiply (identity), conjugate round-trip, zero-omega derivative |
| [tests/test_dynamics.py](../tests/test_dynamics.py) | `TestQuadDynamics` | Free-fall vertical accel ≈ −g; hover motor speed → net force ≈ 0 |
| [tests/test_pid.py](../tests/test_pid.py) | `TestPID` | Proportional response; output saturation; integral clamp; reset |
| [tests/test_ekf.py](../tests/test_ekf.py) | `TestEKF` | EKF state propagation; identity quaternion init; covariance positive-definite |
| [tests/test_cascaded.py](../tests/test_cascaded.py) | `TestCascadedFlight` | **Regression gate:** controller reaches hover and waypoint targets without plant diverging |

Run all tests:

```bash
cd /Users/thanhndv212/Develop/drones_sim
pytest tests/ -v
```

### 5.2 Closed-Loop Regression Gate

`test_cascaded.py` is the most important test: it runs a short closed-loop simulation and asserts that the
drone converges to a target position within a time budget:

```python
assert np.linalg.norm(quad.get_position() - target) < convergence_threshold
```

This guards against any changes to the dynamics, controller gains, or allocation matrix that cause
instability or divergence.

---

## 6. Simulation Metrics

### 6.1 Tracking Performance

| Metric | Formula | Good threshold |
|--------|---------|---------------|
| RMS position error | $\sqrt{\frac{1}{N}\sum\|\mathbf{p}_k - \hat{\mathbf{p}}_k\|^2}$ | < 0.05 m |
| Waypoint settle time | Time from arrival at approach 1 m to 0.05 m convergence | < 3 s |
| Position overshoot | Peak error after first crossing of target | < 20% of step size |

### 6.2 Estimation Quality

| Metric | Formula |
|--------|---------|
| Position RMSE | $\sqrt{\frac{1}{N}\sum\|\mathbf{p}_\text{true} - \hat{\mathbf{p}}\|^2}$ |
| Attitude RMSE | Angle between true and estimated quaternions |
| Innovation statistics | $\mathbb{E}[\mathbf{y}_k^2]$ should ≈ $S_k$ (diagonal) |

### 6.3 Control Effort

| Metric | Interpretation |
|--------|---------------|
| Motor speed variance | High variance → aggressive / unsmooth control |
| Motor saturation ratio | Fraction of time motors hit $\omega_\text{max}$ |
| $\sum u_k^T R u_k$ | LQR-style cost for benchmarking controllers |

---

## 7. Planned Extensions

### 7.1 Monte Carlo Evaluation

Run $M$ independent trials with different random seeds and disturbance profiles:

```python
results = [run_trial(seed=i) for i in range(M)]
position_errors = np.array([r['pos_rmse'] for r in results])
print(f"Median: {np.median(position_errors):.3f} m, p95: {np.percentile(position_errors, 95):.3f} m")
```

### 7.2 Hardware-in-the-Loop (HIL)

Replace one simulated component at a time with real hardware/firmware:

- **Sensor HIL:** Stream real IMU data; keep simulated dynamics and estimator
- **Controller HIL:** Run real flight controller firmware; use simulated plant and sensors
- **Full HIL:** Simulated dynamics only; all other components are real

This validates timing assumptions, I/O interfaces, and latency budgets before arming hardware.

### 7.3 SITL Interface

Add a UDP adapter so PX4 or ArduPilot SITL can drive `QuadcopterDynamics` as its physics backend:

```
PX4 SITL ──MAVLink UDP──► sim_adapter.py ──► QuadcopterDynamics.update()
                                        ◄── state broadcast
```

### 7.4 Disturbance and Failure Campaigns

Scripted test modes:

| Campaign | Disturbance injected |
|----------|---------------------|
| Wind gust | Step/sinusoidal world-frame force |
| Sensor dropout | Zero IMU for 0.5 s |
| Motor degradation | Reduce one rotor to 50% thrust |
| Payload change | Double mass mid-flight |

### 7.5 Real-Time Profiling

Add timing instrumentation per loop stage:

```python
import time
t0 = time.perf_counter()
quad.update(dt, motors)
dynamics_ms = (time.perf_counter() - t0) * 1000
```

Budget allocation target for 100 Hz loop (10 ms/step):

| Stage | Budget |
|-------|--------|
| Dynamics | < 0.5 ms |
| Sensors | < 0.3 ms |
| EKF | < 2.0 ms |
| Controller | < 1.0 ms |
| Logging/viz | < 3.0 ms |

---

## 8. Related External Tools

| Tool | Use |
|------|-----|
| Gazebo + ROS | Full robot simulation with physics engine, URDF models |
| MATLAB/Simulink | Model-based design, automatic control law generation |
| PX4 SITL | Autopilot-in-the-loop simulation with MAVLink |
| ArduPilot SITL | Same with ArduPilot firmware |
| Webots | Python-friendly robotics simulator |

---

## 9. Application Areas

This simulator is suitable for prototyping and evaluating algorithms for:

- Aerial photography and cinematography (controller smoothness)
- Inspection and mapping (waypoint coverage accuracy)
- Agriculture (coverage path planning)
- Search and rescue (robust estimation under GPS denial)
- Autonomous delivery prototyping (wind disturbance handling)


