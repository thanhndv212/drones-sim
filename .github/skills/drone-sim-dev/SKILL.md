---
name: drone-sim-dev
description: "Develop, review, tune, and visualize quadcopter drone simulations. Use when: building simulation components (dynamics, sensors, estimation, control), tuning PID gains for trajectory tracking, adding new sensor models or disturbances, visualizing 3D trajectories in viser, reviewing architecture against state-of-art, running the full closed-loop pipeline, committing changes via git."
argument-hint: "Describe what you want to build, tune, review, or visualize in the drone sim"
---

# Drone Simulation Development

## When to Use

- Build or extend simulation components (dynamics, sensors, estimation, control, visualization)
- Review and update core components against state-of-the-art drone simulation practices
- Tune cascaded PID controller for trajectory tracking performance
- Visualize trajectories and playback in 3D via viser
- Add new features: sensor types, wind disturbances, multi-drone, ROS integration
- Track changes with git after each milestone

## Architecture Overview

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  Trajectory   │────▶│   Dynamics    │────▶│   Sensors    │────▶│   Estimation          │
│  (reference)  │     │ QuadcopterDyn│     │ IMUSimulator │     │ EKF / AdaptiveEKF     │
└──────────────┘     └──────┬───────┘     └──────────────┘     │ AHRS                  │
                            │                                   └──────────┬────────────┘
                            │                                              │
                     ┌──────▼───────┐                            ┌────────▼────────┐
                     │   Control     │◀───────────────────────────│  Estimated State │
                     │ CascadedPID   │                            └─────────────────┘
                     └──────┬───────┘
                            │
                     ┌──────▼───────┐
                     │ Visualization │
                     │ plots / viser │
                     └──────────────┘
```

**Source layout:** `src/drones_sim/`

| Module | Path | Purpose |
|--------|------|---------|
| Math utils | `math_utils.py` | Quaternion ops (wxyz), rotation matrices, Euler conversions |
| Trajectory | `trajectory.py` | Hover-accel-cruise, circular, waypoint generators |
| Dynamics | `dynamics/quadcopter.py` | 12-state Newton-Euler rigid body, Euler integration |
| Sensors | `sensors/imu.py`, `sensors/models.py` | IMU noise model (bias + scale + Gaussian + temp) |
| Estimation | `estimation/ekf.py`, `estimation/ahrs.py` | 10-state EKF, 9-state Adaptive EKF, complementary AHRS |
| Control | `control/pid.py`, `control/cascaded.py` | Scalar PID with anti-windup, 3-loop cascaded controller |
| Visualization | `visualization/plots.py`, `visualization/viewer.py` | Matplotlib plots, viser 3D viewer with playback |

## Procedures

### 1. Developing Simulation Components

**Dynamics — expanding the quadcopter model:**

1. Read `src/drones_sim/dynamics/quadcopter.py` to understand the 12-state model:
   - State: `[x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r]`
   - Motor layout: X-config with 4 rotors, allocation matrix maps ω² → [T, τ_φ, τ_θ, τ_ψ]
   - Integration: forward Euler (consider upgrading to RK4 for stiff dynamics)
2. Physical params: mass, arm_length, inertia (3×3 diagonal), k_f, k_m, k_d, g
3. When adding new effects (wind, ground effect, rotor drag):
   - Add force/torque terms in `update()` after `# External forces`
   - Keep the allocation matrix consistent — only modify if changing motor config
   - Add parameters to `__init__` with sensible defaults
4. Run `tests/test_dynamics.py` after changes

**Sensors — adding new sensor models:**

1. Read `sensors/models.py` for the `SensorNoiseModel` and `TemperatureModel` patterns
2. Read `sensors/imu.py` for the `IMUSimulator` and `IMUConfig` dataclass
3. To add a new sensor (GPS, barometer, optical flow):
   - Create a new dataclass for config (like `IMUConfig`)
   - Create a simulator class with `simulate(traj: TrajectoryData)` method
   - Noise stack per axis: `true_value × scale_factor + bias + N(0, σ²) + temp_offset`
4. Register in `sensors/__init__.py`

**Estimation — extending EKF:**

1. Two estimation modes exist:
   - **Full-state EKF** (10-state): `[position(3), velocity(3), quaternion(4)]` — analytical Jacobians
   - **Adaptive EKF** (9-state): `[position(3), velocity(3), accel_bias(3)]` + separate AHRS for attitude
2. To add new measurement sources:
   - Define `h(x)` measurement model and `H` Jacobian in EKF class
   - Add measurement noise `R` for the new sensor
   - Call update step with new measurement data
3. For GPS fusion: add position measurement model `h(x) = x[0:3]` with H = [I₃, 0, ...]
4. Run `tests/test_ekf.py` after changes

**Trajectory — adding trajectory types:**

1. Read `trajectory.py` for the `TrajectoryData` dataclass structure:
   - Fields: `t, position, velocity, acceleration, orientation_quat, angular_velocity`
2. All generators return `TrajectoryData`
3. Existing types: `generate_hover_accel_cruise()`, `generate_circular()`, `generate_waypoint_trajectory()`
4. New trajectories must compute consistent derivatives (velocity = d/dt position, etc.)

### 2. Reviewing and Updating Components

**State-of-the-art checklist:**

| Area | Current | State-of-Art Improvement |
|------|---------|-------------------------|
| Integration | Forward Euler | RK4 or semi-implicit Euler for stability |
| Quaternion | Euler angles in dynamics state | Full quaternion state avoids gimbal lock |
| Aerodynamics | Linear drag only | Blade element theory, ground effect, rotor wash |
| Wind | Not modeled | Dryden/von Kármán turbulence model |
| Motor dynamics | Instantaneous response | First-order lag: τ·ω̇ + ω = ω_cmd |
| Battery | Not modeled | Voltage-dependent thrust limits |
| Sensors | IMU only | GPS, barometer, optical flow, camera |
| EKF | Separate attitude/position | Unified 16-state: [p, v, q, b_a, b_g] |
| Control | Cascaded PID | Geometric control on SE(3), INDI, L1 adaptive |
| Multi-drone | Not supported | Swarm with collision avoidance |

**Review procedure:**

1. Pick one area from the checklist above
2. Read the relevant source files
3. Implement the improvement:
   - Keep backward-compatible API (add new params with defaults)
   - Add tests for new functionality
   - Update examples if behavior changes
4. Validate with existing tests: `pytest tests/`
5. Commit with descriptive message

### 3. Tuning PID Controller for Trajectory Tracking

**Controller architecture** (cascaded 3-loop in `control/cascaded.py`):

```
Position PID → Velocity PID → Attitude PID → Motor Mixing
```

**Default gains and limits:**

| Loop | Axis | kp | ki | kd | Output limits | Windup limits |
|------|------|----|----|----|---------------|---------------|
| Position | x, y | 6.0 | 1.2 | 5.0 | (-π/2.5, π/2.5) | (-5, 5) |
| Position | z | 22.0 | 6.0 | 12.0 | (0, 35) | (-20, 20) |
| Velocity | x, y | 4.0 | 0.8 | 1.5 | (-π/3, π/3) | (-3, 3) |
| Velocity | z | 6.0 | 1.2 | 2.5 | (0, 30) | (-10, 10) |
| Attitude | roll, pitch | 28.0 | 3.0 | 10.0 | (-10, 10) | (-6, 6) |
| Attitude | yaw | 12.0 | 1.5 | 5.0 | (-5, 5) | (-4, 4) |

**Additional tunable parameters in `compute()`:**
- Feed-forward gain: `ff_gain = 0.8 * (0.5 + 0.5 * dist_factor)` — controls how much target velocity is fed forward
- Acceleration feed-forward: scaled by `0.2` — damps aggressive prediction
- Proportional navigation blend: `blend = min(1.0, err_mag)` — blends approach vector vs PID output
- Hover thrust compensation: `hover_thrust = g * mass * k_f` — keeps drone airborne at zero velocity error

**Step-by-step tuning procedure (inner-to-outer):**

**Step 1 — Attitude loop** (fastest, ~50-100 Hz effective bandwidth):
1. In `examples/03_waypoint_nav.py`, set a single waypoint at hover position (no translation)
2. Add an initial attitude perturbation: `quad.state[6:8] = [0.1, 0.1]` (10° roll/pitch)
3. Tune `roll_ctrl` and `pitch_ctrl`:
   - Set ki=0, kd=0, increase kp from 10 up — watch for oscillation threshold
   - Back off kp by 30%, then add kd ≈ 0.3× kp for damping
   - Add ki ≈ 0.1× kp if steady-state offset remains
4. Verify: attitude should settle within 0.3s with < 5% overshoot
5. Similarly tune `yaw_ctrl` — can be 2-3× slower (lower kp)

**Step 2 — Velocity loop** (~15-30 Hz effective bandwidth):
1. Set a waypoint that requires constant velocity (e.g., ramp displacement)
2. Tune `vx_ctrl`, `vy_ctrl`:
   - Output limits constrain desired roll/pitch angles — keep within (-π/3, π/3)
   - kp drives responsiveness, kd smooths transitions
3. Tune `vz_ctrl`:
   - Output feeds directly to thrust — output limits (0, 30) prevent negative thrust
   - Verify hover thrust compensation is correct for your mass/k_f

**Step 3 — Position loop** (~3-10 Hz effective bandwidth):
1. Set multiple waypoints with varying distances
2. Tune `x_ctrl`, `y_ctrl`:
   - Output limits (-π/2.5, π/2.5) cap the desired velocity angles
   - Higher kp → faster waypoint acquisition, but can cause overshoot
3. Tune `z_ctrl`:
   - Output limits (0, 35) — z-loop is asymmetric (can't push down, only reduce thrust)
   - kp=22 is aggressive — reduce if altitude oscillates

**Step 4 — Feed-forward and navigation tuning:**
1. Adjust `ff_gain` scaling (currently `0.8`) — reduce for smoother but slower tracking
2. Adjust acceleration feed-forward coefficient (currently `0.2`) — higher = more predictive
3. Adjust proportional navigation blend — disable by setting `blend = 0` for pure PID

**Validation script:**

```python
import numpy as np
from drones_sim.dynamics.quadcopter import QuadcopterDynamics
from drones_sim.control.cascaded import QuadcopterController

quad = QuadcopterDynamics()
quad.reset(position=np.array([0.0, 0.0, 1.0]))
ctrl = QuadcopterController(quad)

waypoints = [(0, [0, 0, 1]), (3, [2, 0, 1.5]), (6, [2, 2, 1]), (9, [0, 0, 1])]
dt = 0.002
errors = []

for i in range(int(9 / dt)):
    t = i * dt
    # Find current target waypoint
    target = waypoints[0][1]
    for wt, wp in waypoints:
        if t >= wt:
            target = wp
    target = np.array(target, dtype=float)
    
    motors = ctrl.compute(target, 0.0, dt)
    quad.update(dt, motors)
    errors.append(np.linalg.norm(quad.get_position() - target))

rms_error = np.sqrt(np.mean(np.array(errors[-1000:])**2))
print(f"Steady-state RMS error: {rms_error:.4f} m")
# Target: < 0.1 m for good tracking
```

**Diagnostics:**

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Slow convergence to waypoint | Position kp too low | Increase x/y/z_ctrl kp |
| Overshoot at waypoints | Position kp too high, kd too low | Reduce kp by 20%, increase kd |
| Oscillation in hover | Attitude loop underdamped | Increase roll/pitch_ctrl kd |
| Altitude drops on lateral move | Thrust coupling | Increase z_ctrl kp, verify hover_thrust |
| Spiral divergence | Velocity loop instability | Reduce vx/vy_ctrl kp, check output limits |
| Steady-state offset (constant wind) | No integral action fighting disturbance | Increase ki, widen windup_limits |
| Motor speed spikes | Output limits too wide | Tighten attitude output_limits |
| Jerky motion | Feed-forward too aggressive | Reduce ff_gain or accel coefficient |
| Yaw drift during maneuver | Yaw loop too slow | Increase yaw_ctrl kp |
| Controller resets near target | convergence_threshold too large | Reduce `self.convergence_threshold` |

### 4. Visualizing in 3D with Viser

**Quick start:**

```python
from drones_sim.visualization.viewer import DroneViewer

viewer = DroneViewer(host="0.0.0.0", port=8080)
viewer.add_trajectory(positions, name="actual", color=(0, 0, 255))
viewer.add_trajectory(target_positions, name="target", color=(255, 0, 0))
viewer.add_waypoints(waypoints, name="waypoints")
viewer.add_quadcopter_frame("drone", arm_length=0.2)
viewer.playback(t, positions, rotation_matrices, waypoints=waypoints)
```

**Viewer capabilities:**

- `add_trajectory(positions, name, color, line_width)` — point cloud path
- `add_waypoints(waypoints_list, name)` — sphere markers at goals
- `add_quadcopter_frame(name, arm_length)` — body + 4 rotor spheres
- `update_quadcopter_pose(frame_name, position, rotation_matrix)` — live update
- `playback(t, positions, rotation_matrices, ...)` — time slider on web UI

**Access at:** `http://localhost:8080/`

**Adding custom visuals:**

1. Use `viewer.server` (viser.ViserServer) for raw API access
2. Add meshes via `viewer.server.scene.add_mesh()`
3. For real-time streaming, call `update_quadcopter_pose()` in simulation loop

**Example workflow — visualizing full pipeline:**

See `examples/04_viser_viewer.py` for the reference pattern:
1. Generate waypoints and build dynamics + controller
2. Run simulation loop collecting positions and rotation matrices
3. Create viewer and call `playback()`

### 5. Git Workflow for Changes

**Commit conventions:**

```
<type>(<scope>): <description>

Types: feat, fix, tune, refactor, test, docs
Scopes: dynamics, sensors, estimation, control, viz, trajectory, examples
```

**Examples:**
- `feat(dynamics): add Dryden wind turbulence model`
- `tune(control): adjust z-axis PID gains for hover stability`
- `fix(estimation): correct EKF Jacobian for magnetometer update`
- `test(sensors): add GPS simulator unit tests`

**Milestone workflow:**

1. Create a feature branch: `git checkout -b feat/<feature-name>`
2. Implement changes following the procedures above
3. Run tests: `pytest tests/ -v`
4. Run an example to validate end-to-end: `python examples/05_full_pipeline.py`
5. Stage and commit: `git add -p && git commit`
6. Push when ready: `git push origin feat/<feature-name>`

## Reference: Key Data Structures

**Quaternion convention:** `[w, x, y, z]` (scalar-first) — all of `math_utils.py` uses this. Scipy uses `[x, y, z, w]`, so convert at boundaries.

**State vectors:**
- Dynamics: 12D `[x, y, z, vx, vy, vz, φ, θ, ψ, p, q, r]`
- Full EKF: 10D `[px, py, pz, vx, vy, vz, qw, qx, qy, qz]`
- Adaptive EKF: 9D `[px, py, pz, vx, vy, vz, bax, bay, baz]`

**TrajectoryData fields:** `t(N,)`, `position(N,3)`, `velocity(N,3)`, `acceleration(N,3)`, `orientation_quat(N,4)`, `angular_velocity(N,3)`

**Motor ordering (X-config, top view):**
```
    1(front/+X)
4(left/-Y)  2(right/+Y)
    3(back/-X)
```

## Reference: Running Tests and Examples

```bash
# From project root
pip install -e ".[dev]"       # Install with dev deps
pytest tests/ -v              # Run all tests
pytest tests/test_dynamics.py # Run specific test

# Examples (from project root or examples/)
python examples/01_imu_ekf_basic.py     # EKF basics
python examples/02_ekf_adaptive.py      # Adaptive EKF with temperature
python examples/03_waypoint_nav.py      # Waypoint navigation
python examples/04_viser_viewer.py      # 3D viser visualization
python examples/05_full_pipeline.py     # Full closed-loop pipeline
```
