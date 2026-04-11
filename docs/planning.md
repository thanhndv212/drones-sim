# Trajectory Planning

**Implementation:** [src/drones_sim/trajectory.py](../src/drones_sim/trajectory.py)

---

## 1. Overview

Planning converts mission goals (waypoints, hover positions, circular paths) into time-parameterized
reference trajectories that the cascaded controller can track. The current planning is purely
**reference generation** — no obstacle awareness and no dynamic feasibility guarantees beyond explicit
velocity/acceleration parameterization.

The common output is a `TrajectoryData` struct consumed by all downstream modules (IMU simulator,
controller, visualizer).

---

## 2. TrajectoryData — Common Interface

```python
@dataclass
class TrajectoryData:
    t: NDArray              # (N,) time vector [s]
    position: NDArray       # (N, 3) world-frame [x, y, z] [m]
    velocity: NDArray       # (N, 3) world-frame [vx, vy, vz] [m/s]
    acceleration: NDArray   # (N, 3) world-frame [ax, ay, az] [m/s²]
    orientation_quat: NDArray  # (N, 4) Hamilton [w, x, y, z]
    angular_velocity: NDArray  # (N, 3) body-frame [p, q, r] [rad/s]
```

All generators must provide consistent derivatives: $\mathbf{v} = \dot{\mathbf{p}}$, $\mathbf{a} = \dot{\mathbf{v}}$.
Angular velocity is derived from quaternion differences (not hand-authored) to maintain kinematic consistency:

$$
\boldsymbol\omega_k = \frac{1}{\Delta t}\, \log\!\left(\mathbf{q}_{k-1}^{-1} \otimes \mathbf{q}_k\right) \cdot 2
$$

In code:

```python
r_diff = R.from_quat(q_curr_xyzw) * R.from_quat(q_prev_xyzw).inv()
angular_velocity[i] = r_diff.as_rotvec() / dt
```

The `as_rotvec()` output is the rotation vector $\boldsymbol\phi = \theta\hat{\mathbf{n}}$ with magnitude
equal to rotation angle; dividing by $\Delta t$ gives body angular rate.

---

## 3. Implemented Trajectory Generators

### 3.1 Hover-Accel-Cruise — `generate_hover_accel_cruise()`

A five-phase test trajectory:

```
Phase 1: Hover         (t = 0 → hover_time)
Phase 2: Accelerate    (t += accel_time)   — constant accel in +x
Phase 3: Cruise        (t += cruise_time)  — constant vx + sinusoidal z oscillation
Phase 4: Decelerate    (t += decel_time)   — constant decel in +x
Phase 5: Final hover   (remainder)         — hold last position
```

**Kinematics:**

Phase 2 (acceleration, starting at $t = 0$):

$$
a_x = \frac{v_{\max}}{t_\text{accel}},\quad v_x(t) = a_x t,\quad x(t) = \frac{1}{2} a_x t^2
$$

Phase 3 (cruise):

$$
v_x = v_\max,\quad x(t) = x_0 + v_\max t
$$

$$
z(t) = 0.2\sin(2t),\quad v_z = 0.4\cos(2t),\quad a_z = -0.8\sin(2t)
$$

Phase 4 (deceleration):

$$
a_x = -\frac{v_\max}{t_\text{decel}},\quad v_x = \max(0,\, v_\max + a_x t)
$$

Orientation during accel/decel simulates a nose-tilt:

$$
\theta = -0.2 \cdot \min\!\left(1,\; \frac{t}{t_\text{accel}}\right) \quad \text{(nose-down during accel)}
$$

✅ **Implemented:** `generate_hover_accel_cruise(duration, sample_rate, max_vel, hover_time, accel_time,
cruise_time, decel_time)`

### 3.2 Circular — `generate_circular()`

Circular motion in the xy-plane with sinusoidal altitude variation:

$$
x(t) = r\cos(\omega_c t),\quad y(t) = r\sin(\omega_c t),\quad z(t) = 0.5\sin(0.5\omega_c t)
$$

Derivatives (exact, not numerical):

$$
\dot{x} = -r\omega_c\sin(\omega_c t),\quad \dot{y} = r\omega_c\cos(\omega_c t),\quad \dot{z} = 0.25\omega_c\cos(0.5\omega_c t)
$$

$$
\ddot{x} = -r\omega_c^2\cos(\omega_c t),\quad \ddot{y} = -r\omega_c^2\sin(\omega_c t)
$$

Yaw aligned with velocity direction:

$$
\psi(t) = \arctan2\!\left(\dot{y}(t),\; \dot{x}(t)\right)
$$

Angular velocity computed from successive quaternion differences.

✅ **Implemented:** `generate_circular(duration, sample_rate, radius, angular_vel)`

### 3.3 Waypoint — `generate_waypoint_trajectory()`

Piecewise-linear interpolation through a list of 3D waypoints:

$$
\mathbf{p}(t) = (1-\alpha)\,\mathbf{w}_j + \alpha\,\mathbf{w}_{j+1},\quad \alpha = \frac{t - t_j}{t_{j+1} - t_j}
$$

Velocity and acceleration computed by finite differences:

$$
\mathbf{v}_k = \frac{\mathbf{p}_k - \mathbf{p}_{k-1}}{\Delta t},\quad
\mathbf{a}_k = \frac{\mathbf{v}_k - \mathbf{v}_{k-1}}{\Delta t}
$$

Orientation: identity quaternion throughout (no yaw tracking in this generator).

⚠️ **Limitation:** Piecewise-linear generates velocity discontinuities at waypoints, causing impulsive
acceleration spikes. The controller's feed-forward is tuned to handle these, but smooth polynomial
interpolation would be preferable.

✅ **Implemented:** `generate_waypoint_trajectory(waypoints, waypoint_times, dt)`

---

## 4. How Trajectory is Used

### Open-Loop (Pre-generated)

```python
traj = generate_hover_accel_cruise()
imu_data = IMUSimulator().simulate(traj)     # sensor generation
ekf = ExtendedKalmanFilter(dt=0.01)
for i, t in enumerate(traj.t):
    ekf.predict(imu_data.gyro[i])
    ekf.correct_accel(imu_data.accel[i])
```

### Closed-Loop (Online Waypoint Targeting)

In [examples/03_waypoint_nav.py](../examples/03_waypoint_nav.py), the controller does not pre-generate
a trajectory; instead it receives the next waypoint each step:

```python
target_pos = waypoints[wp_idx]
motors = controller.compute(target_pos, target_yaw, dt, prev_target_pos)
quad.update(dt, motors)
# advance to next waypoint when within convergence_threshold (0.05 m)
```

Feed-forward: `prev_target_pos` is passed to `compute()` so the controller can estimate target velocity
and blend it into the velocity setpoint.

---

## 5. Planned Extensions

### 5.1 Differential Flatness

The quadrotor is **differentially flat** in the flat outputs $\sigma = [x, y, z, \psi]$. Any smooth trajectory
in $\sigma$ corresponds to a unique sequence of states and inputs.

Given flat output trajectory $\sigma(t)$ and its derivatives up to 4th order (snap), the attitude and
thrust commands are recovered via:

$$
T = m \left\|\ddot{\mathbf{p}} - \mathbf{g}\right\|
$$

$$
\hat{\mathbf{z}}_B = \frac{\ddot{\mathbf{p}} - \mathbf{g}}{\|\ddot{\mathbf{p}} - \mathbf{g}\|}
$$

**Benefit:** The controller can be simplified to a pure flatness-based feed-forward with feedback
linearization, removing the need for nested PID loops.

### 5.2 Minimum-Snap Polynomial Trajectories — ✅ Implemented (commit `8fdef56`)

For a sequence of waypoints $\mathbf{w}_0, \ldots, \mathbf{w}_M$ with times $t_0, \ldots, t_M$,
find piecewise polynomial $p_i(t)$ of degree $n \geq 7$ that minimizes:

$$
J = \int_0^{t_M} \left\|\frac{d^4\mathbf{p}}{dt^4}\right\|^2 dt
$$

Subject to:
- $p_i(t_i) = \mathbf{w}_i$, $p_i(t_{i+1}) = \mathbf{w}_{i+1}$
- $C^3$ continuity at interior waypoints (velocity, acceleration, jerk match)
- Zero velocity, acceleration, and jerk at the first and last waypoint

This leads to a **Quadratic Program (QP)** solved via the bordered KKT system:

$$
J = \mathbf{c}^T Q \mathbf{c}, \quad \text{s.t.} \quad A_\text{eq}\mathbf{c} = \mathbf{b}_\text{eq}
$$

$$
\begin{bmatrix} Q & A^T \\ A & 0 \end{bmatrix}
\begin{bmatrix} \mathbf{c} \\ \boldsymbol\lambda \end{bmatrix}
=
\begin{bmatrix} \mathbf{0} \\ \mathbf{b} \end{bmatrix}
$$

Solved per axis with `scipy.linalg.lstsq`.

**Usage:**
```python
from drones_sim.trajectory import generate_minimum_snap

traj = generate_minimum_snap(
    waypoints=[(0,0,0), (1,0,1), (2,1,2)],
    times=[0.0, 3.0, 6.0],   # optional; auto-allocated from distances if None
    order=7,                  # polynomial degree per segment (must be >= 7)
    dt=0.01,
)
# traj.position, traj.velocity, traj.acceleration are C³-continuous
```

Time allocation when `times=None`: proportional to Euclidean inter-waypoint distances.
Tests in `tests/test_trajectory.py`.

### 5.3 Time Allocation

Before minimum snap, allocate times between waypoints based on segment length and kinematic limits:

$$
t_i = \frac{|\mathbf{w}_{i+1} - \mathbf{w}_i|}{v_{\max}}\cdot k_\text{safety}
$$

More sophisticated: trapezoidal velocity profile per segment.

### 5.4 Kinodynamic RRT* (Global Planner)

For obstacle-aware planning with dynamics constraints:

1. Sample random state $\mathbf{x}_\text{rand} \in \mathcal{X}_\text{free}$
2. Steer from $\mathbf{x}_\text{near}$ with bounded acceleration to reach $\mathbf{x}_\text{rand}$
3. Rewire to minimize cost (usually time or control effort)

Output: a path satisfying velocity/acceleration bounds, collision-free, globally optimal in the limit.

**Implementation plan:** Use `rtree` for neighbor queries; `scipy.integrate.solve_ivp` for local
steering with the quadrotor dynamics.

### 5.5 MPC-Based Replanning

A local MPC layer can replan within a rolling horizon when:

- New obstacles are detected
- Estimator drift causes position error
- External disturbances push the drone off the reference

This bridges the global planner and the tracking controller, providing smooth, feasible trajectories
under uncertainty.

### 5.6 Constraint-Aware Generation

Build kinematic limits directly into trajectory generation:

| Constraint | Source |
|-----------|--------|
| $\|\mathbf{v}\| \leq v_\max$ | From `max_lateral_speed`, `max_vertical_speed` in controller |
| $\|\mathbf{a}\| \leq a_\max$ | From `max_lateral_accel`, `max_vertical_accel` |
| $|\phi|, |\theta| \leq \theta_\max$ | From attitude-to-acceleration mapping |
| $T \in [T_\min, T_\max]$ | Motor limits |

---

## 6. Example Coverage

| Example | Planning mode | Feature |
|---------|-------------|---------|
| [examples/01_imu_ekf_basic.py](../examples/01_imu_ekf_basic.py) | Pre-generated | `generate_hover_accel_cruise` as sensor test input |
| [examples/02_ekf_adaptive.py](../examples/02_ekf_adaptive.py) | Pre-generated | Same trajectory, temperature-enabled |
| [examples/03_waypoint_nav.py](../examples/03_waypoint_nav.py) | Min-snap | `generate_minimum_snap` — C³-continuous reference from waypoints |
| [examples/04_viser_viewer.py](../examples/04_viser_viewer.py) | Online | Waypoint list targeted at runtime + 3D interactive playback |
| [examples/05_full_pipeline.py](../examples/05_full_pipeline.py) | Online | Waypoint targeting with EKF feedback |
| [examples/06_trajectory_following.py](../examples/06_trajectory_following.py) | Pre-generated | `generate_circular` — full pipeline 3-D trajectory tracking |

---

## 7. Further Reading

- Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors", ICRA 2011
- Mueller, Hehn & D'Andrea, "A Computationally Efficient Algorithm for State-to-State Quadrocopter Trajectory Generation", IROS 2013
- Richter, Bry & Roy, "Polynomial Trajectory Planning for Aggressive Quadrotor Flight in Dense Indoor Environments", ISRR 2013
- Faessler et al., "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag", IEEE RA-L 2018


