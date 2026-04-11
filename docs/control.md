# Control

**Implementation:** [src/drones_sim/control/pid.py](../src/drones_sim/control/pid.py),
[src/drones_sim/control/cascaded.py](../src/drones_sim/control/cascaded.py),
[src/drones_sim/control/lqr.py](../src/drones_sim/control/lqr.py)

---

## 1. Overview

The simulator implements a three-loop cascaded PID controller that maps position/yaw targets to
motor speed commands. The cascade decomposes the problem into physically meaningful stages with
appropriate bandwidth separation.

```
target_pos, target_yaw
        │
┌───────▼──────────┐
│  Position Loop    │  (outer)  → target velocity
│  x_ctrl, y_ctrl,  │
│  z_ctrl           │
└───────┬──────────┘
        │
┌───────▼──────────┐
│  Velocity Loop    │  (middle) → desired acceleration
│  vx_ctrl, vy_ctrl,│
│  vz_ctrl          │
└───────┬──────────┘
        │
    Small-angle mapping + thrust allocation
        │
┌───────▼──────────┐
│  Attitude Loop    │  (inner)  → torque commands
│  roll_ctrl,       │
│  pitch_ctrl,      │
│  yaw_ctrl         │
└───────┬──────────┘
        │
    Allocation matrix solve A·ω² = [T, τ_φ, τ_θ, τ_ψ]
        │
    ω = √(max(ω², 0))  clipped to [0, ω_max]
        │
    motor_speeds (4,)
```

✅ **Implemented in:** `QuadcopterController` (`control/cascaded.py`)

---

## 2. PID Primitive (`PIDController`)

### 2.1 Control Law

$$
u(t) = K_p\, e(t) + K_i \int_0^t e(\tau)\, d\tau + K_d\, \frac{de}{dt}
$$

Discrete form with time step $\Delta t$:

$$
u_k = K_p e_k + K_i \sum_{j=0}^{k} e_j \Delta t + K_d \frac{e_k - e_{k-1}}{\Delta t}
$$

### 2.2 Anti-Windup

Integral term is clamped element-wise:

$$
I_k \leftarrow \text{clip}(I_k,\; I_\text{min},\; I_\text{max})
$$

Output is saturated:

$$
u_k \leftarrow \text{clip}(u_k,\; u_\text{min},\; u_\text{max})
$$

### 2.3 Constructor Parameters

```python
PIDController(kp, ki, kd, output_limits=(lo, hi), windup_limits=(lo, hi))
```

| Method | Description |
|--------|-------------|
| `update(setpoint, measurement, dt)` | One step; returns scalar output |
| `reset()` | Clears integral and previous error |

✅ **Implemented in:** `PIDController` (`control/pid.py`)

---

## 3. Cascaded Controller — Gains and Limits

### 3.1 Nine PID Instances

| Instance | Axis | $K_p$ | $K_i$ | $K_d$ | Output limits | Windup limits |
|----------|------|--------|--------|--------|---------------|---------------|
| `x_ctrl` | x pos | 1.2 | 0.1 | 0.35 | (−1.5, 1.5) m/s | (−1.0, 1.0) |
| `y_ctrl` | y pos | 1.2 | 0.1 | 0.35 | (−1.5, 1.5) m/s | (−1.0, 1.0) |
| `z_ctrl` | z pos | 1.6 | 0.25 | 0.45 | (−1.0, 1.0) m/s | (−0.8, 0.8) |
| `vx_ctrl` | vx | 2.2 | 0.3 | 0.2 | (−3.0, 3.0) m/s² | (−1.0, 1.0) |
| `vy_ctrl` | vy | 2.2 | 0.3 | 0.2 | (−3.0, 3.0) m/s² | (−1.0, 1.0) |
| `vz_ctrl` | vz | 5.0 | 1.5 | 0.3 | (−4.0, 4.0) m/s² | (−1.0, 1.0) |
| `roll_ctrl` | roll | 0.12 | 0.02 | 0.0 | (−0.25, 0.25) N·m | (−0.15, 0.15) |
| `pitch_ctrl` | pitch | 0.12 | 0.02 | 0.0 | (−0.25, 0.25) N·m | (−0.15, 0.15) |
| `yaw_ctrl` | yaw | 0.05 | 0.01 | 0.0 | (−0.08, 0.08) N·m | (−0.08, 0.08) |

### 3.2 Hard Physical Limits

| Limit | Value | Applied where |
|-------|-------|---------------|
| `convergence_threshold` | 0.05 m | Waypoint acceptance radius |
| `max_lateral_speed` | 1.5 m/s | Velocity setpoint clamp |
| `max_vertical_speed` | 1.0 m/s | Velocity setpoint clamp |
| `max_lateral_accel` | 3.0 m/s² | Desired acceleration clamp |
| `max_vertical_accel` | 4.0 m/s² | Desired acceleration clamp |
| `max_tilt` | 17° (0.297 rad) | Roll/pitch setpoint clamp |
| `max_torque` | [0.25, 0.25, 0.08] N·m | Torque output clamp |
| `rate_damping` | [0.03, 0.03, 0.015] | Angular rate damping gains |
| `min_thrust` | $0.2mg$ | Lower thrust bound |
| `max_thrust` | $2.0mg$ | Upper thrust bound |
| `max_motor_speed` | 4000 rad/s | Motor speed output clamp |

---

## 4. Control Law — Step by Step

### 4.1 Position → Target Velocity

PID feedback:

$$
\mathbf{v}_\text{ref} = \begin{bmatrix}
K_p^{x}(x_d - x) + K_i^{x}\int\! e_x + K_d^{x}\dot{e}_x \\
K_p^{y}(y_d - y) + K_i^{y}\int\! e_y + K_d^{y}\dot{e}_y \\
K_p^{z}(z_d - z) + K_i^{z}\int\! e_z + K_d^{z}\dot{e}_z
\end{bmatrix}
$$

Feed-forward from target position derivative (when `prev_target_pos` is provided):

$$
\mathbf{v}_\text{ff} = \text{clip}\!\left(\frac{\mathbf{p}_d^k - \mathbf{p}_d^{k-1}}{\Delta t},\; \pm v_\text{max}\right)
$$

Blended with gains $[0.25, 0.25, 0.15]$:

$$
\mathbf{v}_\text{ref} \leftarrow \mathbf{v}_\text{ref} + [0.25, 0.25, 0.15] \odot \mathbf{v}_\text{ff}
$$

### 4.2 Velocity → Desired Acceleration

$$
\mathbf{a}_\text{des} = \begin{bmatrix}
K_p^{vx}(v_{xd} - v_x) + \ldots \\
K_p^{vy}(v_{yd} - v_y) + \ldots \\
K_p^{vz}(v_{zd} - v_z) + \ldots
\end{bmatrix}
+ [0.1, 0.1, 0.05] \odot \frac{\mathbf{v}_\text{ref}^k - \mathbf{v}_\text{ref}^{k-1}}{\Delta t}
$$

(Acceleration feed-forward with small gain to reduce lag.)

### 4.3 Small-Angle Attitude Mapping

For small angles, lateral acceleration ↔ attitude angle via gravitational coupling:

$$
\theta_d = \text{clip}\!\left(\frac{a_{x,\text{des}}}{g},\; -\theta_\text{max},\; \theta_\text{max}\right)
\quad \text{(pitch)}
$$

$$
\phi_d = \text{clip}\!\left(\frac{-a_{y,\text{des}}}{g},\; -\phi_\text{max},\; \phi_\text{max}\right)
\quad \text{(roll)}
$$

Sign convention: positive pitch produces +x acceleration; positive roll produces −y acceleration.

### 4.4 Thrust Computation

$$
T = \frac{m(g + a_{z,\text{des}})}{\max(\cos\phi \cos\theta,\; 0.7)}
$$

Clamped to $[T_\text{min}, T_\text{max}] = [0.2mg,\; 2.0mg]$.

The cosine correction compensates for the reduced vertical component of thrust when tilted. The floor of 0.7 prevents division by tiny values at extreme tilt.

### 4.5 Attitude → Torques

With angular-rate damping:

$$
\tau_\phi = K_p^{\phi}(\phi_d - \phi) + K_i^{\phi}\int\! e_\phi - D_r^{\phi}\, p
$$

$$
\tau_\theta = K_p^{\theta}(\theta_d - \theta) + K_i^{\theta}\int\! e_\theta - D_r^{\theta}\, q
$$

$$
\tau_\psi = K_p^{\psi}(\psi_d - \psi) + K_i^{\psi}\int\! e_\psi - D_r^{\psi}\, r
$$

where $D_r = [0.03, 0.03, 0.015]$. The rate damping suppresses oscillation and acts like a derivative
filtered through measured angular rate rather than numerical differentiation of angle.

### 4.6 Motor Allocation (Inverse)

Solve the allocation matrix $A$ (see [modeling.md](modeling.md)):

$$
A\, \boldsymbol\omega^2 = \begin{bmatrix} T \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix}
\implies
\boldsymbol\omega^2 = A^{-1} \begin{bmatrix} T \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix}
$$

Motor speeds:

$$
\omega_i = \sqrt{\max(\omega_i^2, 0)}, \quad \omega_i \leftarrow \text{clip}(\omega_i, 0, \omega_\text{max})
$$

If $A$ is singular (unlikely for X-config), the controller falls back to hover speed:

$$
\omega_\text{hover} = \sqrt{\frac{mg}{4 k_f}}
$$

---

## 5. Controller Family Comparison

The project's cascaded PID is the baseline. Other families offer different trade-offs:

### 5.1 PID (✅ Implemented)

**Law:** $u = K_p e + K_i \int e + K_d \dot{e}$

Strength: simple, robust, tunable. Best for: attitude, altitude, position hold.

Weakness: no constraint handling; gains are fixed; performance degrades with large model error.

### 5.2 Linear Quadratic Regulator (LQR) — ✅ Implemented (commit `50a8cc0`)

**Law:** $\mathbf{u} = -K\mathbf{x}$, where $K = R^{-1}B^T P$ and $P$ satisfies the **Algebraic Riccati
Equation (ARE)**:

$$
A^T P + P A - P B R^{-1} B^T P + Q = 0
$$

- $Q$: state penalty matrix (penalizes state deviations)
- $R$: input penalty matrix (penalizes control effort)
- $P$: positive-definite solution determining optimal gain

The plant is linearized at hover ($\phi=\theta=0$, $T=mg$). The resulting $A \in \mathbb{R}^{12\times12}$,
$B \in \mathbb{R}^{12\times4}$ matrices have simple structure: translational and rotational channels
decouple. Gain $K$ is computed once offline with `scipy.linalg.solve_continuous_are(A, B, Q, R)`.

**Interface** (same as `QuadcopterController`):

```python
lqr = LQRController(quad)           # computes K at construction
motors = lqr.compute(target_pos, target_yaw, dt, prev_target_pos)
```

**Default weights:**

| State group | Q diagonal | R diagonal |
|-------------|-----------|------------|
| Position x, y, z | 10, 10, 12 | — |
| Velocity vx, vy, vz | 2, 2, 4 | — |
| Attitude φ, θ, ψ | 5, 5, 3 | — |
| Body rate p, q, r | 0.5, 0.5, 0.3 | — |
| Thrust T | — | 0.01 |
| Torques τφ, τθ, τψ | — | 50, 50, 20 |

**Strength:** optimal local stabilization; no integrator windup; single-gain matrix easy to tune.

**Limitation:** only optimal at the linearization point (hover); no explicit constraint satisfaction;
performance degrades at large departure from hover (strong nonlinearity).

**Implemented in:** `LQRController` (`control/lqr.py`). Tests in `tests/test_lqr.py`.

### 5.3 Model Predictive Control (MPC) — 🗺️ Planned

Solves a constrained optimization over horizon $N$:

$$
\min_{\mathbf{u}_0,\ldots,\mathbf{u}_{N-1}} \sum_{k=0}^{N-1} \left(\mathbf{x}_k^T Q \mathbf{x}_k + \mathbf{u}_k^T R \mathbf{u}_k\right) + \mathbf{x}_N^T P_f \mathbf{x}_N
$$

Subject to:

$$
\mathbf{x}_{k+1} = A\mathbf{x}_k + B\mathbf{u}_k,\quad \mathbf{u}_k \in \mathcal{U},\quad \mathbf{x}_k \in \mathcal{X}
$$

Only the first action $\mathbf{u}_0$ is applied (receding horizon).

**Best for:** constrained trajectory tracking; online replanning; tight integration of actuator limits.

### 5.4 Adaptive Control — 🗺️ Planned

Gain matrix $K(t)$ is updated online to handle parameter changes:

$$
\mathbf{u} = K(t)\mathbf{x}, \quad \dot{K}(t) = -\gamma\, \mathbf{x}\, \mathbf{e}^T
$$

**Best for:** payload variation; motor degradation; environmental adaptation.

### 5.5 Sliding Mode Control (SMC) — 🗺️ Planned

Define sliding surface $\sigma = ce + \dot{e}$. Force $\sigma \to 0$:

$$
\mathbf{u} = \mathbf{u}_\text{eq} - k \cdot \text{sign}(\sigma)
$$

**Best for:** aggressive maneuvers; guaranteed convergence despite bounded uncertainty.

**Caveat:** discontinuous control causes chattering; use boundary layer smoothing.

### 5.6 Geometric Control on SE(3) — 🗺️ Planned

Works directly with rotation matrices; no local angle approximations, no gimbal-lock risk:

$$
\mathbf{e}_R = \frac{1}{2}\left(R_d^T R - R^T R_d\right)^\vee
$$

$$
\boldsymbol\tau = -K_R \mathbf{e}_R - K_\Omega \mathbf{e}_\Omega + \boldsymbol\Omega \times J\boldsymbol\Omega
$$

**Best for:** aggressive 3D trajectories; flips and full aerobatic maneuvers.

### 5.7 Controller Usage Heuristics

| Controller | Attitude | Altitude | Position | Trajectory | Constraints | Adaptation |
|------------|----------|----------|----------|------------|-------------|------------|
| PID ✅ | ✅ | ✅ | ✅ | — | — | — |
| LQR ✅ | ✅ | — | ✅ | ✅ | — | — |
| MPC 🗺️ | — | — | ✅ | ✅ | ✅ | — |
| Adaptive 🗺️ | — | ✅ | — | — | — | ✅ |
| SMC 🗺️ | ✅ | — | — | — | — | ✅ |
| Geometric 🗺️ | ✅ | — | ✅ | ✅ | — | — |

---

## 6. Tuning Guide

### 6.1 Recommended Order: Inner-to-Outer

**Step 1 — Attitude loop** (fastest; tune first):

1. Hover at fixed position, apply a small roll/pitch perturbation: `quad.state[6:8] = [0.1, 0.1]`
2. Set $K_i = K_d = 0$; increase $K_p$ until onset of oscillation, then back off 30%
3. Add rate damping $D_r$ to dampen oscillation without derivative noise
4. Add $K_i$ only if steady-state offset remains
5. Settling target: < 0.3 s with < 5% overshoot

**Step 2 — Velocity loop:**

1. Hold altitude fixed; command step lateral velocity
2. Watch for oscillation in velocity response
3. Tune $K_p$, then $K_d$, then small $K_i$

**Step 3 — Position loop (outermost; tune last):**

1. Command step position change; watch settle time and overshoot
2. $K_p$ sets converge speed; $K_d$ damps position overshoot
3. Feed-forward gain $[0.25, 0.25, 0.15]$ reduces tracking lag during motion

### 6.2 Common Failure Modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Roll/pitch oscillation | Attitude $K_p$ too high | Reduce `roll_ctrl.kp` / `pitch_ctrl.kp` |
| Slow position response | Velocity $K_p$ too low | Increase `vx_ctrl.kp`, `vy_ctrl.kp` |
| Altitude drift | Low $K_i$ in z loop | Increase `z_ctrl.ki` or `vz_ctrl.ki` |
| Yaw windup | Yaw $K_i$ too aggressive | Reduce `yaw_ctrl.ki` or tighten windup limits |
| Motor saturation | Gains or trajectory too aggressive | Reduce max tilt / max acceleration limits |

---

## 7. Planned Extensions

### 7.1 Disturbance Observer

Estimate unknown external force/torque $\mathbf{d}$:

$$
\dot{\hat{\mathbf{d}}} = L\!\left(\mathbf{y}_\text{obs} - \hat{\mathbf{y}}_\text{obs}\right)
$$

Apply feed-forward compensation: $\mathbf{u} \leftarrow \mathbf{u} - \hat{\mathbf{d}}$.

### 7.2 INDI (Incremental Nonlinear Dynamic Inversion)

Uses measured acceleration feedback to cancel model mismatch:

$$
\delta\mathbf{u} = G^{-1}\!\left(\dot{\boldsymbol\nu}_d - \dot{\boldsymbol\nu}_\text{meas}\right)
$$

Avoids tuning for every flight condition.

### 7.3 Gain Scheduling

Choose gains based on flight regime (hover vs. agile):

```python
if speed < 1.0:   # hover regime
    kp, ki, kd = hover_gains
else:             # forward flight
    kp, ki, kd = cruise_gains
```

### 7.4 LQR Implementation Plan

```python
from scipy.linalg import solve_continuous_are
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P
u = -K @ x_error
```

Add `LQRController` class alongside `PIDController`; expose same interface (`compute(target_pos, ...)`).

---

## 8. Test Coverage

| Test | File | What is checked |
|------|------|----------------|
| Proportional response | [tests/test_pid.py](../tests/test_pid.py) | Output = $K_p \cdot e$ when $K_i = K_d = 0$ |
| Output saturation | [tests/test_pid.py](../tests/test_pid.py) | Large error clips to output limits |
| Integral clamp | [tests/test_pid.py](../tests/test_pid.py) | Integral does not grow past windup limits |
| Reset | [tests/test_pid.py](../tests/test_pid.py) | `reset()` clears integral and prev error |
| Closed-loop stability | [tests/test_cascaded.py](../tests/test_cascaded.py) | Controller reaches hover/waypoint without diverging |

---

## 9. Further Reading

- Mahony et al., "Multirotor Aerial Vehicles", IEEE RAS Magazine 2012 — cascaded control foundations
- Mellinger & Kumar, "Minimum Snap Trajectory Generation and Control for Quadrotors", ICRA 2011
- Lee et al., "Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC 2010
- Kamel et al., "Linear vs Nonlinear MPC for Quadrotor Position Control", IFAC 2017


