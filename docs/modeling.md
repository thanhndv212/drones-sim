# Quadcopter Modeling

**Implementation:** [src/drones_sim/dynamics/quadcopter.py](../src/drones_sim/dynamics/quadcopter.py),
[src/drones_sim/math_utils.py](../src/drones_sim/math_utils.py)

---

## 1. Overview

The simulator models the quadcopter as a rigid body governed by the Newton-Euler equations. The plant state
is a 12-element vector:

$$
\mathbf{x} = \begin{bmatrix} x & y & z & \dot{x} & \dot{y} & \dot{z} & \phi & \theta & \psi & p & q & r \end{bmatrix}^T
$$

where $\phi, \theta, \psi$ are roll, pitch, yaw (ZYX Euler angles) and $p, q, r$ are body-frame angular rates.

**Status:** ✅ Fully implemented in `QuadcopterDynamics` (`dynamics/quadcopter.py`).

---

## 2. Reference Frames and Conventions

### 2.1 Coordinate Frames

| Frame | Symbol | Description |
|-------|--------|-------------|
| World (inertial) | $\mathcal{W}$ | NED-like, z points up (positive altitude) |
| Body | $\mathcal{B}$ | Origin at drone center of mass; x forward, y right, z up |

### 2.2 Rotation Convention — ZYX Euler

The rotation matrix from body to world is built by the ZYX sequence:

$$
R(\phi, \theta, \psi) = R_z(\psi)\, R_y(\theta)\, R_x(\phi)
$$

$$
R_x = \begin{bmatrix} 1 & 0 & 0 \\ 0 & c_\phi & -s_\phi \\ 0 & s_\phi & c_\phi \end{bmatrix},\quad
R_y = \begin{bmatrix} c_\theta & 0 & s_\theta \\ 0 & 1 & 0 \\ -s_\theta & 0 & c_\theta \end{bmatrix},\quad
R_z = \begin{bmatrix} c_\psi & -s_\psi & 0 \\ s_\psi & c_\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

✅ **Implemented:** `euler_to_rotation_matrix(phi, theta, psi)` in `math_utils.py`.

### 2.3 Quaternion Convention

All quaternion utilities in the package use the **Hamilton** convention with component order $[w, x, y, z]$:

$$
\mathbf{q} = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}, \quad w^2 + x^2 + y^2 + z^2 = 1
$$

The identity (no rotation) is $\mathbf{q} = [1, 0, 0, 0]$.

> **Boundary note:** `scipy.spatial.transform.Rotation` uses $[x, y, z, w]$ (JPL order). Conversions always
> apply `np.roll(q, ±1)` at module boundaries:
> ```python
> q_xyzw = np.roll(q_wxyz, -1)   # wxyz → xyzw for scipy
> q_wxyz = np.roll(q_xyzw,  1)   # xyzw → wxyz after scipy
> ```

✅ **Implemented:** `quat_from_euler`, `quat_to_euler`, `quat_to_rotation_matrix` in `math_utils.py`.

---

## 3. Motor Configuration

### 3.1 X-Configuration Layout

```
        Motor 1 (+x, front)
             ^
Motor 4      |      Motor 2
(-y, left) --+-- (+y, right)
             |
        Motor 3 (-x, back)
```

Motor spin directions alternate CW/CCW to balance gyroscopic and reaction torques:

| Motor | Position | Spin | Thrust | Yaw contribution |
|-------|----------|------|--------|-----------------|
| 1 | +x (front) | CW  | +T | +τ_ψ |
| 2 | +y (right) | CCW | +T | -τ_ψ |
| 3 | -x (back)  | CW  | +T | +τ_ψ |
| 4 | -y (left)  | CCW | +T | -τ_ψ |

### 3.2 Force and Torque from Rotors

Each rotor $i$ spinning at angular speed $\omega_i$ (rad/s) produces:

$$
F_i = k_f \omega_i^2, \qquad M_i = k_m \omega_i^2
$$

where $k_f$ is the thrust coefficient and $k_m$ is the drag-torque (reaction moment) coefficient.

Total thrust and body torques:

$$
\begin{aligned}
T &= k_f \sum_{i=1}^{4} \omega_i^2 \\
\tau_\phi &= k_f L (\omega_2^2 - \omega_4^2) \\
\tau_\theta &= k_f L (\omega_3^2 - \omega_1^2) \\
\tau_\psi &= k_m (\omega_1^2 - \omega_2^2 + \omega_3^2 - \omega_4^2)
\end{aligned}
$$

where $L$ is the arm length (center to rotor).

### 3.3 Allocation Matrix

In matrix form, the mapping from squared rotor speeds to wrench is:

$$
\begin{bmatrix} T \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix}
= \underbrace{\begin{bmatrix}
k_f & k_f & k_f & k_f \\
0 & k_f L & 0 & -k_f L \\
-k_f L & 0 & k_f L & 0 \\
k_m & -k_m & k_m & -k_m
\end{bmatrix}}_{A}
\begin{bmatrix} \omega_1^2 \\ \omega_2^2 \\ \omega_3^2 \\ \omega_4^2 \end{bmatrix}
$$

The inverse, $A^{-1}$, maps desired wrench back to squared rotor speeds (used in the controller).

✅ **Implemented:** `QuadcopterDynamics.allocation_matrix()` returns $A$; the controller calls
`np.linalg.solve(A, desired_wrench)` to invert it.

---

## 4. Equations of Motion

### 4.1 Translational Dynamics

In world frame:

$$
m\ddot{\mathbf{p}} = \mathbf{F}_\text{thrust} + \mathbf{F}_\text{drag} + \mathbf{F}_\text{grav}
$$

$$
\mathbf{F}_\text{thrust} = R\, [0,\; 0,\; T]^T
$$

$$
\mathbf{F}_\text{drag} = -k_d\, \dot{\mathbf{p}}
$$

$$
\mathbf{F}_\text{grav} = [0,\; 0,\; -mg]^T
$$

Expanding the thrust projection in scalar form:

$$
\begin{aligned}
m\ddot{x} &= T(\cos\phi\sin\theta\cos\psi + \sin\phi\sin\psi) - k_d\dot{x} \\
m\ddot{y} &= T(\cos\phi\sin\theta\sin\psi - \sin\phi\cos\psi) - k_d\dot{y} \\
m\ddot{z} &= T\cos\phi\cos\theta - mg - k_d\dot{z}
\end{aligned}
$$

### 4.2 Rotational Dynamics

In body frame (Newton-Euler):

$$
\mathbf{I}\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})
$$

where $\boldsymbol{\omega} = [p, q, r]^T$ and $\boldsymbol{\tau} = [\tau_\phi, \tau_\theta, \tau_\psi]^T$.

Expanded Euler equations with inertia $\mathbf{I} = \text{diag}(I_x, I_y, I_z)$:

$$
\begin{aligned}
I_x \dot{p} &= \tau_\phi - (I_y - I_z)\, q\, r \\
I_y \dot{q} &= \tau_\theta - (I_z - I_x)\, p\, r \\
I_z \dot{r} &= \tau_\psi - (I_x - I_y)\, p\, q
\end{aligned}
$$

### 4.3 Kinematics

Position update:

$$
\dot{\mathbf{p}} = \mathbf{v}
$$

Euler angle rates from body angular velocity via the kinematic transform matrix $W$:

$$
\begin{bmatrix} \dot\phi \\ \dot\theta \\ \dot\psi \end{bmatrix}
= W^{-1} \begin{bmatrix} p \\ q \\ r \end{bmatrix},\qquad
W = \begin{bmatrix}
1 & 0 & -\sin\theta \\
0 & \cos\phi & \cos\theta\sin\phi \\
0 & -\sin\phi & \cos\theta\cos\phi
\end{bmatrix}
$$

✅ **Implemented:** `angular_vel_to_euler_rates(phi, theta, omega_body)` solves $W \dot\Phi = \boldsymbol\omega$.

### 4.4 State-Space Form (Linearization)

Around a hover trim point ($\phi=\theta=0$, $T=mg$), the system linearizes to:

$$
\dot{\mathbf{X}} = A\mathbf{X} + B\mathbf{U}
$$

where:

$$
\mathbf{X} = \begin{bmatrix} \mathbf{p} \\ \dot{\mathbf{p}} \\ \boldsymbol\Phi \\ \boldsymbol\omega \end{bmatrix} \in \mathbb{R}^{12},\quad
\mathbf{U} = \begin{bmatrix} T \\ \tau_\phi \\ \tau_\theta \\ \tau_\psi \end{bmatrix} \in \mathbb{R}^4
$$

The hover linearization has simple structure: translational and rotational channels decouple.

⚠️ **Not implemented:** Full symbolic $A, B$ matrices are not in the repo; this form is only used conceptually
for understanding. Implementing LQR from this linearization is on the roadmap.

---

## 5. Default Physical Parameters

| Parameter | Symbol | Default | Unit |
|-----------|--------|---------|------|
| Mass | $m$ | 1.0 | kg |
| Arm length | $L$ | 0.2 | m |
| Inertia $x$ | $I_x$ | 0.01 | kg·m² |
| Inertia $y$ | $I_y$ | 0.01 | kg·m² |
| Inertia $z$ | $I_z$ | 0.018 | kg·m² |
| Thrust coeff | $k_f$ | 1×10⁻⁶ | N/(rad/s)² |
| Drag-torque coeff | $k_m$ | 1×10⁻⁷ | N·m/(rad/s)² |
| Aero drag | $k_d$ | 0.1 | N·s/m |
| Gravity | $g$ | 9.81 | m/s² |

---

## 6. Math Utility Library

All quaternion and rotation helpers live in `src/drones_sim/math_utils.py`.

| Function | Description | Used by |
|----------|-------------|---------|
| `quat_normalize(q)` | Unit-normalize $\mathbf{q}$ | EKF, AHRS after every integration step |
| `quat_multiply(q1, q2)` | Hamilton product | EKF quaternion composition |
| `quat_conjugate(q)` | Conjugate / inverse for unit $\mathbf{q}$ | EKF |
| `quat_to_rotation_matrix(q)` | $\mathbf{q} \to R \in \mathbb{R}^{3\times3}$ | EKF, AHRS, IMU simulator |
| `quat_from_euler(roll, pitch, yaw)` | Euler → $\mathbf{q}$ (scipy boundary) | Trajectory generators |
| `quat_to_euler(q)` | $\mathbf{q}$ → Euler (scipy boundary) | Visualizer |
| `quat_derivative(q, omega)` | $\dot{\mathbf{q}} = \frac{1}{2}\mathbf{q} \otimes [0,\boldsymbol\omega]$ | EKF prediction |
| `quat_angular_velocity_jacobian(omega)` | $\Omega(\boldsymbol\omega)$ for linearization | EKF covariance propagation |
| `euler_to_rotation_matrix(phi, theta, psi)` | ZYX $R$ from Euler angles | Dynamics |
| `angular_vel_to_euler_rates(phi, theta, omega)` | $W^{-1}\boldsymbol\omega = \dot\Phi$ | Dynamics kinematics |

### Quaternion Rotation Matrix Expansion

For $\mathbf{q} = [w, x, y, z]$:

$$
R = \begin{bmatrix}
1 - 2(y^2+z^2) & 2(xy - wz) & 2(xz + wy) \\
2(xy + wz) & 1 - 2(x^2+z^2) & 2(yz - wx) \\
2(xz - wy) & 2(yz + wx) & 1 - 2(x^2+y^2)
\end{bmatrix}
$$

### Quaternion Derivative

$$
\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol\omega \end{bmatrix}
= \frac{1}{2} \Omega(\boldsymbol\omega)\, \mathbf{q}
$$

where the $4\times4$ skew-symmetric matrix is:

$$
\Omega(\boldsymbol\omega) = \frac{1}{2}\begin{bmatrix}
0 & -\omega_x & -\omega_y & -\omega_z \\
\omega_x & 0 & \omega_z & -\omega_y \\
\omega_y & -\omega_z & 0 & \omega_x \\
\omega_z & \omega_y & -\omega_x & 0
\end{bmatrix}
$$

---

## 7. Numerical Integration

✅ **Implemented:** Forward Euler with fixed time step $dt$:

```python
self.state += deriv * dt   # in QuadcopterDynamics.update()
```

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + f(\mathbf{x}_k,\mathbf{u}_k)\, \Delta t
$$

⚠️ **Limitation:** Forward Euler is first-order accurate and can become unstable for stiff dynamics or large $\Delta t$.

✅ **Implemented:** Upgrade to 4th-order Runge-Kutta (RK4):

$$
\begin{aligned}
k_1 &= f(\mathbf{x}_k,\, \mathbf{u}) \\
k_2 &= f\!\left(\mathbf{x}_k + \tfrac{\Delta t}{2}k_1,\, \mathbf{u}\right) \\
k_3 &= f\!\left(\mathbf{x}_k + \tfrac{\Delta t}{2}k_2,\, \mathbf{u}\right) \\
k_4 &= f(\mathbf{x}_k + \Delta t\, k_3,\, \mathbf{u}) \\
\mathbf{x}_{k+1} &= \mathbf{x}_k + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

---

## 8. Current Limitations and Known Issues

| Limitation | Impact | Status |
|-----------|--------|--------|
| Euler attitude (gimbal lock near $\theta = \pm90°$) | Singularity in kinematic transform $W$ | ⚠️ Known |
| Forward Euler integration | Numerical drift at large $\Delta t$ | ✅ Solved |
| Linear drag only $(-k_d \mathbf{v})$ | Under-estimates drag at high speed | ⚠️ Known |
| Instant motor response | Ignores actuator bandwidth | ✅ Implemented (1st-order lag) |
| No wind model | Cannot test disturbance rejection | 🗺️ Roadmap |

---

## 9. Planned Extensions

### 9.1 Motor Dynamics (First-Order Lag)

✅ **Implemented** (commit `f108740`)

Real rotors do not follow commands instantaneously. A first-order lag model:

$$
\tau_m \dot\omega_i + \omega_i = \omega_{i,\text{cmd}}
$$

Discretized:

$$
\omega_i^{k+1} = \omega_i^k + \frac{\Delta t}{\tau_m}\left(\omega_{i,\text{cmd}} - \omega_i^k\right)
$$

Typical $\tau_m$: 30–80 ms for small UAVs.

**Implementation:**
```python
# QuadcopterDynamics.__init__
motor_time_constant: float = 0.0   # 0 = disabled (backward-compatible)
motor_states: NDArray              # actual rotor speeds after lag, shape (4,)

# update() applies lag first, then RK4 on plant
if self.motor_time_constant > 0:
    alpha = dt / self.motor_time_constant
    self.motor_states += alpha * (motor_speeds - self.motor_states)
else:
    self.motor_states = motor_speeds.copy()
```

Access actual rotor speeds via `quad.get_motor_speeds()`. Tests in `tests/test_dynamics.py`.

### 9.2 Wind and Turbulence (Dryden Model)

The Dryden model generates low-altitude wind turbulence via shaping filters on white noise:

$$
H_u(s) = \sigma_u \sqrt{\frac{2L_u}{\pi V}} \cdot \frac{1}{1 + \frac{L_u}{V}s}
$$

where $L_u$ is turbulence scale length and $V$ is airspeed. The induced body-frame force is added before the
translational acceleration computation.

**Implementation plan:** Add `wind_callback(t, pos) -> np.ndarray` optional parameter to `update()`.

### 9.3 Ground Effect

Near the ground (altitude $h < 2D$ where $D$ is rotor diameter), effective thrust increases:

$$
T_{\text{eff}} \approx T \left(1 + \frac{k_{\text{ge}}}{(h/D)^2}\right)
$$

### 9.4 Full Quaternion State

Replace $[\phi, \theta, \psi]$ in the plant state with quaternion $\mathbf{q}$, eliminating the kinematic
singularity near $\theta = \pm 90°$.

---

## 10. Gimbal Modeling (Future Payload)

A gimbal coupled to the drone adds attitude-dependent torques. The gimbal dynamics:

$$
I_g \ddot\theta_g + b_g \dot\theta_g + k_g \theta_g = \tau_g - \tau_{\text{coupling}}
$$

where $\tau_{\text{coupling}}$ feeds back into the body torque equations. This is relevant when modeling
camera-stabilization payloads.

---

## 11. Validation

| Test | File | What is checked |
|------|------|----------------|
| Quaternion identities | [tests/test_math_utils.py](../tests/test_math_utils.py) | normalize, multiply (identity), conjugate, round-trip Euler, zero-omega derivative |
| Free-fall | [tests/test_dynamics.py](../tests/test_dynamics.py) | acceleration ≈ −g with zero thrust |
| Hover consistency | [tests/test_dynamics.py](../tests/test_dynamics.py) | net vertical force ≈ 0 at hover motor speed |

## 12. Further Reading

- Randal W. Beard, *Quadrotor Dynamics and Control* (BYU textbook) — accessible derivations of Newton-Euler model
- PX4 and ArduPilot vehicle-modeling references — practical implementation constraints
- Mahony et al., "Nonlinear Complementary Filters on the Special Orthogonal Group", IEEE TAC 2008
- Faessler et al., "Differential Flatness of Quadrotor Dynamics Subject to Rotor Drag", IEEE RA-L 2018
