# State Estimation

**Implementation:** [src/drones_sim/estimation/ekf.py](../src/drones_sim/estimation/ekf.py),
[src/drones_sim/estimation/ahrs.py](../src/drones_sim/estimation/ahrs.py)

---

## 1. Overview

State estimation recovers position, velocity, and orientation from noisy sensor data. The package provides
two estimators that can be chosen based on accuracy requirements and compute budget:

| Estimator | State dim | Attitude | Key strength |
|-----------|-----------|----------|--------------|
| `ExtendedKalmanFilter` | 10 | Quaternion (in state) | Rigorous, analytical Jacobians |
| `AdaptiveEKF` + `AHRS` | 9 + AHRS | Quaternion (AHRS) | Computationally lighter; adaptive noise |

All estimators operate on `IMUData` produced by `IMUSimulator` and output state dictionaries used by the
controller.

---

## 2. Extended Kalman Filter Theory

### 2.1 Why EKF for Drones

Quadcopter dynamics and sensor models are nonlinear. The EKF linearizes these around the current estimate
at each step, enabling the Kalman filter recursion:

- Sensor data is noisy, biased, and from multiple modalities (accel, gyro, mag)
- Some states (orientation from gravity direction) are not directly observed
- Process and measurement uncertainty must be tracked to weight data correctly

### 2.2 EKF Predict Step

Given state estimate $\hat{\mathbf{x}}_{k-1|k-1}$ and covariance $P_{k-1|k-1}$:

$$
\hat{\mathbf{x}}_{k|k-1} = f\!\left(\hat{\mathbf{x}}_{k-1|k-1},\, \mathbf{u}_k\right)
$$

$$
P_{k|k-1} = F_k\, P_{k-1|k-1}\, F_k^T + Q_k
$$

where $F_k = \frac{\partial f}{\partial \mathbf{x}}\big|_{\hat{\mathbf{x}}_{k-1|k-1}}$ is the state
transition Jacobian and $Q_k$ is the process noise covariance.

### 2.3 EKF Update Step

Given measurement $\mathbf{z}_k$ with model $h(\mathbf{x})$:

$$
\mathbf{y}_k = \mathbf{z}_k - h\!\left(\hat{\mathbf{x}}_{k|k-1}\right) \quad \text{(innovation)}
$$

$$
S_k = H_k\, P_{k|k-1}\, H_k^T + R_k \quad \text{(innovation covariance)}
$$

$$
K_k = P_{k|k-1}\, H_k^T\, S_k^{-1} \quad \text{(Kalman gain)}
$$

$$
\hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + K_k\, \mathbf{y}_k
$$

$$
P_{k|k} = (I - K_k H_k)\, P_{k|k-1}
$$

where $H_k = \frac{\partial h}{\partial \mathbf{x}}\big|_{\hat{\mathbf{x}}_{k|k-1}}$ is the measurement Jacobian
and $R_k$ is the measurement noise covariance.

> **Numerical note:** The standard form $(I - KH)P$ can lose positive-definiteness in finite precision.
> The Joseph form $(I - KH)P(I - KH)^T + KRK^T$ is more robust. AdaptiveEKF uses this.

---

## 3. Full-State EKF (10-State) — `ExtendedKalmanFilter`

### 3.1 State Vector

$$
\mathbf{x} = \begin{bmatrix} p_x & p_y & p_z & v_x & v_y & v_z & q_w & q_x & q_y & q_z \end{bmatrix}^T \in \mathbb{R}^{10}
$$

Initial quaternion: $\mathbf{q}_0 = [1, 0, 0, 0]$ (identity).

### 3.2 Predict Step Implementation

Input: body-frame gyroscope $\boldsymbol\omega_\text{meas}$.

Bias-corrected rate: $\boldsymbol\omega = \boldsymbol\omega_\text{meas} - \hat{\mathbf{b}}_g$

Quaternion prediction via `quat_derivative`:

$$
\dot{\mathbf{q}} = \frac{1}{2}\mathbf{q} \otimes [0,\; \boldsymbol\omega]^T
$$

$$
\mathbf{q}_{k+1} = \text{normalize}\!\left(\mathbf{q}_k + \dot{\mathbf{q}}\, \Delta t\right)
$$

Position kinematics (constant-velocity assumption in predict):

$$
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k\, \Delta t
$$

State transition Jacobian $F \in \mathbb{R}^{10\times10}$:

$$
F = \begin{bmatrix}
I_3 & \Delta t\, I_3 & 0 \\
0 & I_3 & 0 \\
0 & 0 & I_4 + \Omega(\boldsymbol\omega)\, \Delta t
\end{bmatrix}
$$

where $\Omega(\boldsymbol\omega)$ is the $4\times4$ skew-symmetric matrix from `quat_angular_velocity_jacobian`.

✅ **Implemented in:** `ExtendedKalmanFilter.predict(gyro)`

### 3.3 Accelerometer Correction

The accelerometer in hover measures the reaction to gravity in the body frame:

$$
\mathbf{z}_\text{accel} = -R^T \mathbf{g} + \mathbf{b}_a + \boldsymbol\eta_a
$$

Innovation: $\mathbf{y} = (\mathbf{a}_\text{meas} - \hat{\mathbf{b}}_a) - (-R(\hat{\mathbf{q}})^T \mathbf{g})$

The $3\times10$ Jacobian $H_\text{accel}$ has non-zero entries only in the quaternion columns (indices 6–9).
For $\mathbf{g} = [g_x, g_y, g_z]^T$ and $\mathbf{q} = [w, x, y, z]$:

$$
\frac{\partial (R^T \mathbf{g})}{\partial q_w} = -2\begin{bmatrix} 2wg_z - 2g_yy + 2g_xz \\ 2g_zy - 2g_xz \\ -2g_yw - 2g_xx \end{bmatrix}, \quad \text{(and similarly for } q_x, q_y, q_z\text{)}
$$

After correction the quaternion is renormalized.

Additionally, a soft velocity update blends world-frame acceleration:

$$
\mathbf{v}_{k+1} = (1-\alpha)\mathbf{v}_k + \alpha\!\left(\mathbf{v}_k + (R\mathbf{a} + \mathbf{g})\Delta t\right), \quad \alpha = 0.1
$$

✅ **Implemented in:** `ExtendedKalmanFilter.correct_accel(accel)`, `_accel_jacobian(q)`

### 3.4 Magnetometer Correction

Reference magnetic field vector in world frame: $\mathbf{m}_\text{ref} = [20, 0, -40]^T$ μT (configurable).

Expected body-frame measurement:

$$
\mathbf{z}_\text{mag} = R^T \mathbf{m}_\text{ref}
$$

The $3\times10$ Jacobian $H_\text{mag}$ mirrors $H_\text{accel}$ structure for the quaternion columns.

✅ **Implemented in:** `ExtendedKalmanFilter.correct_mag(mag)`, `_mag_jacobian(q)`

### 3.5 Noise Parameters

| Parameter | Default value | Meaning |
|-----------|--------------|---------|
| `P[0:3,0:3]` | $0.0001 \cdot I$ | Initial position uncertainty |
| `P[3:6,3:6]` | $0.001 \cdot I$ | Initial velocity uncertainty |
| `P[6:10,6:10]` | $0.00001 \cdot I$ | Initial quaternion uncertainty |
| `Q[0:3,0:3]` | $0.000001 \cdot I$ | Position process noise |
| `Q[3:6,3:6]` | $0.00001 \cdot I$ | Velocity process noise |
| `Q[6:10,6:10]` | $0.000001 \cdot I$ | Quaternion process noise |
| `R_accel` | $0.05 \cdot I$ | Accelerometer measurement noise |
| `R_mag` | $0.5 \cdot I$ | Magnetometer measurement noise |

---

## 4. AHRS — Complementary Filter (`AHRS`)

### 4.1 Concept

AHRS is a lightweight alternative for attitude-only estimation. It combines:

- **Gyroscope:** integrates angular rate for high-frequency attitude tracking
- **Accelerometer:** provides gravity direction as a low-frequency tilt reference
- **Magnetometer:** provides heading (yaw) reference

The complementary filter avoids the full Kalman recursion and is significantly cheaper to run.

### 4.2 Cross-Product Error Feedback

Orientation error from accelerometer:

$$
\mathbf{e}_a = \hat{\mathbf{a}}_\text{meas} \times \hat{\mathbf{g}}_\text{expected},\quad
\hat{\mathbf{g}}_\text{expected} = -R(\mathbf{q})^T \hat{\mathbf{g}}
$$

Orientation error from magnetometer:

$$
\mathbf{e}_m = \hat{\mathbf{m}}_\text{meas} \times \hat{\mathbf{m}}_\text{expected},\quad
\hat{\mathbf{m}}_\text{expected} = R(\mathbf{q})^T \hat{\mathbf{m}}_\text{ref}
$$

Combined correction:

$$
\mathbf{e} = \alpha_a\, \mathbf{e}_a + \alpha_m\, \mathbf{e}_m
$$

where $\alpha_a = 0.02$, $\alpha_m = 0.01$ (configurable `accel_weight`, `mag_weight`).

### 4.3 Gyro Bias Learning

Each step updates an estimated gyro bias:

$$
\hat{\mathbf{b}}_g \leftarrow \hat{\mathbf{b}}_g + \beta\, \mathbf{e}
$$

with $\beta = 0.001$ (`bias_learn_rate`). The corrected rate used for integration is:

$$
\boldsymbol\omega_\text{corr} = \boldsymbol\omega_\text{meas} - \hat{\mathbf{b}}_g + \mathbf{e}
$$

### 4.4 Quaternion Integration

$$
\dot{\mathbf{q}} = \frac{1}{2} \begin{bmatrix}
-q_x \omega_x - q_y \omega_y - q_z \omega_z \\
 q_w \omega_x + q_y \omega_z - q_z \omega_y \\
 q_w \omega_y + q_z \omega_x - q_x \omega_z \\
 q_w \omega_z + q_x \omega_y - q_y \omega_x
\end{bmatrix}
$$

$$
\mathbf{q}_{k+1} = \text{normalize}\!\left(\mathbf{q}_k + \dot{\mathbf{q}}\, \Delta t\right)
$$

✅ **Implemented in:** `AHRS.update(gyro, accel, mag)`

---

## 5. AHRS-Aided Adaptive EKF (9-State) — `AdaptiveEKF`

### 5.1 State Vector

$$
\mathbf{x} = \begin{bmatrix} p_x & p_y & p_z & v_x & v_y & v_z & b_{ax} & b_{ay} & b_{az} \end{bmatrix}^T \in \mathbb{R}^9
$$

Attitude is maintained by the embedded `AHRS` instance. The EKF operates only on
translational states and accelerometer bias.

### 5.2 Predict Step

1. `AHRS.update(gyro, accel, mag)` → orientation quaternion $\mathbf{q}$, gyro bias $\hat{\mathbf{b}}_g$
2. Accelerometer bias corrected: $\mathbf{a}_\text{corr} = \mathbf{a}_\text{meas} - \hat{\mathbf{b}}_a$
3. World-frame acceleration: $\mathbf{a}_W = R(\mathbf{q})\, \mathbf{a}_\text{corr} + \mathbf{g}$
4. Kinematic integration:

$$
\mathbf{p}_{k+1} = \mathbf{p}_k + \mathbf{v}_k \Delta t + \tfrac{1}{2}\mathbf{a}_W \Delta t^2
$$

$$
\mathbf{v}_{k+1} = \mathbf{v}_k + \mathbf{a}_W \Delta t
$$

State transition Jacobian $F \in \mathbb{R}^{9\times9}$:

$$
F = \begin{bmatrix}
I_3 & \Delta t\, I_3 & 0 \\
0 & I_3 & -R\, \Delta t \\
0 & 0 & I_3
\end{bmatrix}
$$

Process noise scaled by adaptive factor: $Q = Q_\text{base} \cdot \alpha_\text{proc}$

✅ **Implemented in:** `AdaptiveEKF.predict(gyro, accel, mag, adaptive_factor)`

### 5.3 Measurement Update (Accel Bias Correction)

Measurement model — accelerometer as a gravity+bias sensor:

$$
h(\mathbf{x}) = R^T (-\mathbf{g}) + \mathbf{b}_a
$$

$$
H = \begin{bmatrix} 0_{3\times3} & 0_{3\times3} & I_3 \end{bmatrix} \in \mathbb{R}^{3\times9}
$$

Innovation: $\mathbf{y} = \mathbf{a}_\text{meas} - h(\hat{\mathbf{x}})$

### 5.4 Adaptive Noise Scaling

An innovation window of length $N = 20$ accumulates recent innovations:

$$
\hat{C}_{\mathbf{y}} = \frac{1}{N}\sum_{j=0}^{N-1} \mathbf{y}_{k-j}\, \mathbf{y}_{k-j}^T
$$

Adapted measurement noise:

$$
R = (R_\text{base} + \text{diag}(\hat{C}_{\mathbf{y}}))\cdot \alpha_\text{meas}
$$

This automatically increases $R$ during turbulent or biased periods, reducing the weight of unreliable
measurements.

✅ **Implemented in:** `AdaptiveEKF.correct(accel, adaptive_factor)`

---

## 6. State Vector Comparison

| Layer | State dim | Variables | Attitude representation |
|-------|-----------|-----------|------------------------|
| Plant (`QuadcopterDynamics`) | 12 | pos(3), vel(3), euler(3), omega(3) | ZYX Euler angles |
| `ExtendedKalmanFilter` | 10 | pos(3), vel(3), quat(4) | Quaternion $[w,x,y,z]$ |
| `AdaptiveEKF` | 9 | pos(3), vel(3), accel_bias(3) | Delegated to AHRS |
| `AHRS` | (implicit) | quat, gyro_bias | Quaternion $[w,x,y,z]$ |

The estimator bridges the Euler-angle plant and quaternion-based orientation computation.

---

## 7. How to Choose

| Scenario | Recommended |
|----------|-------------|
| Maximum accuracy; analytical Jacobians available | `ExtendedKalmanFilter` |
| High-rate inner attitude loop; CPU constrained | `AHRS` alone |
| Moderate accuracy + adaptive noise; full position estimate | `AdaptiveEKF` + `AHRS` |
| Future: GPS fusion, position correction | `ExtendedKalmanFilter` (add position update step) |

---

## 8. Example and Test Coverage

| Example | Estimator used | Demonstrated capability |
|---------|------|------------------------|
| [examples/01_imu_ekf_basic.py](../examples/01_imu_ekf_basic.py) | `ExtendedKalmanFilter` | Predict-correct loop on generated trajectory |
| [examples/02_ekf_adaptive.py](../examples/02_ekf_adaptive.py) | `AdaptiveEKF` + `AHRS` | Adaptive estimation under temperature-dependent noise |
| [examples/05_full_pipeline.py](../examples/05_full_pipeline.py) | `AdaptiveEKF` + `AHRS` | Estimated state fed to controller (not truth state) |

| Test | File | What is checked |
|------|------|----------------|
| State propagation | [tests/test_ekf.py](../tests/test_ekf.py) | Quaternion normalization after predict; covariance positive-definite |
| Initialization | [tests/test_ekf.py](../tests/test_ekf.py) | Identity quaternion initial state |

---

## 9. Planned Extensions

### 9.1 GPS Measurement Update

Add a position measurement model:

$$
h_\text{GPS}(\mathbf{x}) = \mathbf{p},\quad H_\text{GPS} = [I_3\; 0_{3\times7}]
$$

Measurement noise $R_\text{GPS} \approx \text{diag}(1, 1, 2)$ m² (typical consumer GPS).

**Implementation plan:** Add `correct_gps(pos_meas)` to `ExtendedKalmanFilter`.

### 9.2 Innovation Gating (Outlier Rejection)

Chi-squared test on innovation norm before applying update:

$$
\mathbf{y}^T S^{-1} \mathbf{y} < \chi^2_{n,\alpha}
$$

where $n$ is measurement dimension and $\alpha$ is confidence level (e.g., 0.997 → 3σ). Reject updates
that fail this gate.

**Implementation plan:** Add `_innovation_gate(y, S, alpha)` helper; apply in `correct_accel` and `correct_mag`.

### 9.3 Error-State EKF (ESEKF)

Represent orientation as nominal $\mathbf{q}$ plus small error $\delta\boldsymbol\theta \in \mathbb{R}^3$:

$$
\mathbf{q} = \hat{\mathbf{q}} \otimes \begin{bmatrix} 1 \\ \delta\boldsymbol\theta/2 \end{bmatrix}
$$

Benefits: avoids quaternion over-parameterization; 15-state error-state: $[\delta\mathbf{p}, \delta\mathbf{v},
\delta\boldsymbol\theta, \mathbf{b}_a, \mathbf{b}_g]$.

### 9.4 Unscented Kalman Filter (UKF)

Replace linearization with sigma-point sampling:

$$
\mathcal{X}^{(i)} = \hat{\mathbf{x}} \pm (\sqrt{(n+\lambda)P})_i
$$

Avoids Jacobian derivation; better accuracy for strongly nonlinear attitude dynamics.

### 9.5 Multi-Rate Fusion

Run predict at IMU rate (~200 Hz), apply measurement updates only when each sensor arrives:

```
t=0ms: predict (gyro)
t=5ms: predict (gyro) + correct_accel
t=10ms: predict (gyro)
t=100ms: predict (gyro) + correct_mag + correct_gps  ← slower sensors
```

---

## 10. Tuning Reference

| Parameter | Effect | Direction |
|-----------|--------|-----------|
| Increase `R_accel` | Trust accelerometer less | More gyro-dominated attitude |
| Decrease `R_accel` | Trust accelerometer more | Faster correction but noisier |
| Increase `Q[6:10]` | Expect rapid attitude change | Higher covariance, larger K for attitude |
| `AHRS.accel_weight` ↑ | Stronger gravity correction | Faster level convergence; more noise |
| `AdaptiveEKF` window size ↑ | Slower noise adaptation | Smoother but lags sudden changes |

---

## 11. Further Reading

- Mahony et al., "Nonlinear Complementary Filters on the Special Orthogonal Group", IEEE TAC 2008 — AHRS theory
- Madgwick, "An Efficient Orientation Filter", 2010 — Practical CF variant
- Joan Solà, "Quaternion Kinematics for Error-State KF", 2017 — ESEKF reference
- Mourikis & Roumeliotis, "A Multi-State Constraint Kalman Filter", ICRA 2007 — Multi-sensor fusion


