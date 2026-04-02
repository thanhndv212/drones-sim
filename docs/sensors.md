# Sensor Simulation

**Implementation:** [src/drones_sim/sensors/imu.py](../src/drones_sim/sensors/imu.py),
[src/drones_sim/sensors/models.py](../src/drones_sim/sensors/models.py)

---

## 1. Overview

The sensor simulation layer converts ground-truth trajectory data into realistic noisy IMU
measurements that feed the state estimators. It models the key corruptions present in real MEMS
sensors: Gaussian noise, constant bias, scale-factor errors, and optional temperature drift.

**Current sensor types:** IMU (accelerometer + gyroscope + magnetometer).

**Planned additions:** GPS, barometer, optical flow, rangefinder (see Section 8).

---

## 2. Measurement Model

### 2.1 General Noise Stack

For each sensor axis, the measurement is:

$$
y_k = s_i\, x_k + b_i + \eta_k(T)
$$

where:

- $x_k$: true physical quantity (acceleration, angular rate, or magnetic field)
- $s_i \sim \mathcal{U}(s_\min, s_\max)$: per-axis scale factor (randomized at construction)
- $b_i \sim \mathcal{U}(-b_\text{range}, b_\text{range})$: per-axis constant bias (randomized at construction)
- $\eta_k \sim \mathcal{N}(0,\; \sigma^2 \cdot \gamma(T)^2)$: temperature-scaled Gaussian noise

### 2.2 Temperature Drift

Temperature follows a sinusoidal profile over the trajectory duration:

$$
T(t) = T_\text{base} + A\sin\!\left(\frac{2\pi t}{t_\text{dur}}\right)
$$

with $T_\text{base} = 25°C$, $A = 10°C$ (amplitude).

Per-sensor temperature offset:

$$
\delta\mathbf{x}_\text{temp}(T) = \mathbf{c} \cdot (T - T_\text{base})
$$

where $\mathbf{c} \in \mathbb{R}^3$ are per-axis sensitivity coefficients randomized at construction:

| Sensor | Coefficient range |
|--------|------------------|
| Accel | $c_i \sim \mathcal{U}(-0.002, 0.002)$ m/s²/°C |
| Gyro | $c_i \sim \mathcal{U}(-0.0005, 0.0005)$ rad/s/°C |
| Mag | $c_i \sim \mathcal{U}(-0.05, 0.05)$ μT/°C |

Noise scaling factor:

$$
\gamma(T) = 1 + 0.01 \cdot \frac{T - T_\text{base}}{10}
$$

✅ **Implemented in:** `TemperatureModel` (`sensors/models.py`)

---

## 3. Sensor Noise Model — `SensorNoiseModel`

```python
@dataclass
class SensorNoiseModel:
    noise_std: float           # σ for Gaussian noise
    bias_range: float          # bias drawn from U(-range, +range)
    scale_factor_range: tuple  # scale drawn from U(lo, hi)
    bias: NDArray              # (3,) — set in __post_init__
    scale_factor: NDArray      # (3,) — set in __post_init__
```

### 3.1 Apply Method

```python
def apply(self, true_value: NDArray, temp_factor: float = 1.0) -> NDArray:
    noisy = true_value * self.scale_factor + self.bias
    noisy += np.random.normal(0, self.noise_std * temp_factor, 3)
    return noisy
```

### 3.2 Default Noise Parameters (from `IMUConfig`)

| Sensor | Noise σ | Bias range | Scale range |
|--------|---------|------------|-------------|
| Accel | 0.05 m/s² | ±0.10 m/s² | (0.98, 1.02) |
| Gyro | 0.01 rad/s | ±0.005 rad/s | (0.99, 1.01) |
| Mag | 0.5 μT | ±1.0 μT | (0.97, 1.03) |

✅ **Implemented in:** `SensorNoiseModel` (`sensors/models.py`)

---

## 4. IMU Simulator — `IMUSimulator`

### 4.1 Data Flow

```
TrajectoryData
    └─ orientation_quat [w,x,y,z]  ──► scipy → R (rotation matrix, body←world)
    └─ linear acceleration (world)
    └─ angular_velocity (body)
         │
         ▼
    gravity_body    = R^T · g_world        (accelerometer gravity component)
    lin_accel_body  = R^T · (−accel_world)  (reaction force)
    true_accel      = gravity_body + lin_accel_body
         │
         ▼
    [Optional] temperature offset applied to true_accel, true_gyro, true_mag
         │
         ▼
    SensorNoiseModel.apply(true_*, temp_factor)  ── for each of accel, gyro, mag
         │
         ▼
    IMUData (t, accel, gyro, mag, temperature?)
```

> **Convention note:** The accelerometer measures the **specific force** (reaction to gravity + linear acceleration),
> not just linear acceleration. In hover, it reads $+g$ upward rather than 0.

### 4.2 Config Dataclass

```python
@dataclass
class IMUConfig:
    accel_noise_std: float = 0.05      # m/s²
    accel_bias_range: float = 0.1      # m/s²
    accel_scale: tuple = (0.98, 1.02)

    gyro_noise_std: float = 0.01       # rad/s
    gyro_bias_range: float = 0.005     # rad/s
    gyro_scale: tuple = (0.99, 1.01)

    mag_noise_std: float = 0.5         # μT
    mag_bias_range: float = 1.0        # μT
    mag_scale: tuple = (0.97, 1.03)

    gravity: NDArray                   # default [0, 0, 9.81]
    mag_field_ref: NDArray             # default [25, 5, −40] μT
    enable_temperature: bool = False
```

### 4.3 Output Dataclass

```python
@dataclass
class IMUData:
    t: NDArray          # (N,)
    accel: NDArray      # (N, 3)  m/s²
    gyro: NDArray       # (N, 3)  rad/s
    mag: NDArray        # (N, 3)  μT
    temperature: NDArray | None  # (N,) °C if enabled
    accel_bias: NDArray  # ground truth, for validation
    gyro_bias: NDArray
    mag_bias: NDArray
```

✅ **Implemented in:** `IMUSimulator.simulate(traj)` (`sensors/imu.py`)

---

## 5. Signal Processing for Drone Sensors

### 5.1 Aliasing

Sampling at rate $f_s$ can represent only frequencies below $f_s/2$ (Nyquist). Motor vibrations near the
propulsion frequency can alias onto position/attitude signals:

- Anti-aliasing filter before sampling
- Typical IMU sample rates: 200–1000 Hz; motor freqs: 50–400 Hz

### 5.2 Low-Pass Filtering

Butterworth filter (2nd order) for accelerometer smoothing:

$$
H(s) = \frac{\omega_c^2}{s^2 + \sqrt{2}\,\omega_c\, s + \omega_c^2}
$$

Python: `scipy.signal.butter(2, cutoff, fs=sample_rate)` + `filtfilt` for zero-phase.

### 5.3 Complementary Filter

Combines gyro (high-frequency) with accelerometer (low-frequency attitude reference):

$$
\hat\phi_k = \alpha\!\left(\hat\phi_{k-1} + \omega_\text{gyro}\, \Delta t\right) + (1-\alpha)\, \phi_\text{accel}
$$

Typical $\alpha = 0.98$.

### 5.4 Notch Filter

Eliminates motor vibration frequency $f_m$:

$$
H_\text{notch}(z) = \frac{1 - 2\cos(2\pi f_m/f_s)\, z^{-1} + z^{-2}}{1 - 2r\cos(2\pi f_m/f_s)\, z^{-1} + r^2 z^{-2}}
$$

where $r < 1$ controls notch width ($r \to 1$ is narrower).

### 5.5 FFT Vibration Analysis

Identify mechanical resonances from accelerometer data:

```python
import numpy as np
freqs = np.fft.rfftfreq(N, d=1/sample_rate)
spectrum = np.abs(np.fft.rfft(accel_x)) / N
dominant = freqs[np.argmax(spectrum)]  # Hz
```

### 5.6 Kalman Filter as 1-D Smoother

Optimal estimator for a scalar measurement with noise $r$ and process noise $q$:

$$
P_k = P_{k-1} + q
$$
$$
K_k = P_k / (P_k + r)
$$
$$
\hat{x}_k = \hat{x}_{k-1} + K_k(z_k - \hat{x}_{k-1}),\quad P_k \leftarrow (1 - K_k) P_k
$$

---

## 6. Sensor Fusion Architecture

The sensor pipeline feeds the EKF/AHRS estimators:

```
IMUData.accel[k] ────► EKF.correct_accel()   (orientation correction)
IMUData.gyro[k]  ────► EKF.predict()          (quaternion propagation)
IMUData.mag[k]   ────► EKF.correct_mag()      (yaw/heading correction)
                  └──► AHRS.update()           (complementary filter path)
```

Practical multi-rate separation:

| Sensor | Typical rate | Role |
|--------|-------------|------|
| Gyro | 200–1000 Hz | High-rate predict |
| Accel | 200–1000 Hz | Tilt correction |
| Mag | 10–100 Hz | Heading correction |
| GPS | 1–10 Hz | Position correction (planned) |
| Baro | 10–50 Hz | Altitude correction (planned) |

---

## 7. Current Limitations

| Limitation | Description |
|-----------|-------------|
| White noise only | Real IMUs have random walk and flicker noise (colored noise) |
| Constant bias | Real sensors have slowly-varying bias (Gauss-Markov process) |
| Diagonal scale factor | Cross-axis coupling and misalignment not modeled |
| No magnetometer distortion | Hard-iron / soft-iron distortions absent |
| No fault injection | No dropout, stuck-at, or spike modes |
| IMU only | No GPS, barometer, optical flow |

---

## 8. Planned Extensions

### 8.1 Bias Random Walk (Gauss-Markov)

A more realistic bias model uses a first-order Gauss-Markov process:

$$
\dot{b} = -\frac{1}{\tau_b} b + \sigma_b w, \quad w \sim \mathcal{N}(0, 1)
$$

Discrete form:

$$
b_{k+1} = e^{-\Delta t/\tau_b} b_k + \sigma_b \sqrt{1 - e^{-2\Delta t/\tau_b}}\, \eta_k
$$

Typical $\tau_b \approx 100$–$1000$ s for MEMS gyroscopes.

**Implementation plan:** Replace static `bias` in `SensorNoiseModel` with a state variable updated
in `apply()`.

### 8.2 Full Calibration Matrix

Replace diagonal scale factor with a full $3\times3$ calibration matrix $C$:

$$
\mathbf{y} = C\,\mathbf{x} + \mathbf{b} + \boldsymbol\eta
$$

$C$ captures cross-axis coupling and axis misalignment.

### 8.3 Magnetometer Hard/Soft-Iron Model

$$
\mathbf{m}_\text{meas} = W\,\mathbf{m}_\text{true} + \mathbf{v}
$$

- $\mathbf{v}$: hard-iron offset (constant additive distortion from nearby magnets)
- $W$: soft-iron matrix (shape distortion of field)

Standard calibration recovers $W^{-1}$ and $\mathbf{v}$ from an ellipsoid fit.

### 8.4 GPS Simulator

```python
@dataclass
class GPSConfig:
    position_noise_std: float = 1.0   # m (CEP)
    velocity_noise_std: float = 0.1   # m/s
    update_rate: float = 5.0          # Hz
    dropout_probability: float = 0.0

class GPSSimulator:
    def simulate(self, traj: TrajectoryData) -> GPSData: ...
```

### 8.5 Barometer Simulator

Models altitude from atmospheric pressure:

$$
h \approx \frac{R_\text{air} T}{Mg}\ln\!\left(\frac{P_0}{P}\right),\quad P = P_0 e^{-Mgh/(R_\text{air}T)}
$$

Add Gaussian noise ($\sigma \approx 0.5$ m) and a slow drift term.

### 8.6 Fault Injection

Add a `FaultMode` enum to `IMUConfig`:

- `DROPOUT`: zero output for N samples
- `STUCK`: hold last value
- `SPIKE`: occasional large outlier
- `BIAS_JUMP`: sudden step change in bias

---

## 9. Example and Test Coverage

| Example | Features demonstrated |
|---------|----------------------|
| [examples/01_imu_ekf_basic.py](../examples/01_imu_ekf_basic.py) | `IMUSimulator` → `ExtendedKalmanFilter` basic loop |
| [examples/02_ekf_adaptive.py](../examples/02_ekf_adaptive.py) | `enable_temperature=True` → `AdaptiveEKF` noise adaptation |
| [examples/05_full_pipeline.py](../examples/05_full_pipeline.py) | `SensorNoiseModel` applied inline in closed-loop |

---

## 10. Further Reading

- Titterton & Weston, *Strapdown Inertial Navigation Technology*, 2nd ed. — IMU error models
- Allan Deviation for characterizing IMU noise (IEEE Std 952)
- Woodman, "An Introduction to Inertial Navigation", University of Cambridge TR 2007
- Gebre-Egziabher et al., "A Non-Linear, Two-Step Estimation Algorithm for Calibrating Solid-State Strapdown Magnetometers", 2001


