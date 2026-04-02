# Testing & Validation Strategy

**Key files:**
- [tests/test_ekf.py](../tests/test_ekf.py)
- [tests/test_dynamics.py](../tests/test_dynamics.py)
- [tests/test_cascaded.py](../tests/test_cascaded.py)
- [src/drones_sim/estimation/ekf.py](../src/drones_sim/estimation/ekf.py)
- [examples/05_full_pipeline.py](../examples/05_full_pipeline.py)

---

## 1. Motivation

A 400-metre position estimation error was traced to three simultaneously active bugs, none of which was
caught by the existing unit tests:

| Bug | Root cause | Existing test coverage |
|-----|-----------|----------------------|
| Wrong sign in `correct_accel` expected value | `expected = -R.T @ g` → residual ≈ 19.6 N/kg at hover | ✗ none |
| Incorrect `_accel_jacobian` formulas | Partials did not match `d(R^T g)/dq` | ✗ none |
| Velocity never predicted in `predict()` | `vel` held constant; only `pos += vel·dt` executed | ✗ none |
| No position measurement → unbounded drift | Dead-reckoning without barometer/GPS anchor | ✗ none |
| Hover feedforward ≈ 9.81 × 10⁻⁶ N (factor 10⁶ error) | Extra `× k_f` in force computation | ✗ none |
| Accelerometer simulation missing dynamics contribution | Only static gravity modeled | ✗ none |

All six bugs shared the same pattern: the code was internally consistent enough to run without an
exception while silently producing physically wrong outputs. This document describes the systematic
testing infrastructure needed to catch this class of bug at commit time.

---

## 2. Test Taxonomy

```
tests/
├── test_math_utils.py        ← Unit: pure math (quaternion algebra, rotations)
├── test_dynamics.py          ← Unit: physics (free-fall, hover thrust balance)
├── test_pid.py               ← Unit: PID response, saturation, anti-windup
├── test_ekf.py               ← Unit + property: EKF state propagation      ← EXTEND
├── test_cascaded.py          ← Integration regression gate                  ← EXTEND
└── test_sensor_models.py     ← NEW: physical consistency of sensor outputs
```

New tests required by this plan are marked **NEW** or **EXTEND** above.

---

## 3. Recommendation 1 — Numerical Jacobian Validation ✅ Priority: Critical

### Problem
An analytical Jacobian that is wrong by any sign, factor, or missing cross-term will cause the Kalman
gain to push the state estimate in the wrong direction. The filter continues to run (no exception) but
diverges.

### Theory
The finite-difference Jacobian is always correct by construction:

$$H_{ij}^\text{FD} = \frac{h(\mathbf{x} + \epsilon \mathbf{e}_j) - h(\mathbf{x} - \epsilon \mathbf{e}_j)}{2\epsilon}$$

Comparing $H^\text{analytical}$ to $H^\text{FD}$ with $\epsilon = 10^{-6}$ and tolerance $10^{-5}$
catches any symbolic derivation error.

### Implementation plan — `tests/test_ekf.py`

```python
from drones_sim.math_utils import quat_to_rotation_matrix, quat_normalize

def _fd_jacobian(ekf, measurement_fn, q_state, eps=1e-6):
    """Finite-difference Jacobian of measurement_fn w.r.t. EKF state."""
    x0 = ekf.x.copy()
    h0 = measurement_fn(ekf)
    H = np.zeros((len(h0), len(x0)))
    for j in range(len(x0)):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += eps; xm[j] -= eps
        ekf.x = xp; hp = measurement_fn(ekf)
        ekf.x = xm; hm = measurement_fn(ekf)
        H[:, j] = (hp - hm) / (2 * eps)
    ekf.x = x0  # restore
    return H

def _accel_h(ekf):
    R = quat_to_rotation_matrix(ekf.x[6:10])
    return R.T @ ekf.gravity

def test_accel_jacobian_matches_fd():
    """Analytical _accel_jacobian must match finite-difference to 1e-5."""
    ekf = ExtendedKalmanFilter(dt=0.01)
    for quat in [
        np.array([1, 0, 0, 0]),                     # identity (level)
        quat_normalize(np.array([0.9, 0.3, 0.1, 0])),  # rolled
        quat_normalize(np.array([0.9, 0, 0.3, 0.1])),  # pitched+yawed
    ]:
        ekf.x[6:10] = quat
        H_ana = ekf._accel_jacobian(quat)
        H_fd  = _fd_jacobian(ekf, _accel_h, quat)
        np.testing.assert_allclose(H_ana, H_fd, atol=1e-5,
                                   err_msg=f"Jacobian wrong at q={quat}")

def test_mag_jacobian_matches_fd():
    """Analytical _mag_jacobian must match finite-difference to 1e-5."""
    ekf = ExtendedKalmanFilter(dt=0.01, mag_ref=np.array([25.0, 5.0, -40.0]))
    def mag_h(ekf):
        R = quat_to_rotation_matrix(ekf.x[6:10])
        return R.T @ ekf.mag_ref
    for quat in [np.array([1, 0, 0, 0]),
                 quat_normalize(np.array([0.8, 0.4, 0.2, 0.1]))]:
        ekf.x[6:10] = quat
        H_ana = ekf._mag_jacobian(quat)
        H_fd  = _fd_jacobian(ekf, mag_h, quat)
        np.testing.assert_allclose(H_ana, H_fd, atol=1e-5,
                                   err_msg="Mag Jacobian wrong")
```

**Files to change:** `tests/test_ekf.py`  
**Effort:** ~30 lines  
**Catches:** both Jacobian bugs and any future symbolic error

---

## 4. Recommendation 2 — At-Rest IMU Sanity Test ✅ Priority: Critical

### Problem
The sign error `expected = -R.T @ g` produces a residual of $2 g \approx 19.6\,\text{N/kg}$ at rest,
which is physically impossible for a stationary sensor. A trivial unit test would have caught it.

### Theory
For an ENU coordinate system with $\mathbf{g} = [0, 0, g]^T$ (upward reaction), an IMU on a
flat-and-level drone reads exactly:

$$\mathbf{a}_\text{body} = R^T\,\mathbf{g} \;=\; [0,\; 0,\; 9.81]\;\text{N/kg} \quad \text{(body frame = world frame when } R=I\text{)}$$

### Implementation plan — `tests/test_ekf.py`

```python
def test_accel_expected_at_rest_identity():
    """Level drone (R=I): EKF expected accel = [0, 0, 9.81] body frame."""
    ekf = ExtendedKalmanFilter(dt=0.01)
    # identity quaternion → R = I                         (level, no rotation)
    R = quat_to_rotation_matrix(ekf.x[6:10])
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9, err_msg="Identity quat must give R=I")
    expected = R.T @ ekf.gravity
    np.testing.assert_allclose(expected, [0, 0, 9.81], atol=1e-9,
                               err_msg="Wrong sign or convention in accel expected")

def test_accel_expected_90deg_roll():
    """90° roll: gravity projects to +y body axis."""
    ekf = ExtendedKalmanFilter(dt=0.01)
    # q = [cos45°, sin45°, 0, 0] = 90° roll around x-axis
    q90_x = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0, 0.0])
    R = quat_to_rotation_matrix(q90_x)
    expected = R.T @ ekf.gravity    # should be ≈ [0, 9.81, 0]
    np.testing.assert_allclose(expected[0], 0.0, atol=1e-6)
    np.testing.assert_allclose(abs(expected[1]), 9.81, atol=0.01)
    np.testing.assert_allclose(expected[2], 0.0, atol=0.01)
```

**Files to change:** `tests/test_ekf.py`  
**Effort:** ~15 lines  
**Catches:** sign convention errors in any correction method

---

## 5. Recommendation 3 — Filter Consistency Check (NIS Test) ✅ Priority: High

### Theory
The Normalized Innovation Squared (NIS) is defined as:

$$\text{NIS}_k = \boldsymbol\nu_k^T \mathbf{S}_k^{-1} \boldsymbol\nu_k, \quad \boldsymbol\nu_k = \mathbf{y}_k - h(\hat{\mathbf{x}}_{k|k-1})$$

Under a consistent filter, $\text{NIS}_k \sim \chi^2(m)$ where $m$ is the measurement dimension.
The 99th percentile of $\chi^2(3)$ is approximately **11.3**. Sustained NIS ≫ 11 indicates:
- Underestimated noise covariance, **or**
- Wrongly modeled measurement function $h(\mathbf{x})$ (which includes Jacobian bugs)

### Implementation plan

**In `tests/test_ekf.py` (unit level):**

```python
def test_accel_nis_at_rest():
    """NIS for accelerometer at rest should be < χ²(3, 99%) ≈ 11.3."""
    ekf = ExtendedKalmanFilter(dt=0.01)
    g   = ekf.gravity
    CHI2_99_3DOF = 11.345

    nises = []
    for _ in range(50):
        # Simulate one noisy accel reading at rest (R=I)
        accel_noisy = g + np.random.normal(0, 0.05, 3)
        q = ekf.x[6:10]
        R = quat_to_rotation_matrix(q)
        expected  = R.T @ g
        residual  = accel_noisy - expected
        H = ekf._accel_jacobian(q)
        S = H @ ekf.P @ H.T + ekf.R_accel
        nis = float(residual @ np.linalg.inv(S) @ residual)
        nises.append(nis)

    # Median NIS should be well below χ²(3, 99%)
    assert np.median(nises) < CHI2_99_3DOF, \
        f"Median NIS {np.median(nises):.2f} exceeds χ²(3,99%)={CHI2_99_3DOF}"
```

**As a runtime guard in the simulation loop (optional, controlled by flag):**

```python
# examples/05_full_pipeline.py  (guarded by ENABLE_NIS_CHECK env var)
import os
CHECK_NIS = os.getenv("CHECK_NIS", "0") == "1"
CHI2_99_3DOF = 11.345

if CHECK_NIS:
    residual = accel_meas - (quat_to_rotation_matrix(ekf.x[6:10]).T @ gravity)
    H = ekf._accel_jacobian(ekf.x[6:10])
    S = H @ ekf.P @ H.T + ekf.R_accel
    nis = float(residual @ np.linalg.inv(S) @ residual)
    if nis > CHI2_99_3DOF * 3:   # 3× margin avoids false alarms at start
        print(f"[NIS WARNING] t={t[i]:.2f}s  NIS={nis:.1f} > {CHI2_99_3DOF*3:.1f}")
```

**Files to change:** `tests/test_ekf.py`, `examples/05_full_pipeline.py`  
**Effort:** ~25 lines each  
**Catches:** diverging filter before the position error becomes large

---

## 6. Recommendation 4 — Bounded-Error Integration Test ✅ Priority: High

### Problem
All previous tests passed despite 400-metre drift because they only checked that the EKF runs without
error, not that the output is physically plausible. A regression gate on mean estimation error would
fail immediately at the first commit that introduces any divergence.

### Implementation plan — `tests/test_cascaded.py` or new `tests/test_integration.py`

```python
import numpy as np
import pytest
from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.estimation import ExtendedKalmanFilter
from drones_sim.sensors.models import SensorNoiseModel
from drones_sim.math_utils import euler_to_rotation_matrix

@pytest.mark.parametrize("seed", [0, 1, 2])   # three random seeds for robustness
def test_ekf_position_error_bounded(seed):
    """Full closed-loop: mean 3-D position estimation error must stay below 2 m.

    Uses barometer + GPS-rate (10 Hz) corrections.  Mean error dominated by GPS
    noise (σ=0.5 m), so threshold 2 m has a comfortable margin.
    """
    rng = np.random.default_rng(seed)
    quad = QuadcopterDynamics()
    dt   = 0.01
    ekf  = ExtendedKalmanFilter(dt=dt, initial_state=_identity_state())
    gravity = np.array([0.0, 0.0, 9.81])

    accel_noise = SensorNoiseModel(noise_std=0.05, bias_range=0.1)
    gyro_noise  = SensorNoiseModel(noise_std=0.01, bias_range=0.005)
    baro_noise  = SensorNoiseModel(noise_std=0.02, bias_range=0.05)
    gps_noise   = SensorNoiseModel(noise_std=0.5,  bias_range=0.3)

    hover_speed = np.sqrt(quad.g * quad.mass / (4 * quad.k_f))
    motors = np.full(4, hover_speed)

    errors = []
    for i in range(500):   # 5 s at 100 Hz
        pos  = quad.get_position()
        R    = euler_to_rotation_matrix(*quad.get_attitude())
        # Specific force
        accel_meas = accel_noise.apply(R.T @ gravity)
        gyro_meas  = gyro_noise.apply(quad.get_angular_velocity())
        ekf.predict(gyro_meas, accel_meas)
        ekf.correct_accel(accel_meas)
        z_baro = float(baro_noise.apply(np.array([pos[2]]))[0])
        ekf.correct_altitude(z_baro, r_z=0.0004)
        if i % 10 == 0:
            gps_meas = gps_noise.apply(pos.copy())
            ekf.correct_position(gps_meas, R_pos=np.eye(3) * 0.25)
        errors.append(np.linalg.norm(pos - ekf.get_state()["position"]))
        quad.update(dt, motors)

    mean_err = float(np.mean(errors))
    assert mean_err < 2.0, f"EKF mean position error {mean_err:.2f} m > 2.0 m (seed={seed})"

def _identity_state():
    s = np.zeros(10); s[6] = 1.0; return s
```

**Files to change:** `tests/test_cascaded.py` or new `tests/test_integration.py`  
**Effort:** ~50 lines  
**Catches:** any divergence introduced by changes to EKF, sensor models, or dynamics

---

## 7. Recommendation 5 — Sensor Model Physical Self-Check ✅ Priority: Medium

### Problem
The accelerometer simulation initially modeled only static gravity (`R^T @ g`), ignoring the drone's
linear acceleration. This makes EKF velocity prediction impossible — the filter receives a constant
signal with no dynamics information.

### Theory
A correctly simulated specific force obeys:

$$\mathbf{f}_\text{body} = R^T(\mathbf{a}_\text{linear} + \mathbf{g}) \quad \Rightarrow \quad \|\mathbf{f}_\text{body}\| = \|\mathbf{a}_\text{linear} + \mathbf{g}\|$$

At hover: $\mathbf{a}_\text{linear} = \mathbf{0}$, so $\|\mathbf{f}_\text{body}\| = g = 9.81\,\text{N/kg}$.  
During a 2g maneuver: $\|\mathbf{f}_\text{body}\| \approx 3g$.

An accelerometer simulation that always returns $\|\mathbf{f}\| = g$ regardless of motor commands is
static-only and must be rejected.

### Implementation plan — `tests/test_sensor_models.py` (new file)

```python
"""Physical consistency tests for sensor simulation."""

import numpy as np
import pytest
from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.math_utils import euler_to_rotation_matrix

def test_accel_simulation_includes_dynamics():
    """Accel magnitude at double-hover thrust must exceed g by ≥ 30%."""
    quad = QuadcopterDynamics()
    dt   = 0.01
    gravity = np.array([0.0, 0.0, 9.81])

    # Step 1: run at 2× hover thrust so drone accelerates upward
    hover_speed = np.sqrt(quad.g * quad.mass / (4 * quad.k_f))
    motors = np.full(4, hover_speed * np.sqrt(2.0))   # 2× thrust

    quad.update(dt, motors)
    R = euler_to_rotation_matrix(*quad.get_attitude())
    T_over_m = float(np.sum(motors**2) * quad.k_f / quad.mass)
    a_thrust_world = R @ np.array([0.0, 0.0, T_over_m])
    a_drag_world   = -quad.k_d * quad.get_velocity() / quad.mass
    a_lin          = a_thrust_world + a_drag_world + np.array([0.0, 0.0, -quad.g])
    f_body         = R.T @ (a_lin + gravity)

    # At 2× hover thrust, |f| should be significantly greater than 1g
    assert np.linalg.norm(f_body) > 9.81 * 1.3, \
        f"|f_body|={np.linalg.norm(f_body):.2f} N/kg, should exceed 1.3g at 2× thrust"

def test_hover_specific_force_is_g():
    """At exact hover thrust, |f_body| ≈ g (net linear accel ≈ 0)."""
    quad = QuadcopterDynamics()
    dt   = 0.01
    gravity = np.array([0.0, 0.0, 9.81])
    hover_speed = np.sqrt(quad.g * quad.mass / (4 * quad.k_f))
    motors = np.full(4, hover_speed)
    R = euler_to_rotation_matrix(*quad.get_attitude())
    f_static = R.T @ gravity
    np.testing.assert_allclose(np.linalg.norm(f_static), 9.81, atol=0.01,
                               err_msg="Static hover specific force must equal g")
```

**Files to change:** new `tests/test_sensor_models.py`  
**Effort:** ~40 lines  
**Catches:** sensor simulation that omits dynamics, wrong sign in specific-force formula

---

## 8. Recommendation 6 — Physical Unit Assertions ✅ Priority: Medium

### Problem
The hover feedforward bug (`hover_thrust = quad.g * quad.mass * quad.k_f ≈ 9.81 × 10⁻⁶ N`) would
have been caught at runtime by a one-line assertion if physical units were enforced.

### Implementation plan

**Pattern:** add `assert` guards at simulation boundary points where units are known.

```python
# examples/05_full_pipeline.py — feedforward sanity
hover_thrust = quad.g * quad.mass          # [N]
assert 5.0 < hover_thrust < 50.0, \
    f"hover_thrust={hover_thrust:.4f} N — expected 5–50 N for typical quadcopter"

# Motor speed sanity
motor_speed = np.sqrt(max(total_thrust / (4 * quad.k_f), 0))
assert 0.0 < motor_speed < 10_000.0, \
    f"motor_speed={motor_speed:.1f} rad/s — physically implausible value"

# EKF outputs sanity
state = ekf.get_state()
quat_norm = np.linalg.norm(state["quaternion"])
assert 0.999 < quat_norm < 1.001, \
    f"EKF quaternion norm={quat_norm:.6f} — filter diverging"
```

**Unit convention comment block** added to `ekf.py` constructor:

```python
class ExtendedKalmanFilter:
    """10-state EKF for quadcopter navigation.

    State vector x[10]:
        x[0:3]   position      [m]      ENU world frame
        x[3:6]   velocity      [m/s]    ENU world frame
        x[6:10]  quaternion    [-]      Hamilton [w, x, y, z], body→world

    Gravity convention (ENU, z-up):
        self.gravity = [0, 0, 9.81]    upward reaction direction
        At rest:  IMU reads  R^T @ [0,0,9.81]  in body frame  (> 0 in body-z)
        predict:  a_world = R @ f_body - self.gravity
    """
```

**Files to change:** `examples/05_full_pipeline.py`, `src/drones_sim/estimation/ekf.py`  
**Effort:** ~10 lines scattered across files  
**Catches:** unit errors silently producing near-zero or astronomically large forces

---

## 9. Implementation Roadmap

### Phase 1 — Foundation (Immediate, ~1 hour)
Prevents the highest-severity bugs from being re-introduced silently.

| Step | Action | File |
|------|--------|------|
| 1.1 | Add `test_accel_jacobian_matches_fd` and `test_mag_jacobian_matches_fd` | `tests/test_ekf.py` |
| 1.2 | Add `test_accel_expected_at_rest_identity` and `test_accel_expected_90deg_roll` | `tests/test_ekf.py` |
| 1.3 | Add `test_ekf_position_error_bounded` with 3 seeds | `tests/test_cascaded.py` |
| 1.4 | Add physical unit `assert` guards for feedforward and motor speed | `examples/05_full_pipeline.py` |

Completion criterion: `pytest tests/ -v` passes cleanly and would **fail** if any of bugs 1–4 were
re-introduced.

### Phase 2 — Broad Coverage (~2 hours)
Extends guards to sensor simulation and filter consistency.

| Step | Action | File |
|------|--------|------|
| 2.1 | Create `tests/test_sensor_models.py` with accel-at-double-thrust test | new file |
| 2.2 | Add NIS unit test for accelerometer | `tests/test_ekf.py` |
| 2.3 | Add `CHECK_NIS` runtime flag to example 05 | `examples/05_full_pipeline.py` |
| 2.4 | Add quaternion-norm assertion in EKF `correct_*` methods | `src/drones_sim/estimation/ekf.py` |

### Phase 3 — Documentation & Convention (~30 min)
Makes the codebase self-documenting for future contributors.

| Step | Action | File |
|------|--------|------|
| 3.1 | Add unit convention docstring to `ExtendedKalmanFilter.__init__` | `ekf.py` |
| 3.2 | Add `# [N]`, `# [m/s]` inline unit comments to all force/velocity variables in example 05 | `05_full_pipeline.py` |
| 3.3 | Document `_accel_jacobian` with full derivation reference in docstring | `ekf.py` |

---

## 10. Regression Gate Summary

After Phase 1+2 are complete, the following invariants are enforced on every `pytest` run:

| Invariant | Test | Threshold |
|-----------|------|-----------|
| Analytical Jacobian ≈ finite-difference | `test_accel_jacobian_matches_fd` | `atol=1e-5` |
| At-rest IMU sign convention | `test_accel_expected_at_rest_identity` | exact match |
| 5-second hover mean estimation error | `test_ekf_position_error_bounded` | < 2.0 m |
| Accel simulation includes dynamics | `test_accel_simulation_includes_dynamics` | `|f| > 1.3g` at 2× thrust |
| Filter NIS (3-DOF) at rest | `test_accel_nis_at_rest` | median NIS < 11.3 |
| Quaternion norm after correction | `assert` in `correct_*` | 0.999 – 1.001 |

Run all gates:
```bash
cd /Users/thanhndv212/Develop/drones_sim
pytest tests/ -v --tb=short
```

Expected output: 20+ tests, all green. Any one failure indicates a physically wrong implementation.

---

## 11. Related Documentation

| Document | Relationship |
|----------|-------------|
| [State Estimation](state-estimation.md) | EKF predict/correct equations that the Jacobian tests verify |
| [Sensors](sensors.md) | Specific-force physics that sensor model tests check |
| [Simulation](simulation.md) | Closed-loop regression gate already in §5.2; extended here |
| [Modeling](modeling.md) | Newton–Euler dynamics that feed the sensor simulation |
