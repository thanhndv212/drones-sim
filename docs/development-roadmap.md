# Development Roadmap — `drones_sim`

**Status:** Living document — last updated 2026-09-19
**Scope:** Repository assessment, consolidated milestone roadmap, and reinforcement-learning pipeline design.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Repository Assessment](#2-repository-assessment)
3. [Delivery Roadmap](#3-delivery-roadmap)
4. [Reinforcement Learning Pipeline](#4-reinforcement-learning-pipeline)
5. [Cross-Cutting Concerns](#5-cross-cutting-concerns)

---

## 1. Executive Summary

`drones_sim` is a pure-Python quadcopter simulation framework (~5,500 LOC) implementing the full
robotics loop: **trajectory → dynamics → sensors → estimation → control → visualization**. It has no
heavyweight simulator dependency (no Gazebo/ROS/PyBullet), making it ideal for algorithm prototyping
and teaching.

**Maturity highlights:** analytical EKF Jacobians with Joseph-form covariance, C³-continuous minimum-snap
trajectory planning, Gauss-Markov bias random walks in the sensor stack, hover-linearized LQR, and
synchronized telemetry and `.rrd` replay via Rerun by default, with Viser retained for optional mission controls.

**Foundation status:** the quaternion plant, CI regression suite, disturbance framework, aligned
simulation results, telemetry logging, and the baseline RL pipeline are implemented. Rerun is the
default visualization and replay backend.

**The three highest-leverage next steps are now:**

1. **Model predictive control** with explicit actuator and flight-envelope constraints
2. **Monte Carlo robustness evaluation** across seeds, disturbances, and model uncertainty
3. **RL domain randomization and curriculum learning** for a credible sim-to-real path

The baseline RL stack (PyTorch + Stable-Baselines3 + Gymnasium) is implemented. Section 3 is the
single source of truth for delivery status; §4 documents the RL design.

---

## 2. Repository Assessment

### 2.1 Architecture at a Glance

```
Trajectory ─► Dynamics ─► Sensors ─► Estimation ─► Control ─► Visualization
(reference)  (13-state   (IMU/GPS/   (16-state EKF,  (geometric,     (Rerun 3D,
              quaternion  baro, temp) AdaptiveEKF,    cascaded PID,   matplotlib,
              RK4 plant)               AHRS)            LQR)            optional Viser)
```

| Module | Path | Assessment |
|---|---|---|
| Math utils | `math_utils.py` | Quaternion (wxyz) algebra, rotation matrices, Euler conversions |
| Trajectory | `trajectory.py` | Hover-accel-cruise, circular, waypoint, **minimum-snap QP** |
| Dynamics | `dynamics/` | 13-state quaternion Newton–Euler plant, RK4, motor lag, disturbances, drag, ground effect |
| Sensors | `sensors/` | Gauss-Markov bias random walk, scale-factor, temperature drift, GPS dropout |
| Estimation | `estimation/` | 16-state EKF with **analytical Jacobians**, AdaptiveEKF, AHRS |
| Control | `control/` | Geometric control, bounded allocation, cascaded PID, hover-linearized LQR |
| Simulation | `simulation.py` | Seeded closed-loop orchestration and time-aligned `SimulationResult` |
| Models | `models/urdf_loader.py` | URDF parsing and mesh conversion for visualization |
| Visualization | `visualization/` | Rerun default 3D/telemetry/replay, Matplotlib reports, optional Viser mission controls |
| RL | `rl/`, `training/` | Gymnasium environment, four action spaces, three tasks, PPO training/evaluation |
| Examples | `examples/01–10` | Ten focused and end-to-end demos |
| Tests | `tests/` | Unit, integration, architecture, visualization, and optional RL training coverage |

### 2.2 Strengths

- **Mathematical rigor:** EKF uses analytical Jacobians (not finite-difference), Joseph-form covariance
  updates, tangent-space projection for quaternion normalization, and bias random walks. Rare for a
  hobby project.
- **Swappable controllers:** `QuadcopterController` (PID) and `LQRController` share the same
  `compute(target_pos, target_yaw, dt, prev_target_pos)` interface — drop-in replacement.
- **Production-grade planning:** minimum-snap trajectory generator solves a proper QP via the bordered
  KKT system with C³ continuity constraints (Mellinger & Kumar style).
- **Realistic sensor stack:** Gauss-Markov bias dynamics, scale-factor errors, temperature drift,
  GPS dropout, update-rate modeling.
- **Documentation depth:** 8 markdown files with full LaTeX derivations and implementation status markers.
- **Clean integration point:** `QuadcopterDynamics.update(dt, motor_speeds)` is a one-line interface
  that makes the codebase trivially wrappable for RL, MPC, or SITL.

### 2.3 Current Gaps (honest assessment)

| # | Gap | Impact |
|---|---|---|
| G1 | **Resolved in v0.2:** 13-state quaternion plant | Plant and estimator now use compatible attitude representations |
| G2 | **Resolved in v0.2:** GitHub Actions on Python 3.10–3.12 | Ruff and the full test suite gate pushes and pull requests |
| G3 | **Partially resolved:** wind, gusts, failures, drag, and ground effect exist; obstacles and terrain do not | Robustness experiments are possible, but collision-aware missions are not |
| G4 | **Resolved in v0.2:** aligned `SimulationResult`, CSV/JSON logging, and portable Rerun `.rrd` recordings | Replay and run comparison are now supported |
| G5 | **LQR is hover-only** — constant gain | No gain scheduling for aggressive maneuvers |
| G6 | **Resolved in v0.2:** closed-loop truth and estimator-in-loop regression tests | Tracking and estimation divergence now fail CI |
| G7 | **GPS model is basic** — no satellite geometry (DOP), no RTK, no multipath | Limits realism of GPS-denied scenarios |
| G8 | **Partially resolved:** Gymnasium/PPO baseline exists; domain randomization, curriculum, SAC, and TD3 remain | Learning experiments work; research-grade robustness is still planned |

---

## 3. Delivery Roadmap

This is the canonical delivery sequence and status tracker. Each item appears once under the milestone
that owns it; the technical sections that follow provide design detail without maintaining separate
completion checklists.

### M1 — Foundation Hardening — 🟡 Core complete

#### M1.1 ✅ Quaternion-based Dynamics Plant

The plant uses the 13-state quaternion representation
`[pos(3), vel(3), quat(4), omega_body(3)]`, with normalized quaternion integration and
backward-compatible Euler accessors. This removes the Euler-state singularity and aligns the plant
with the EKF.

**Files:** `src/drones_sim/dynamics/quadcopter.py`, `tests/test_dynamics.py`

#### M1.2 ✅ CI + Closed-Loop Regression Tests

- `.github/workflows/ci.yml` runs Ruff and pytest on Python 3.10, 3.11, and 3.12.
- `tests/test_reworked_architecture.py` gates truth-state tracking below 0.15 m RMSE and the
  estimator-in-loop run below 0.30 m tracking / 0.25 m estimation RMSE.

**Files:** `.github/workflows/ci.yml`, `tests/test_reworked_architecture.py`

#### M1.3 ✅ Disturbance Injection Framework

`ConstantWind`, `StepWind`, `DrydenGust`, `MotorFailure`, `PayloadDrop`, and `GroundEffect` are
integrated with `QuadcopterDynamics`. Wind uses relative airspeed, and the plant includes configurable
linear and quadratic drag.

**Files:** `src/drones_sim/dynamics/disturbances.py`, `tests/test_disturbances.py`

#### M1.4 🟡 Telemetry and Replay

| Output | Status | Use case |
|---|---|---|
| `CsvLogger` | ✅ | Pandas analysis and quick plots |
| `JsonLogger` | ✅ | Structured telemetry |
| Rerun `.rrd` recording | ✅ | Synchronized 3D/telemetry replay and run sharing |
| Binary ULog export | ⬜ | QGroundControl / FlightPlot interoperability |

**Acceptance:** nominal CI and closed-loop bounds are automated. A disturbed-flight RMSE gate and
ULog interoperability remain before M1 is fully complete.

### M2 — RL Baseline — ✅ Implemented

#### M2.1 ✅ Gymnasium and PPO Pipeline

Implemented: four action parameterizations, relative-state observations, hover/waypoint/tracking
tasks, modular reward, PPO training and deterministic evaluation, YAML configs, examples 07–08,
and optional SB3 smoke tests. Section 4 documents the architecture.

**Acceptance:** the recorded LQR-residual checkpoint reports 0.137 m hover RMSE, meeting the original
< 0.15 m target. The trained-policy quality threshold is not yet enforced in CI.

### M3 — Robustness and Scale — ⬜ Planned

#### M3.1 ⬜ Model Predictive Control

Build receding-horizon control from the LQR linearization, using a 20–50 step horizon and explicit
motor, tilt, velocity, and later obstacle constraints. Prototype with `cvxpy`/OSQP.

**Files:** `src/drones_sim/control/mpc.py`, `tests/test_mpc.py`

**Estimate:** ~2 weeks

#### M3.2 ⬜ Monte Carlo Evaluation

Evaluate controllers across seeds, disturbances, and model uncertainty, reporting success rate and
RMSE distributions rather than single-run results.

**Files:** `src/drones_sim/evaluation/monte_carlo.py`

**Estimate:** ~3 days

#### M3.3 ⬜ RL Domain Randomization and Curriculum

Randomize mass, inertia, actuator coefficients, wind, and sensor noise at reset, then progress from
hover through waypoint and trajectory tasks. Add SAC/TD3 entry points after the PPO robustness gate.

**Acceptance:** publish a robustness report versus disturbance magnitude for PID, LQR, MPC, and RL;
the randomized RL policy must beat PID on random-mass hover.

### M4 — Research Extensions — ⬜ Planned

#### M4.1 ⬜ Vision Sensors and VIO

Add pinhole camera simulation, noisy feature tracks, and an MSCKF or factor-graph VIO pipeline.

#### M4.2 ⬜ Obstacle-Aware Planning

Add environment geometry, collision checking, A* or kinodynamic RRT*, and minimum-snap local
replanning.

#### M4.3 ⬜ Gain-Scheduled LQR

Linearize at multiple operating points and interpolate gains for aggressive flight.

#### M4.4 ⬜ Multi-Drone Simulation

Generalize the simulation loop to multiple vehicles and add decentralized consensus or centralized MPC.

#### M4.5 🟡 Aerodynamic Effects

- [x] Ground-effect thrust augmentation
- [x] Configurable linear and quadratic wind-relative drag
- [ ] Blade-element momentum theory
- [ ] Propeller wake interference

#### M4.6 ⬜ SITL Bridge

Add a UDP adapter so PX4 or ArduPilot firmware can drive `QuadcopterDynamics` as its physics backend.

### Continuous Quality-of-Life Backlog

| Item | Status | Description |
|---|---|---|
| Config files | 🟡 Partial | Typed `QuadcopterConfig` / `SimulationConfig` and RL YAML configs exist; a unified experiment schema does not |
| CLI runner | ⬜ Planned | `drones-sim run examples/05 --no-viz --seed 42` |
| Type checking | ⬜ Planned | Add `mypy` to CI |
| Benchmarks | ⬜ Planned | Add EKF and simulation hot-path regression benchmarks |
| Documentation gallery | ⬜ Planned | Auto-render example screenshots into `docs/` |

---

## 4. Reinforcement Learning Pipeline

This section records both the implemented RL baseline and the remaining research design. The stack
uses `QuadcopterDynamics.update(dt, motor_speeds)` as its integration point.

### 4.1 Goals

1. Train neural policies for **hover stabilization**, **waypoint reaching**, and **trajectory tracking**.
2. Compare RL controllers head-to-head with the existing cascaded PID and LQR baselines.
3. Prepare for **sim-to-real** via domain randomization (mass, inertia, wind, sensor noise).
4. Keep the pipeline reproducible, observable (TensorBoard / W&B), and modular.

### 4.2 Library Stack

| Layer | Library | Version pin | Why |
|---|---|---|---|
| Deep learning | **PyTorch** | `torch>=2.2` | Industry standard, mature CUDA/MPS support |
| RL algorithms | **Stable-Baselines3 (SB3)** | `stable-baselines3>=2.3` | High-quality PPO/SAC/TD3 implementations, active maintenance |
| Environment API | **Gymnasium** | `gymnasium>=0.29` | Standard `Env` contract, replaces deprecated `gym` |
| Vectorized envs | **SB3 `SubprocVecEnv` + `VecNormalize`** | (bundled) | Parallel rollouts + observation/reward normalization |
| Logging | **TensorBoard** + optional **Weights & Biases** | `tensorboard>=2.15`, `wandb` | Loss curves, episode returns, custom metrics |
| Video recording | **`gymnasium[box2d]` moviepy** or viser | — | Render evaluation rollouts |
| Hyperparameters | **Optuna** (optional) | `optuna>=3.5` | Bayesian search over PPO/SAC hyperparameters |

**New `pyproject.toml` extras:**

```toml
[project.optional-dependencies]
rl = [
    "torch>=2.2",
    "stable-baselines3>=2.3",
    "gymnasium>=0.29",
    "tensorboard>=2.15",
]
rl-dev = [
    "drones-sim[rl]",
    "wandb",
    "optuna>=3.5",
    "moviepy",
]
```

### 4.3 File Layout

```
src/drones_sim/rl/
├── __init__.py
├── env.py                 # QuadcopterEnv core wrapper
├── actions.py             # Four action parameterizations
├── observations.py        # Relative-state observation
├── reward.py              # Modular sum-of-terms reward
└── tasks.py               # HoverTask, WaypointTask, TrackingTask

training/
├── train_ppo.py           # SB3 PPO entry point
├── eval_policy.py         # Deterministic evaluation and metrics
├── configs/               # YAML configs per experiment
│   ├── ppo_hover.yaml
│   └── ppo_hover_vel.yaml
└── checkpoints/           # Local model and VecNormalize artifacts

examples/
├── 07_rl_hover.py         # Train a hover policy
└── 08_rl_vs_pid.py        # Compare RL vs cascaded PID on circular trajectory

tests/
├── test_rl_env.py         # Env contract, determinism, actions, tasks, reward
└── test_rl_training.py    # Optional SB3 train/save/load/evaluate smoke tests
```

Planned additions are `disturbances.py`, `curriculum.py`, richer wrappers, and SAC/TD3 entry points.

### 4.4 Environment Specification

The core class wraps the existing `QuadcopterDynamics` and exposes a Gymnasium interface.

#### 4.4.1 `QuadcopterEnv` — Core Wrapper

```python
# src/drones_sim/rl/env.py
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.math_utils import quat_to_rotation_matrix
from .tasks import Task
from .actions import ActionParameterization
from .observations import ObservationBuilder
from .reward import RewardFunction


class QuadcopterEnv(gym.Env):
    """Gymnasium wrapper around QuadcopterDynamics.

    Single integration point: QuadcopterDynamics.update(dt, motor_speeds).
    """

    metadata = {"render_modes": ["human", "viser"]}

    def __init__(
        self,
        task: Task,
        action_param: ActionParameterization,
        obs_builder: ObservationBuilder,
        reward_fn: RewardFunction,
        dt: float = 0.01,
        episode_len_s: float = 10.0,
        render_mode: str | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        self.dt = dt
        self.max_steps = int(episode_len_s / dt)
        self.task = task
        self.action_param = action_param
        self.obs_builder = obs_builder
        self.reward_fn = reward_fn
        self.render_mode = render_mode

        self.quad = QuadcopterDynamics(motor_time_constant=0.04)

        # Spaces — shapes derived from builders so they stay in sync
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_builder.dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=action_param.low, high=action_param.high, dtype=np.float32,
        )

        self._step_idx = 0
        self._rng = np.random.default_rng(seed)

    # -- gym.Env contract --------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.quad.reset()
        self.task.reset(self.quad, rng=self._rng)
        self._step_idx = 0
        obs = self.obs_builder.build(self.quad, self.task)
        info = {"t": 0.0}
        return obs.astype(np.float32), info

    def step(self, action):
        motor_speeds = self.action_param.to_motors(action)
        # --- Integrate plant (single call to existing dynamics) ----------
        self.quad.update(self.dt, motor_speeds)
        self._step_idx += 1

        obs = self.obs_builder.build(self.quad, self.task)
        reward = self.reward_fn(self.quad, self.task, action, self._step_idx)
        terminated = self._is_crashed()
        truncated = self._step_idx >= self.max_steps
        info = {"t": self._step_idx * self.dt, "motor_speeds": motor_speeds}
        return obs.astype(np.float32), float(reward), terminated, truncated, info

    def _is_crashed(self) -> bool:
        pos = self.quad.get_position()
        att = self.quad.get_attitude()
        # Crash: hit ground, excessive tilt, or left bounding box
        if pos[2] < 0.0:
            return True
        if abs(att[0]) > np.deg2rad(60) or abs(att[1]) > np.deg2rad(60):
            return True
        if np.linalg.norm(pos) > 50.0:
            return True
        return False

    def render(self):
        if self.render_mode == "viser":
            # Push current pose to a viser server (set up by enjoy.py)
            ...
```

#### 4.4.2 Action Parameterizations

Four interchangeable action spaces are implemented; choose the abstraction level that matches the
learning task.

```python
# src/drones_sim/rl/actions.py
class ActionParameterization:
    low: np.ndarray
    high: np.ndarray
    dim: int
    def to_motors(self, action: np.ndarray) -> np.ndarray: ...


class MotorSpeedAction(ActionParameterization):
    """Lowest level: 4 motor speeds directly (rad/s).
    Hardest to learn, best for acrobatic maneuvers."""
    dim = 4
    low  = np.zeros(4, dtype=np.float32)
    high = np.full(4, 4000.0, dtype=np.float32)
    def to_motors(self, a):
        return np.clip(a, self.low, self.high)


class ThrustBodyRatesAction(ActionParameterization):
    """Mid level: collective thrust + 3 body rates.
    Industry-standard for RL quad papers. RECOMMENDED DEFAULT."""
    dim = 4
    low  = np.array([0.2*9.81, -5, -5, -5], dtype=np.float32)
    high = np.array([2.0*9.81,  5,  5,  5], dtype=np.float32)
    def to_motors(self, a):
        # Solve allocation for [T, tau_phi, tau_theta, tau_psi]
        # using a rate-controller decomposition (body-rate PID)
        # → returns 4 motor speeds
        ...


class VelocityLevelAction(ActionParameterization):
    """World-frame velocity plus yaw rate, stabilized by cascaded inner loops."""
    dim = 4


class LQRResidualAction(ActionParameterization):
    """Normalized residual added to full-state LQR motor commands."""
    dim = 4
```

**Recommendation:** use `LQRResidualAction` for the strongest current hover baseline and
`ThrustBodyRatesAction` when the policy should retain more control authority.

#### 4.4.3 Observation Builder

```python
# src/drones_sim/rl/observations.py
class ObservationBuilder:
    dim: int
    def build(self, quad, task) -> np.ndarray: ...


class RelativeStateObs(ObservationBuilder):
    """Recommended default.
    Relative position to target (3) + velocity (3) + quaternion (4)
    + body rates (3) + previous action (4) = 17-D.
    All quantities in body frame where possible — improves generalization."""
    dim = 17

    def build(self, quad, task):
        pos_err = task.target_pos(quad) - quad.get_position()      # (3,)
        vel     = quad.get_velocity()                              # (3,)
        quat    = _euler_to_quat(quad.get_attitude())              # (4,)  wxyz
        omega   = quad.get_angular_velocity()                      # (3,)
        prev_a  = getattr(self, "_prev_action", np.zeros(4))       # (4,)
        return np.concatenate([pos_err, vel, quat, omega, prev_a])


class TrajectoryPreviewObs(ObservationBuilder):
    """For trajectory tracking: includes K future reference points.
    dim = 17 + 3*K.  Lets the policy anticipate upcoming curvature."""
    def __init__(self, k_future: int = 5):
        self.k_future = k_future
        self.dim = 17 + 3 * k_future
```

All observations are paired with SB3 `VecNormalize` running mean/std normalizer at training time.

#### 4.4.4 Reward Function — Modular Sum-of-Tererms

```python
# src/drones_sim/rl/reward.py
@dataclass
class RewardConfig:
    w_pos:       float = 1.0     # position tracking (L2)
    w_vel:       float = 0.05    # velocity tracking
    w_attitude:  float = 0.1     # tilt penalty
    w_action:    float = 0.01    # control smoothness
    w_action_d:  float = 0.005   # control rate (Δu)
    w_alive:     float = 0.1     # survival bonus per step
    w_reach:     float = 10.0    # one-shot bonus for reaching target
    w_crash:     float = -10.0   # one-shot penalty for crashing
    reach_radius: float = 0.1    # m — target-reached threshold


def reward(quad, task, action, step_idx, cfg: RewardConfig, prev_action=None):
    pos_err = np.linalg.norm(task.target_pos(quad) - quad.get_position())
    vel     = quad.get_velocity()
    att     = quad.get_attitude()
    tilt    = np.sqrt(att[0]**2 + att[1]**2)

    r_pos      = -cfg.w_pos * pos_err
    r_vel      = -cfg.w_vel * np.linalg.norm(vel - task.target_vel(quad))
    r_att      = -cfg.w_attitude * tilt
    r_action   = -cfg.w_action * np.sum(action**2)
    r_action_d = -cfg.w_action_d * np.linalg.norm(action - prev_action) if prev_action is not None else 0.0
    r_alive    = cfg.w_alive

    r = r_pos + r_vel + r_att + r_action + r_action_d + r_alive

    # Terminal bonuses
    if pos_err < cfg.reach_radius:
        r += cfg.w_reach
    return r
```

**Reward design notes:**

- Use **dense** position error (dominant signal) + **sparse** reach bonus (sharper incentive).
- Always include **control effort** penalty — without it the policy learns twitchy, unrealistic behavior.
- Always include a small **alive bonus** — without it the policy learns to crash immediately to end the episode.
- Reward should be **~O(1)** per step so SB3's default learning rates work without tuning.

### 4.5 Tasks

```python
# src/drones_sim/rl/tasks.py
class Task:
    def reset(self, quad, rng): ...
    def target_pos(self, quad) -> np.ndarray: ...
    def target_vel(self, quad) -> np.ndarray: ...


class HoverTask(Task):
    """Hover at a fixed point. Easiest sanity check."""
    def __init__(self, target=(0, 0, 2)): self.target = np.array(target)
    def reset(self, quad, rng): quad.reset()
    def target_pos(self, quad): return self.target
    def target_vel(self, quad): return np.zeros(3)


class WaypointTask(Task):
    """Sequence of waypoints — advance when within reach_radius."""
    ...


class TrackingTask(Task):
    """Track a pre-generated TrajectoryData (circular / min-snap)."""
    def __init__(self, traj: TrajectoryData): self.traj = traj
    def target_pos(self, quad):
        # Look up nearest time index on the trajectory
        ...


class RandomWaypointTask(Task):
    """Sample a new random waypoint after each reach. Promotes exploration."""
    ...
```

### 4.6 Domain Randomization (M3.3 — Planned)

```python
# src/drones_sim/rl/disturbances.py
class DomainRandomizer:
    """Resample physical params at episode reset — closes sim-to-real gap."""
    def randomize_quad(self, quad, rng):
        quad.mass    = rng.uniform(0.8, 1.2)             # ±20% mass
        quad.k_f     = rng.uniform(0.8e-6, 1.2e-6)       # ±20% thrust coef
        quad.arm_len = rng.uniform(0.18, 0.22)
        # Recompute inertia proportionally to mass × arm²
        ...

    def randomize_wind(self, rng) -> WindDisturbance:
        return WindDisturbance(
            direction=rng.normal(size=3),
            magnitude=rng.uniform(0, 3.0),  # m/s
            profile="dryden",
        )

    def randomize_sensors(self, sensor_models, rng):
        for m in sensor_models:
            m.noise_std *= rng.uniform(0.5, 2.0)
```

Apply in `QuadcopterEnv.reset()` *before* `task.reset()`. The policy must succeed across the whole
distribution, not just the nominal — this is what makes RL robust.

### 4.7 Training Scripts

#### 4.7.1 PPO Entry Point (Default Algorithm)

```python
# training/train_ppo.py
import argparse, yaml, numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from drones_sim.rl.env import QuadcopterEnv
from drones_sim.rl.tasks import HoverTask
from drones_sim.rl.actions import ThrustBodyRatesAction
from drones_sim.rl.observations import RelativeStateObs
from drones_sim.rl.reward import reward, RewardConfig
from drones_sim.rl.wrappers import EpisodeStatsCallback


def make_env(seed):
    def _init():
        return QuadcopterEnv(
            task=HoverTask(target=(0, 0, 2)),
            action_param=ThrustBodyRatesAction(),
            obs_builder=RelativeStateObs(),
            reward_fn=lambda q, t, a, i: reward(q, t, a, i, RewardConfig()),
            dt=0.01, episode_len_s=10.0, seed=seed,
        )
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # Vectorized training envs (parallel rollouts)
    venv = SubprocVecEnv([make_env(seed=i) for i in range(args.n_envs)])
    venv = VecNormalize(venv, obs=True, ret=True, clip_obs=10.0)

    model = PPO(
        policy="MlpPolicy", env=venv,
        learning_rate=3e-4, n_steps=4096, batch_size=256,
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        tensorboard_log="./tb/ppo_hover",
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(save_freq=50_000, save_path="./checkpoints/ppo_hover"),
        EpisodeStatsCallback(),
    ]
    model.learn(total_timesteps=args.timesteps, callback=callbacks)
    model.save("./checkpoints/ppo_hover/final.zip")
    venv.save("./checkpoints/ppo_hover/vecnormalize.pkl")


if __name__ == "__main__":
    main()
```

#### 4.7.2 Algorithm Selection

| Task | Recommended algorithm | Why |
|---|---|---|
| Hover / waypoint (default) | **PPO** | Stable, on-policy, easy to tune, well-documented |
| Sample-efficient prototyping | **SAC** | Off-policy, reuses replay buffer, fewer env steps needed |
| Real-time / embedded inference | **TD3** | Deterministic policy → tiny network, fast inference |
| Competitive maneuvering | **PPO + domain randomization** | What Kaufmann et al. used for champion-level racing |

### 4.8 Evaluation & Comparison

```python
# training/eval_policy.py
def evaluate(model_path, n_episodes=20, seed=0):
    env = make_env(seed=seed)()  # single env, deterministic
    model = PPO.load(model_path, env=env)

    results = {"pos_rmse": [], "success": [], "crash": [], "energy": []}
    for ep in range(n_episodes):
        obs, _ = env.reset()
        errors, energies, crashed = [], [], False
        for _ in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            errors.append(np.linalg.norm(env.task.target_pos(env.quad) - env.quad.get_position()))
            energies.append(np.sum(info["motor_speeds"]**2))
            if term:
                crashed = True
                break
        results["pos_rmse"].append(np.sqrt(np.mean(np.square(errors))))
        results["success"].append(not crashed and errors[-1] < 0.1)
        results["crash"].append(crashed)
        results["energy"].append(np.sum(energies))
    return results
```

**Comparison harness** (`examples/08_rl_vs_pid.py`) runs the same trajectory with:
- `QuadcopterController` (cascaded PID)
- `LQRController` (hover-linearized LQR)
- RL-loaded policy

and emits a 4-panel matplotlib comparison: position RMSE, control effort, success rate, max tilt.

### 4.9 Curriculum Strategy

Train progressively harder tasks in sequence:

| Stage | Task | Episodes | Notes |
|---|---|---|---|
| 1 | Hover at (0,0,2), no noise | 500k steps | Sanity check, should converge in <1 hr |
| 2 | Hover + sensor noise | 1M steps | Add realistic IMU/GPS noise |
| 3 | Random waypoints (1–3 m apart) | 2M steps | Multi-goal navigation |
| 4 | Circular trajectory tracking | 3M steps | Continuous reference |
| 5 | + Domain randomization (mass, wind) | 5M+ steps | Sim-to-real prep |

Implement via `curriculum.py` that swaps the `Task` and `DomainRandomizer` based on a moving-average
success rate (SB3 `BaseCallback`).

### 4.10 Training Infrastructure

| Resource | Recommendation |
|---|---|
| **Hardware** | Apple MPS (M-series) works for prototyping; NVIDIA RTX 16GB+ for serious runs (user preference per project memory) |
| **Parallelism** | `SubprocVecEnv` with 8–16 workers (each runs an independent `QuadcopterDynamics`) |
| **Throughput** | The numpy dynamics is the bottleneck, not the network. Target: 5k–10k env steps/sec on CPU |
| **Logging** | TensorBoard by default; W&B for shared experiments |
| **Reproducibility** | Seed everything (`random`, `numpy`, `torch`, env); commit the config YAML alongside each checkpoint |

### 4.11 RL Testing Strategy

```python
# tests/test_rl_env.py
def test_env_contract():
    """Env satisfies the gymnasium contract."""
    env = make_test_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()

    for _ in range(100):
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert np.isfinite(obs).all()
        assert np.isfinite(r)
        assert isinstance(term, bool) and isinstance(trunc, bool)
        if term or trunc:
            break


def test_reward_bounds():
    """Reward stays within O(1) per step (so default LR works)."""
    env = make_test_env()
    env.reset(seed=0)
    rewards = [env.step(env.action_space.sample())[1] for _ in range(200)]
    assert -20 < min(rewards) and max(rewards) < 20


def test_hover_baseline_policy():
    """A trivial 'apply hover thrust' policy should not crash for 2 s.
    Sanity check that the env is learnable."""
    env = make_test_env()
    env.reset(seed=0)
    hover_thrust = env.quad.mass * env.quad.g
    for _ in range(200):  # 2 s at 100 Hz
        action = np.array([hover_thrust, 0, 0, 0])
        _, _, term, _, _ = env.step(action)
        assert not term, "Hover-thrust policy crashed — env is too harsh"
```

### 4.12 RL Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Reward hacking (policy finds degenerate behavior) | Always include control-effort + alive-bonus terms; sanity-check with hover-thrust baseline |
| Training instability | Start with PPO defaults; use `VecNormalize`; gradient clip at 0.5 |
| Sim-to-real gap | Domain randomization (§4.6); complete the M4.5 aerodynamic model before deployment |
| Long wall-clock training | `SubprocVecEnv` parallelism; start on MPS for prototyping, move to GPU for scale |
| Action oscillation | Include Δu penalty term; use `ThrustBodyRatesAction` (not raw motors) |
| Non-determinism across runs | Seed `random`, `numpy`, `torch`, `gym`; SB3 ≥2.3 supports `seed` in `reset()` |

---

## 5. Cross-Cutting Concerns

### 5.1 Testing Strategy Across the Stack

| Layer | Test type | Examples |
|---|---|---|
| Math | Property-based | Quaternion algebra round-trips, rotation-matrix orthonormality |
| Dynamics | Physical consistency | Free-fall accel ≈ −g; hover thrust balance; energy conservation |
| Estimation | Numerical Jacobian vs. analytical | Already specified in `docs/testing.md` §3 |
| Control | Closed-loop regression | Hover + waypoint convergence thresholds |
| Integration | End-to-end pipeline | M1.2 truth-state tracking < 0.15 m RMSE; estimator-in-loop tracking < 0.30 m and estimation < 0.25 m |
| RL | Env contract + reward bounds | §4.11 above |

### 5.2 Performance Budget

For a 100 Hz loop (10 ms / step), current measurements and performance targets:

| Stage | Current | Target | Notes |
|---|---|---|---|
| Dynamics | ~0.3 ms | < 0.5 ms | RK4 on 13-state quaternion plant |
| Sensors | ~0.2 ms | < 0.3 ms | |
| EKF | ~1.5 ms | < 2.0 ms | 16-state matrix ops |
| Controller | ~0.5 ms | < 1.0 ms | |
| MPC (future) | — | < 5.0 ms | QP solve |
| RL inference (future) | — | < 0.5 ms | Policy forward pass on MPS/CPU |

### 5.3 Documentation Discipline

Every new module lands with:
1. A docstring at the module level explaining the math and convention
2. A section in the relevant `docs/*.md` with status marker (✅ / ⚠️ / 🗺️)
3. At least one test exercising the public interface
4. A runnable example if it's a user-facing capability

### 5.4 Backward Compatibility

- Keep the existing `compute(target_pos, target_yaw, dt, prev_target_pos)` interface stable across
  all controllers (PID, LQR, MPC, RL) — this is what makes them swappable in the sim loop.
- The M1.1 quaternion plant keeps `get_position()`, `get_velocity()`, and `get_attitude()`
  (returning Euler for backward compatibility) and adds `get_quaternion()`.
- RL env should accept a `controller_fallback` kwarg so a failed/in-progress policy degrades to PID.

---

## Appendix A — New Dependency Summary

```toml
# pyproject.toml additions
[project.optional-dependencies]
dev = ["pytest>=7.0", "ruff", "mypy"]
mpc = ["cvxpy>=1.5"]
vio = ["gtsam>=4.2"]                # or minisam
rl  = [
    "torch>=2.2",
    "stable-baselines3>=2.3",
    "gymnasium>=0.29",
    "tensorboard>=2.15",
]
rl-dev = ["drones-sim[rl]", "wandb", "optuna>=3.5", "moviepy"]
```

## Appendix B — External References

- Kaufmann et al., "Champion-level drone racing using deep reinforcement learning", *Nature* 2023
- Mellinger & Kumar, "Minimum snap trajectory generation and control for quadrotors", ICRA 2011
- Schulman et al., "Proximal policy optimization algorithms", arXiv 2017
- Haarnoja et al., "Soft actor-critic: off-policy maximum entropy deep RL", ICML 2018
- OpenAI Gymnasium docs: https://gymnasium.farama.org
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io
