# drones_sim

Quadcopter dynamics simulation with cascaded PID control, EKF sensor fusion, disturbance modeling, and reinforcement learning.

[![CI](https://github.com/thanhndv212/drones-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/thanhndv212/drones-sim/actions/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/thanhndv212/drones-sim)

## Install

```bash
cd drones_sim
pip install -e ".[dev]"
```

RL extras (Gymnasium env, SB3 PPO, TensorBoard):

```bash
pip install -e ".[rl]"
```

Full RL dev stack (adds Weights & Biases, Optuna, moviepy):

```bash
pip install -e ".[rl-dev]"
```

## Package structure

```
src/drones_sim/
├── math_utils.py              # Quaternion ops, rotation matrices, Euler helpers
├── trajectory.py              # Trajectory generators (hover-cruise, circular, waypoints, min-snap)
├── models/
│   ├── urdf_loader.py         # Pure-stdlib URDF parser (no external deps)
│   └── quadcopter.urdf        # Bundled quadcopter model
├── sensors/
│   ├── imu.py                 # 9-axis IMU simulator (accel, gyro, mag)
│   ├── gps.py                 # GNSS receiver simulator (position + velocity)
│   └── models.py              # SensorNoiseModel (Gauss-Markov bias), TemperatureModel
├── estimation/
│   ├── ekf.py                 # 16-state EKF + 9-state adaptive EKF + AHRS
│   └── ahrs.py                # Complementary-filter AHRS
├── dynamics/
│   ├── quadcopter.py          # 13-state quaternion Newton-Euler rigid body
│   └── disturbances.py        # Wind, gust, ground effect, motor failure, payload drop
├── control/
│   ├── pid.py                 # Scalar PID with anti-windup
│   ├── cascaded.py            # Position → Velocity → Attitude cascaded PID
│   └── lqr.py                 # Full-state feedback LQR (CARE solution)
├── rl/
│   ├── env.py                 # QuadcopterEnv (gymnasium.Env wrapper)
│   ├── actions.py             # MotorSpeedAction, ThrustBodyRatesAction, VelocityLevelAction, LQRResidualAction
│   ├── observations.py        # RelativeStateObs (17-D observation)
│   ├── tasks.py               # HoverTask, WaypointTask, TrackingTask
│   └── reward.py              # Weighted multi-term reward function
├── training/
│   ├── train_ppo.py           # PPO training entry point (YAML config, CPU default, W&B)
│   ├── eval_policy.py         # Policy evaluation with success/crash metrics
│   ├── configs/
│   │   ├── ppo_hover.yaml     # Config for thrust_rates / lqr_residual actions
│   │   └── ppo_hover_vel.yaml # Config for velocity-level action
│   └── checkpoints/           # Saved models and VecNormalize stats
├── logging/
│   ├── csv_logger.py          # CSV telemetry logger
│   └── json_logger.py         # JSON Lines telemetry logger
└── visualization/
    ├── plots.py               # Matplotlib multi-panel comparison plots
    └── viewer.py              # Viser interactive 3D viewer with frame handles
```

## Features

### State estimation

| Filter | States | Description |
|--------|--------|-------------|
| Extended Kalman Filter | 16-state | Position(3), velocity(3), quaternion(4), accel bias(3), gyro bias(3). Analytical Jacobians, Joseph form covariance, GPS/baro/velocity corrections |
| Adaptive EKF | 9-state | Position(3), velocity(3), accel bias(3). Innovation-window adaptive noise, Gauss-Markov bias |
| AHRS | Complementary filter | Fuses accel, gyro, mag with gyro bias learning |

### Control

| Controller | Type | Description |
|------------|------|-------------|
| Cascaded PID | 3-loop cascade | Position → Velocity → Attitude. 9 PID instances, motor-speed output |
| LQR | Full-state feedback | Linearized around hover, CARE solution, wrench → motor allocation |

### Disturbances (6 types)

| Disturbance | Category | Description |
|-------------|----------|-------------|
| `ConstantWind` | Wind | Steady world-frame drag force |
| `StepWind` | Wind | Wind that switches on at a given time |
| `DrydenGust` | Wind | Continuous turbulence — Gauss-Markov process (MIL-F-8785C) |
| `MotorFailure` | Failure | Degraded rotor thrust coefficient |
| `PayloadDrop` | Failure | Instantaneous mass change |
| `GroundEffect` | Environment | Thrust augmentation near ground (Cheeseman & Bennett) |

### Reinforcement learning

- **QuadcopterEnv** — Gymnasium `Env` compatible with SB3, CleanRL, Tianshou, RLlib
- **Four action parameterizations** — three levels of abstraction plus a residual:

| Action | Policy outputs | Stabilization |
|--------|---------------|---------------|
| `MotorSpeedAction` | Raw motor speeds (4× rad/s) | None (hardest) |
| `ThrustBodyRatesAction` | Thrust delta + body rates (ωx,ωy,ωz) | Rate → torque P-controller |
| `VelocityLevelAction` | World-frame velocity (vx,vy,vz) + yaw rate | Built-in cascaded P-controller (velocity → attitude → torque) |
| `LQRResidualAction` | Delta on LQR motor speeds (in [-1,1]) | Full-state LQR feedback (CARE solution) |

- **Three tasks**: hover, waypoint sequence, trajectory tracking
- **Weighted multi-term reward** (position, velocity, attitude, action smoothness, alive/crash)
- **PPO training** in `training/train_ppo.py` with YAML configs, TensorBoard logging, and optional W&B tracking
- **Policy evaluation** in `training/eval_policy.py` (RMSE, success rate, crash rate)

#### Training

The training script defaults to CPU for small MLP policies (GPU transfer overhead dominates):

```bash
# LQR residual (recommended — 75%+ success rate at 500k steps)
python -m training.train_ppo \
  --config training/configs/ppo_hover.yaml \
  --timesteps 500000 \
  --action-type lqr_residual

# Velocity-level (0% crash, 1.5m RMSE)
python -m training.train_ppo \
  --config training/configs/ppo_hover_vel.yaml \
  --timesteps 200000 \
  --action-type velocity

# Thrust + body rates (legacy)
python -m training.train_ppo \
  --config training/configs/ppo_hover.yaml \
  --timesteps 200000 \
  --action-type thrust_rates
```

Track training with Weights & Biases:

```bash
python -m training.train_ppo \
  --config training/configs/ppo_hover.yaml \
  --action-type lqr_residual \
  --track --wandb-project drones-sim-ppo
```

Open TensorBoard (logs are saved to `./tb/`):

```bash
tensorboard --logdir tb/
```

#### Evaluation

```bash
# Evaluate a trained checkpoint
python -m training.eval_policy \
  --path training/checkpoints/final.zip \
  --episodes 20 \
  --action-type lqr_residual

# Expected output:
#        pos_rmse: 0.1370
#    success_rate: 0.7500
#      crash_rate: 0.0000
#     mean_reward: 6390.6716
```



### Logging

| Logger | Format | Description |
|--------|--------|-------------|
| `CsvLogger` | CSV | Full state + motor speeds + estimate per row |
| `JsonLogger` | JSON Lines | Per-line JSON objects; machine-readable |

## Examples

| # | Script | Description |
|---|--------|-------------|
| 01 | `01_imu_ekf_basic.py` | IMU simulation + 16-state EKF sensor fusion |
| 02 | `02_ekf_adaptive.py` | Adaptive EKF with temperature-dependent sensor models |
| 03 | `03_waypoint_nav.py` | Waypoint navigation with min-snap trajectory + cascaded PID |
| 04 | `04_viser_viewer.py` | Interactive 3D playback with viser |
| 05 | `05_full_pipeline.py` | Full loop: dynamics → sensors → EKF → PID control → viewer |
| 06 | `06_trajectory_following.py` | Trajectory tracking with interactive viser GUI |
| 07 | `07_rl_hover.py` | Train a PPO policy for hover stabilization |
| 08 | `08_rl_vs_pid.py` | RL policy vs cascaded PID comparison on circular trajectory |

```bash
# Run any example
python examples/01_imu_ekf_basic.py
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).

## Dependencies

| Dependency | Purpose |
|------------|---------|
| numpy, scipy | Numerical computation |
| matplotlib | 2D plotting |
| [viser](https://github.com/nerfstudio-project/viser) | Interactive 3D visualization |
| torch, stable-baselines3, gymnasium | RL training (`[rl]` extra) |
| tensorboard, pyyaml | RL logging & config (`[rl]` extra) |
