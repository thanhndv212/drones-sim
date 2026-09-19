# drones_sim

Composable quadcopter simulation for control and state-estimation experiments. It combines a nonlinear quaternion rigid-body model, bounded actuators, SE(3) geometric control, innovation-gated sensor fusion, reproducible sensor models, and result-native 2D/3D visualization.

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

Legacy Viser mission-control examples:

```bash
pip install -e ".[viser]"
```

## Quick start

The high-level runner owns timing, sensing, estimation, control, logging, and aligned outputs:

```python
from drones_sim import ClosedLoopSimulator, SimulationConfig, visualize
from drones_sim.trajectory import generate_circular

reference = generate_circular(duration=10.0, sample_rate=100, radius=2.0)
simulator = ClosedLoopSimulator(
    config=SimulationConfig(dt=0.01, use_estimator=True, seed=7)
)
result = simulator.run(reference)

print(result.summary())
visualize(result)  # Rerun is the default backend
```

For custom experiments, compose `QuadcopterConfig`, `QuadcopterDynamics`,
`GeometricController`, `IMUSimulator`, `GPSSimulator`, and
`ExtendedKalmanFilter`, then inject them into `ClosedLoopSimulator`.

### Working examples

| # | Script | Install | Description |
|---|--------|---------|-------------|
| 01 | `01_imu_ekf_basic.py` | Base | IMU simulation and 16-state EKF sensor fusion |
| 02 | `02_ekf_adaptive.py` | Base | Adaptive EKF with temperature-dependent sensor models |
| 03 | `03_waypoint_nav.py` | Base | Minimum-snap waypoint navigation with cascaded PID |
| 04 | `04_viser_viewer.py` | `[viser]` | Interactive Viser 3D playback |
| 05 | `05_full_pipeline.py` | `[viser]` | Dynamics, sensors, EKF, PID control, and interactive viewer |
| 06 | `06_trajectory_following.py` | `[viser]` | EKF-fused trajectory tracking with mission controls |
| 07 | `07_rl_hover.py` | `[rl]` | Train and evaluate a PPO hover policy |
| 08 | `08_rl_vs_pid.py` | `[rl]` | Compare a trained RL policy with cascaded PID |
| 09 | `09_unified_simulation.py` | Base | Typed end-to-end simulation and diagnostic dashboard |
| 10 | `10_rerun_viewer.py` | Base | Rerun 3D scene, telemetry, streaming, and `.rrd` export |

Run a base example directly:

```bash
python examples/09_unified_simulation.py
```

The default `visualize(result)` call opens Rerun. It can also stream to an
existing gRPC server or save a portable recording:

```bash
python examples/10_rerun_viewer.py
python examples/10_rerun_viewer.py --save flight.rrd
rerun flight.rrd
```

Matplotlib and Viser remain available explicitly:

```python
visualize(result, backend="matplotlib", title="Flight report")
visualize(result, backend="viser", port=8080)  # requires drones-sim[viser]
```

## Package structure

```
src/drones_sim/
├── state.py                   # Typed state, setpoint, and control contracts
├── simulation.py              # Reproducible closed-loop orchestration and results
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
│   ├── geometric.py           # Nonlinear SE(3) trajectory controller
│   ├── allocation.py          # Bounded weighted least-squares motor allocation
│   └── lqr.py                 # Full-state feedback LQR (CARE solution)
├── rl/
│   ├── env.py                 # QuadcopterEnv (gymnasium.Env wrapper)
│   ├── actions.py             # MotorSpeedAction, ThrustBodyRatesAction, VelocityLevelAction, LQRResidualAction
│   ├── observations.py        # RelativeStateObs (17-D observation)
│   ├── tasks.py               # HoverTask, WaypointTask, TrackingTask
│   └── reward.py              # Weighted multi-term reward function
├── logging/
│   ├── csv_logger.py          # CSV telemetry logger
│   └── json_logger.py         # JSON Lines telemetry logger
└── visualization/
    ├── api.py                 # visualize(); Rerun is the default backend
    ├── plots.py               # Matplotlib multi-panel comparison plots
    ├── dashboard.py           # SimulationResult engineering dashboard
    ├── rerun_viewer.py        # Default synchronized 3D, telemetry, events, and replay
    └── viewer.py              # Optional Viser mission-control UI (`[viser]`)

training/
├── train_ppo.py               # PPO training entry point (YAML config, CPU default, W&B)
├── eval_policy.py             # Policy evaluation with success/crash metrics
├── configs/
│   ├── ppo_hover.yaml         # Config for thrust_rates / lqr_residual actions
│   └── ppo_hover_vel.yaml     # Config for velocity-level action
└── checkpoints/               # Saved models and VecNormalize stats
```

## Features

### Project status

| Area | Status |
|------|--------|
| Quaternion dynamics, bounded allocation, geometric/PID/LQR control | ✅ Implemented and tested |
| Sensor simulation, EKF/AHRS estimation, closed-loop orchestration | ✅ Implemented and tested |
| CI, disturbances, CSV/JSON telemetry, Rerun replay | ✅ Implemented and tested |
| Gymnasium environment, PPO training/evaluation, RL examples | ✅ Baseline implemented |
| ULog export and RL domain randomization/curriculum | 🟡 Partially implemented / planned |
| MPC, Monte Carlo evaluation, VIO, obstacle planning, swarms, SITL | ⬜ Planned |

See the [development roadmap](docs/development-roadmap.md) for acceptance criteria and remaining work.

### State estimation

| Filter | States | Description |
|--------|--------|-------------|
| Extended Kalman Filter | 16-state | Position(3), velocity(3), quaternion(4), accel bias(3), gyro bias(3). Analytical Jacobians, Joseph form covariance, GPS/baro/velocity corrections |
| Adaptive EKF | 9-state | Position(3), velocity(3), accel bias(3). Innovation-window adaptive noise, Gauss-Markov bias |
| AHRS | Complementary filter | Fuses accel, gyro, mag with gyro bias learning |

### Control

| Controller | Type | Description |
|------------|------|-------------|
| Geometric | Nonlinear SE(3) | Quaternion-safe trajectory tracking with velocity/acceleration feedforward and bounded allocation |
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

## Tests

```bash
pytest tests/ -v
```

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## License

MIT — see [LICENSE](LICENSE).

## Dependencies

| Dependency | Purpose |
|------------|---------|
| numpy, scipy | Numerical computation |
| matplotlib | 2D plotting |
| [Rerun](https://rerun.io/) | Default synchronized 3D telemetry, replay, and `.rrd` export |
| [viser](https://github.com/nerfstudio-project/viser) | Optional interactive mission controls (`[viser]`) |
| torch, stable-baselines3, gymnasium | RL training (`[rl]` extra) |
| tensorboard, pyyaml | RL logging & config (`[rl]` extra) |
