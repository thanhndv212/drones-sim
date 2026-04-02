# drones_sim

Quadcopter dynamics simulation with cascaded PID control and EKF sensor fusion.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/thanhndv212/drones-sim)


## Install

```bash
cd drones_sim
pip install -e ".[dev]"
```

## Package structure

```
src/drones_sim/
├── math_utils.py          # Quaternion ops, rotation matrices
├── trajectory.py          # Trajectory generators (hover-cruise, circular, waypoints)
├── sensors/
│   ├── imu.py             # 9-axis IMU simulator
│   └── models.py          # Noise, bias, temperature models
├── estimation/
│   ├── ekf.py             # 10-state EKF + 9-state adaptive EKF
│   └── ahrs.py            # Complementary filter AHRS
├── dynamics/
│   └── quadcopter.py      # 12-state Newton-Euler rigid body
├── control/
│   ├── pid.py             # Generic PID with anti-windup
│   └── cascaded.py        # Position→Velocity→Attitude cascade
└── visualization/
    ├── plots.py           # Matplotlib 2D plots
    └── viewer.py          # Viser interactive 3D viewer
```

## Examples

| # | Script | Description |
|---|--------|-------------|
| 01 | `01_imu_ekf_basic.py` | IMU simulation + 10-state EKF fusion |
| 02 | `02_ekf_adaptive.py` | Adaptive EKF with temperature effects |
| 03 | `03_waypoint_nav.py` | Quadcopter waypoint navigation (cascaded PID) |
| 04 | `04_viser_viewer.py` | Interactive 3D playback with viser |
| 05 | `05_full_pipeline.py` | **Full loop**: dynamics → IMU → EKF → PID control |

```bash
# Run any example
python examples/01_imu_ekf_basic.py
```

## Tests

```bash
pytest tests/ -v
```

## Dependencies

- numpy, scipy, matplotlib
- [viser](https://github.com/nerfstudio-project/viser) — interactive 3D visualization
