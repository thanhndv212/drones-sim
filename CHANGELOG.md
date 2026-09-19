# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-19

### Added

- Typed vehicle state, trajectory setpoint, control output, simulation configuration, and aligned
  simulation-result contracts.
- Reproducible closed-loop simulator covering dynamics, sensing, estimation, control, and logging.
- Validated quadcopter configuration, bounded weighted control allocation, and nonlinear SE(3)
  geometric control.
- Wind-relative linear and quadratic drag, deterministic Dryden gust state, rotor failures, payload
  drops, and ground effect.
- Per-step IMU measurements, seeded sensor streams, EKF innovation gating, and filter diagnostics.
- Rerun as the default visualization backend, including synchronized 3D, telemetry, event markers,
  gRPC streaming, and portable `.rrd` recordings.
- Matplotlib engineering dashboard and direct `SimulationResult` playback in the optional Viser UI.
- Unified simulation and Rerun examples, architecture documentation, and end-to-end regression tests.

### Changed

- Reworked the plant around a validated 13-state quaternion rigid-body model with bounded actuators.
- Updated cascaded PID, LQR, and RL action paths to use bounded allocation and the shared dynamics API.
- Made Viser optional through the `[viser]` extra; the base installation now uses Rerun.
- Expanded the public package API and refreshed the README and consolidated delivery roadmap.

### Fixed

- Prevented stochastic gust state from advancing multiple times inside one RK4 step.
- Corrected accelerometer linear-acceleration convention and isolated sensor random-number streams.
- Improved EKF numerical stability, covariance handling, and rejection of implausible measurements.

## [0.1.0] - 2026-04-02

### Added

- Initial quadcopter dynamics, PID/LQR control, trajectory generation, sensor simulation, EKF/AHRS
  estimation, Viser visualization, telemetry logging, and Gymnasium/PPO baseline.

[Unreleased]: https://github.com/thanhndv212/drones-sim/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/thanhndv212/drones-sim/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/thanhndv212/drones-sim/releases/tag/v0.1.0
