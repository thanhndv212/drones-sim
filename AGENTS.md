# AGENTS.md — `drones_sim`

This file is the operating guide for AI coding agents working in this repository. It applies to the
entire repository unless a more specific `AGENTS.md` exists in a subdirectory.

## Project Mission

`drones_sim` is a composable, pure-Python quadcopter simulator for modeling, control, estimation,
reinforcement learning, and flight-data visualization. Changes should preserve physical clarity,
determinism, typed interfaces, and compatibility with lightweight local experimentation.

The default stack is:

```text
TrajectoryData -> TrajectorySetpoint -> Controller -> ControlOutput
                                              |              |
                                              |              v
                                     VehicleState <- Dynamics
                                              |
                                        IMU and GPS
                                              |
                                              v
                                       Navigation EKF
                                              |
                                              v
                                       SimulationResult
                                       |- Rerun playback
                                       |- Matplotlib report
                                       `- CSV/JSON logging
```

## Session Start Protocol

At the start of every session, load project memory before investigating or changing code:

```bash
python3 ~/.config/opencode/skills/brain/scripts/brain.py context /Users/thanhndv212/Develop/drones_sim
```

This surfaces relevant facts, learnings, and open tasks from previous sessions. Use it to avoid
repeating failed approaches or undoing established design decisions.

If `.codegraph/` exists, use CodeGraph before `rg`, `find`, or broad file reads when locating or
understanding code:

```bash
codegraph explore "question or symbol names"
codegraph node path/to/file.py
```

Use `rg` or `rg --files` for subsequent targeted searches. Do not regenerate the CodeGraph index
unless the user explicitly requests it.

## Repository Map

```text
src/drones_sim/
|- state.py                 Typed state, setpoint, and control contracts
|- simulation.py            Seeded closed-loop orchestration and SimulationResult
|- dynamics/                Physical configuration, plant, and disturbances
|- control/                 PID, LQR, geometric control, and allocation
|- sensors/                 IMU, GPS, bias, and temperature models
|- estimation/              EKF, adaptive EKF, and AHRS
|- trajectory.py            Reference and minimum-snap trajectory generation
|- visualization/           Rerun default, Matplotlib, optional Viser
|- logging/                 CSV and JSON Lines telemetry
|- rl/                      Optional Gymnasium environment and policy interfaces
`- models/                  URDF loader and packaged quadcopter model

examples/                   Runnable numbered examples
training/                   Optional Stable-Baselines3 training and evaluation
tests/                      Unit, integration, visualization, and optional RL tests
docs/                       Architecture, design notes, and delivery roadmap
```

## Physical and Mathematical Conventions

Treat these as API contracts:

- World frame: ENU, positive z upward.
- Body frame: front-right-up.
- Quaternion ordering: Hamilton `[w, x, y, z]`.
- Quaternions rotate body-frame vectors into the world frame.
- Angular velocity and torque are expressed in the body frame.
- Rotor thrust acts along positive body z.
- Accelerometer output is specific force: `R.T @ (a_world + gravity_up)`.
- The plant state is `[position(3), velocity(3), quaternion(4), body_rates(3)]`.
- SI units are mandatory unless an interface explicitly documents otherwise.

Normalize quaternions after integration or correction. Validate array shape and finiteness at public
boundaries. Never silently mix Euler angles, quaternion orderings, or coordinate frames.

## Architecture Rules

- Prefer `VehicleState`, `TrajectorySetpoint`, `ControlOutput`, `QuadcopterConfig`,
  `SimulationConfig`, and `SimulationResult` at module boundaries.
- Keep legacy array accessors working unless a documented breaking release removes them.
- Route actuator commands through `ControlAllocator`; do not duplicate unconstrained allocation
  inversions in controllers or RL actions.
- Advance stochastic disturbances once per outer integration step, not once per RK4 stage.
- Give each component a local, seedable random generator. Avoid global `numpy.random` state in new
  code.
- Reset must restore clocks, actuator state, estimator diagnostics, disturbance state, and random
  streams required for deterministic replay.
- Keep reference, truth, estimate, command, and event arrays aligned on one time axis.
- Prefer dependency injection in `ClosedLoopSimulator` over hidden module-level configuration.
- Preserve the high-level `QuadcopterDynamics.update(dt, motor_speeds)` integration point.

## Control and Estimation Rules

- Controllers must respect configured motor bounds and report allocation saturation where possible.
- Geometric control should work directly with rotation matrices or quaternions; do not introduce an
  Euler-angle singularity into the main control path.
- EKF measurement updates must use stable linear solves, Joseph-form covariance updates, covariance
  symmetrization, and quaternion normalization.
- Add innovation gating and diagnostics for new absolute measurements.
- Accelerometer gravity correction is a quasi-static pseudo-measurement, not a universal correction
  during aggressive flight.
- Every estimator or controller change needs a numerical or closed-loop regression test.

## Visualization Rules

- Rerun is the default backend and belongs in the base installation.
- Matplotlib is the noninteractive reporting backend.
- Viser is optional and must remain behind the `[viser]` extra and lazy imports.
- New visualization paths should consume `SimulationResult`; do not create parallel telemetry
  formats without a compelling reason.
- Keep `.rrd` recordings portable and deterministic from a completed result.
- Tests must not launch GUI processes or require a display server.

## Reinforcement-Learning Rules

- RL dependencies remain optional under `[rl]` and `[rl-dev]`.
- Importing the base `drones_sim` package must not require Gymnasium, PyTorch, SB3, W&B, or Viser.
- Action parameterizations must use the same plant limits and allocation path as classical control.
- Seed environment resets and preserve the Gymnasium `reset`/`step` contract.
- Store generated checkpoints, TensorBoard logs, W&B runs, and model artifacts only in ignored paths.
- A training smoke test proves integration, not policy quality. Keep benchmark acceptance criteria
  explicit and separate.

## Development Setup

Base development installation:

```bash
python -m pip install -e ".[dev]"
```

Optional stacks:

```bash
python -m pip install -e ".[viser]"
python -m pip install -e ".[rl]"
python -m pip install -e ".[rl-dev]"
```

Use supported Python versions from CI: 3.10, 3.11, and 3.12.

## Validation Commands

Run the narrowest relevant tests while iterating, then the complete release gate before handoff:

```bash
ruff check .
pytest -q
python -m compileall -q src examples training
git diff --check
```

For visualization changes:

```bash
pytest -q tests/test_rerun_visualization.py tests/test_reworked_architecture.py
MPLBACKEND=Agg python examples/09_unified_simulation.py
python examples/10_rerun_viewer.py --save /tmp/drones-sim-flight.rrd
```

For release validation:

```bash
python -m build
python -m twine check dist/*
```

Optional RL tests skip when the RL extra is absent. Do not report a skipped policy-quality test as a
validated trained policy.

## Testing Expectations

- Add tests with every bug fix and public behavior change.
- Prefer deterministic seeds and invariant assertions over snapshots of incidental floating-point
  output.
- Dynamics tests should verify physical consistency, bounds, reset behavior, and quaternion norms.
- Estimation tests should cover covariance properties, outlier rejection, and reacquisition.
- Integration tests should bound tracking and estimation error, not merely assert finite values.
- Visualization tests should validate logged entity paths, timelines, metadata, and save behavior
  through fakes or recording spies.
- Keep GUI, long training, and network activity out of the default suite.

## Documentation Requirements

Update documentation with user-visible changes:

- `README.md` for installation, public APIs, examples, and feature status.
- `CHANGELOG.md` for release-facing additions, changes, fixes, and removals.
- `docs/architecture.md` for conventions and cross-module design.
- `docs/development-roadmap.md` for canonical milestone status and future work.

The delivery roadmap is the single status tracker. Technical sections may explain planned designs but
must not duplicate completion checklists.

## Git and Workspace Hygiene

- Preserve unrelated user changes and untracked files.
- Use `apply_patch` for deliberate file edits.
- Do not commit `.codegraph/`, build output, model checkpoints, W&B data, TensorBoard logs, caches,
  virtual environments, or local IDE state.
- Keep commits focused and independently understandable. Use conventional subjects such as
  `feat(core):`, `feat(visualization):`, `fix(estimation):`, `docs:`, `test:`, or `chore(release):`.
- Never rewrite shared history, force-push, delete tags, or publish a package without explicit user
  authorization.
- Before pushing, inspect `git status`, staged diff statistics, and `git diff --cached --check`.

## Release Procedure

For a release explicitly requested by the user:

1. Ensure `pyproject.toml` and `CHANGELOG.md` contain the same version and release date.
2. Run Ruff, pytest, example smoke tests, `python -m build`, and `twine check`.
3. Verify CI succeeds on the release commit for all supported Python versions.
4. Create an annotated tag named `vX.Y.Z` at that exact commit and push it.
5. Publish a GitHub release from the matching changelog section and attach the wheel and sdist.
6. Upload those exact artifacts to PyPI only when explicitly authorized and credentials are available.
7. Verify the GitHub release, package index page, and installation metadata after publication.

Never rebuild between verification and upload: publish the exact artifacts that passed the release
gate.

## Definition of Done

A change is complete only when:

- The implementation follows the frame, unit, state, and dependency conventions above.
- Relevant focused tests and the full default suite pass.
- Ruff and `git diff --check` pass.
- Public documentation and changelog entries are current.
- Generated artifacts remain ignored.
- The handoff states what changed, what was validated, and any optional checks that were skipped.
