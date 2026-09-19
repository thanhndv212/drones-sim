"""Rerun-backed synchronized 3D playback and telemetry visualization."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ..simulation import SimulationResult

RerunMode = Literal["spawn", "connect", "save"]


@dataclass(frozen=True)
class RerunConfig:
    """Output and scene configuration for a Rerun recording."""

    application_id: str = "drones_sim"
    timeline: str = "sim_time"
    arm_length: float = 0.2
    trail_radius: float = 0.012
    velocity_scale: float = 0.25

    def __post_init__(self) -> None:
        if not self.application_id:
            raise ValueError("application_id cannot be empty")
        if not self.timeline:
            raise ValueError("timeline cannot be empty")
        if self.arm_length <= 0.0:
            raise ValueError("arm_length must be positive")
        if self.trail_radius <= 0.0:
            raise ValueError("trail_radius must be positive")
        if self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be positive")


def _load_rerun():
    try:
        rr = importlib.import_module("rerun")
        rrb = importlib.import_module("rerun.blueprint")
    except ImportError as exc:
        raise ImportError(
            "The default visualization backend requires rerun-sdk. "
            "Reinstall drones-sim or run: pip install rerun-sdk"
        ) from exc
    return rr, rrb


def _validate_result(result: SimulationResult) -> int:
    count = len(result.time)
    if count == 0:
        raise ValueError("SimulationResult cannot be empty")
    expected = {
        "position": (count, 3),
        "velocity": (count, 3),
        "quaternion": (count, 4),
        "body_rates": (count, 3),
        "estimated_position": (count, 3),
        "estimated_velocity": (count, 3),
        "estimated_quaternion": (count, 4),
        "reference_position": (count, 3),
        "reference_velocity": (count, 3),
        "motor_speeds": (count, 4),
        "commanded_wrench": (count, 4),
        "allocation_saturated": (count,),
    }
    for name, shape in expected.items():
        value = np.asarray(getattr(result, name))
        if value.shape != shape:
            raise ValueError(
                f"SimulationResult.{name} must have shape {shape}, got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"SimulationResult.{name} contains non-finite values")
    if not np.all(np.isfinite(result.time)):
        raise ValueError("SimulationResult.time contains non-finite values")
    if np.any(np.diff(result.time) < 0.0):
        raise ValueError("SimulationResult.time must be monotonic")
    return count


def rerun_blueprint():
    """Return the default flight-analysis layout."""
    _, rrb = _load_rerun()
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                origin="/world",
                contents="/world/**",
                name="Flight",
                background=[18, 22, 30],
                line_grid=True,
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(
                    origin="/telemetry/errors", name="Position errors [m]"
                ),
                rrb.TimeSeriesView(
                    origin="/telemetry/position", name="Position [m]"
                ),
                rrb.TimeSeriesView(
                    origin="/telemetry/motors", name="Rotor speeds [rad/s]"
                ),
                row_shares=[1.0, 1.2, 1.0],
            ),
            column_shares=[1.7, 1.0],
        ),
        collapse_panels=True,
    )


def _log_series_styles(recording, rr) -> None:
    series = {
        "/telemetry/errors/position": (
            ["tracking", "estimation"],
            [[230, 85, 70], [245, 155, 55]],
        ),
        "/telemetry/position/truth": (
            ["truth x", "truth y", "truth z"],
            [[75, 145, 255], [60, 190, 225], [80, 205, 145]],
        ),
        "/telemetry/position/reference": (
            ["reference x", "reference y", "reference z"],
            [[150, 220, 135], [110, 200, 105], [75, 180, 80]],
        ),
        "/telemetry/position/estimate": (
            ["estimate x", "estimate y", "estimate z"],
            [[255, 185, 70], [245, 145, 55], [225, 105, 45]],
        ),
        "/telemetry/motors/speed": (
            ["front", "right", "rear", "left"],
            [[230, 75, 75], [80, 190, 100], [75, 125, 240], [210, 90, 220]],
        ),
        "/telemetry/control/wrench": (
            ["thrust", "torque x", "torque y", "torque z"],
            [[235, 205, 80], [225, 80, 80], [80, 190, 100], [75, 125, 240]],
        ),
        "/telemetry/state/body_rates": (
            ["p", "q", "r"],
            [[225, 80, 80], [80, 190, 100], [75, 125, 240]],
        ),
        "/telemetry/control/allocation_saturated": (
            ["saturated"],
            [[230, 70, 60]],
        ),
    }
    for path, (names, colors) in series.items():
        recording.log(
            path,
            rr.SeriesLines(names=names, colors=colors),
            static=True,
        )


def _log_static_scene(
    result: SimulationResult,
    recording,
    rr,
    config: RerunConfig,
) -> None:
    recording.log("/world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    recording.log(
        "/world/paths/reference",
        rr.LineStrips3D(
            [result.reference_position],
            colors=[[80, 210, 110]],
            radii=config.trail_radius,
            labels=["reference"],
        ),
        static=True,
    )
    recording.log(
        "/world/paths/truth",
        rr.LineStrips3D(
            [result.position],
            colors=[[55, 135, 255]],
            radii=config.trail_radius * 1.2,
            labels=["truth"],
        ),
        static=True,
    )
    recording.log(
        "/world/paths/estimate",
        rr.LineStrips3D(
            [result.estimated_position],
            colors=[[255, 145, 45]],
            radii=config.trail_radius,
            labels=["estimate"],
        ),
        static=True,
    )

    all_positions = np.vstack([result.position, result.reference_position])
    horizontal_extent = max(
        2.0,
        float(np.max(np.ptp(all_positions[:, :2], axis=0))) * 0.75,
    )
    ground_center = np.mean(all_positions[:, :2], axis=0)
    recording.log(
        "/world/ground",
        rr.Boxes3D(
            centers=[[ground_center[0], ground_center[1], -0.015]],
            half_sizes=[[horizontal_extent, horizontal_extent, 0.01]],
            colors=[[50, 58, 68, 120]],
        ),
        static=True,
    )

    arm = config.arm_length
    recording.log(
        "/world/drone/body/geometry/arms",
        rr.LineStrips3D(
            [
                [[-arm, 0.0, 0.0], [arm, 0.0, 0.0]],
                [[0.0, -arm, 0.0], [0.0, arm, 0.0]],
            ],
            colors=[[220, 225, 235], [220, 225, 235]],
            radii=0.018,
        ),
        static=True,
    )
    recording.log(
        "/world/drone/body/geometry/rotors",
        rr.Points3D(
            [[arm, 0.0, 0.0], [0.0, arm, 0.0], [-arm, 0.0, 0.0], [0.0, -arm, 0.0]],
            colors=[[230, 75, 75], [80, 190, 100], [75, 125, 240], [210, 90, 220]],
            radii=0.065,
            labels=["front", "right", "rear", "left"],
        ),
        static=True,
    )
    recording.log(
        "/world/drone/body/geometry/axes",
        rr.Arrows3D(
            origins=np.zeros((3, 3)),
            vectors=np.eye(3) * arm * 1.4,
            colors=[[235, 70, 70], [70, 215, 95], [70, 125, 245]],
            radii=0.01,
            labels=["body x", "body y", "body z"],
        ),
        static=True,
    )
    _log_series_styles(recording, rr)

    summary = result.summary()
    summary_lines = ["# Drone simulation", ""]
    summary_lines.extend(f"- **{key}**: {value:.5g}" for key, value in summary.items())
    if result.metadata:
        summary_lines.extend(
            ["", "## Metadata", "", "```json", json.dumps(result.metadata, indent=2, default=str), "```"]
        )
    recording.log(
        "/report/summary",
        rr.TextDocument("\n".join(summary_lines), media_type="text/markdown"),
        static=True,
    )


def log_simulation_rerun(
    result: SimulationResult,
    recording,
    *,
    config: RerunConfig | None = None,
) -> None:
    """Log a complete ``SimulationResult`` to an existing recording stream."""
    rr, _ = _load_rerun()
    cfg = config or RerunConfig()
    count = _validate_result(result)
    _log_static_scene(result, recording, rr, cfg)

    previous_saturation = False
    for index in range(count):
        recording.set_time(cfg.timeline, duration=float(result.time[index]))
        quaternion_xyzw = np.roll(result.quaternion[index], -1)
        recording.log(
            "/world/drone/body",
            rr.Transform3D(
                translation=result.position[index],
                rotation=rr.Quaternion(xyzw=quaternion_xyzw),
            ),
        )
        recording.log(
            "/world/current/reference",
            rr.Points3D(
                [result.reference_position[index]],
                colors=[[80, 220, 110]],
                radii=0.055,
                labels=["reference"],
            ),
        )
        recording.log(
            "/world/current/estimate",
            rr.Points3D(
                [result.estimated_position[index]],
                colors=[[255, 145, 45]],
                radii=0.035,
                labels=["estimate"],
            ),
        )
        recording.log(
            "/world/vectors/velocity",
            rr.Arrows3D(
                origins=[result.position[index]],
                vectors=[result.velocity[index] * cfg.velocity_scale],
                colors=[[85, 190, 245]],
                radii=0.012,
                labels=["velocity"],
            ),
        )
        recording.log(
            "/world/vectors/tracking_error",
            rr.Arrows3D(
                origins=[result.position[index]],
                vectors=[result.reference_position[index] - result.position[index]],
                colors=[[235, 75, 65]],
                radii=0.01,
                labels=["tracking error"],
            ),
        )

        tracking_error = float(np.linalg.norm(result.tracking_error[index]))
        estimation_error = float(np.linalg.norm(result.estimation_error[index]))
        recording.log(
            "/telemetry/errors/position",
            rr.Scalars([tracking_error, estimation_error]),
        )
        recording.log(
            "/telemetry/position/truth", rr.Scalars(result.position[index])
        )
        recording.log(
            "/telemetry/position/reference",
            rr.Scalars(result.reference_position[index]),
        )
        recording.log(
            "/telemetry/position/estimate",
            rr.Scalars(result.estimated_position[index]),
        )
        recording.log(
            "/telemetry/motors/speed", rr.Scalars(result.motor_speeds[index])
        )
        recording.log(
            "/telemetry/control/wrench",
            rr.Scalars(result.commanded_wrench[index]),
        )
        recording.log(
            "/telemetry/state/body_rates", rr.Scalars(result.body_rates[index])
        )
        saturated = bool(result.allocation_saturated[index])
        recording.log(
            "/telemetry/control/allocation_saturated",
            rr.Scalars([float(saturated)]),
        )
        if saturated and not previous_saturation:
            recording.log(
                "/events/control",
                rr.TextLog(
                    f"Control allocation saturated at t={result.time[index]:.3f}s",
                    level="WARN",
                ),
            )
        previous_saturation = saturated


def visualize_rerun(
    result: SimulationResult,
    *,
    mode: RerunMode = "spawn",
    path: str | Path | None = None,
    url: str | None = None,
    config: RerunConfig | None = None,
):
    """Visualize, stream, or save a synchronized Rerun recording.

    Parameters
    ----------
    mode:
        ``"spawn"`` opens a local viewer, ``"connect"`` streams to a running
        Rerun gRPC server, and ``"save"`` writes an ``.rrd`` recording.
    path:
        Required in save mode. ``.rrd`` is appended when no suffix is present.
    url:
        Optional gRPC endpoint used in connect mode.
    """
    rr, _ = _load_rerun()
    cfg = config or RerunConfig()
    if mode not in {"spawn", "connect", "save"}:
        raise ValueError(f"unsupported Rerun mode: {mode!r}")
    blueprint = rerun_blueprint()
    recording = rr.RecordingStream(cfg.application_id)
    output_path: Path | None = None
    if mode == "spawn":
        recording.spawn(default_blueprint=blueprint)
    elif mode == "connect":
        recording.connect_grpc(url, default_blueprint=blueprint)
    else:
        if path is None:
            raise ValueError("path is required when mode='save'")
        output_path = Path(path).expanduser()
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".rrd")
        if output_path.suffix.lower() != ".rrd":
            raise ValueError("Rerun recording path must use the .rrd suffix")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        recording.save(output_path, default_blueprint=blueprint)
    log_simulation_rerun(result, recording, config=cfg)
    recording.flush()
    return output_path if output_path is not None else recording
