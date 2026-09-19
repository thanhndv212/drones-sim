"""Contract and recording tests for the default Rerun backend."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from drones_sim.simulation import SimulationResult
from drones_sim.visualization import api as visualization_api
from drones_sim.visualization.rerun_viewer import (
    RerunConfig,
    log_simulation_rerun,
    rerun_blueprint,
    visualize_rerun,
)


def _result() -> SimulationResult:
    count = 3
    position = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.2, 0.1, 1.05]]
    )
    quaternion = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (count, 1))
    return SimulationResult(
        time=np.array([0.0, 0.01, 0.02]),
        position=position,
        velocity=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.5]]),
        quaternion=quaternion,
        body_rates=np.zeros((count, 3)),
        estimated_position=position + 0.01,
        estimated_velocity=np.zeros((count, 3)),
        estimated_quaternion=quaternion.copy(),
        reference_position=position + np.array([0.02, 0.0, 0.0]),
        reference_velocity=np.zeros((count, 3)),
        motor_speeds=np.full((count, 4), 1566.0),
        commanded_wrench=np.tile(np.array([9.81, 0.0, 0.0, 0.0]), (count, 1)),
        allocation_saturated=np.array([False, True, False]),
        metadata={"controller": "test"},
    )


class _RecordingSpy:
    def __init__(self) -> None:
        self.logs: list[tuple[str, bool]] = []
        self.times: list[tuple[str, float]] = []

    def log(self, path, entity, *extra, static=False, **kwargs) -> None:
        self.logs.append((str(path), bool(static)))

    def set_time(self, timeline, *, duration) -> None:
        self.times.append((timeline, float(duration)))


def test_logging_contract_contains_scene_telemetry_and_event():
    recording = _RecordingSpy()
    result = _result()
    log_simulation_rerun(result, recording, config=RerunConfig(timeline="flight"))

    paths = [path for path, _ in recording.logs]
    assert recording.times == [("flight", 0.0), ("flight", 0.01), ("flight", 0.02)]
    assert "/world/paths/reference" in paths
    assert "/world/drone/body" in paths
    assert "/world/vectors/tracking_error" in paths
    assert "/telemetry/errors/position" in paths
    assert "/telemetry/motors/speed" in paths
    assert "/telemetry/control/wrench" in paths
    assert paths.count("/events/control") == 1


def test_blueprint_builds_with_installed_sdk():
    assert rerun_blueprint() is not None


def test_save_produces_nonempty_rrd(tmp_path):
    output = visualize_rerun(
        _result(), mode="save", path=tmp_path / "flight_recording"
    )
    assert output == tmp_path / "flight_recording.rrd"
    assert output.is_file()
    assert output.stat().st_size > 1_000


def test_save_requires_rrd_path():
    with pytest.raises(ValueError, match="path is required"):
        visualize_rerun(_result(), mode="save")
    with pytest.raises(ValueError, match=".rrd suffix"):
        visualize_rerun(_result(), mode="save", path="recording.txt")


def test_invalid_result_is_rejected_before_logging():
    malformed = replace(_result(), motor_speeds=np.zeros((2, 4)))
    with pytest.raises(ValueError, match="motor_speeds"):
        log_simulation_rerun(malformed, _RecordingSpy())


def test_visualize_uses_rerun_by_default(monkeypatch):
    calls = []

    def fake_rerun(result, **kwargs):
        calls.append((result, kwargs))
        return "rerun-recording"

    monkeypatch.setattr(visualization_api, "visualize_rerun", fake_rerun)
    result = _result()
    returned = visualization_api.visualize(result, mode="save", path="flight.rrd")
    assert returned == "rerun-recording"
    assert calls == [(result, {"mode": "save", "path": "flight.rrd"})]


def test_visualize_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown visualization backend"):
        visualization_api.visualize(_result(), backend="unknown")
