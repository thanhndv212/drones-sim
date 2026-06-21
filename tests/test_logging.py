"""Tests for telemetry loggers."""

import csv
import json
import os
import tempfile

import numpy as np

from drones_sim.logging import CsvLogger, JsonLogger


def _example_state() -> np.ndarray:
    s = np.zeros(13)
    s[:3] = [1.0, 2.0, 3.0]
    s[3:6] = [0.1, 0.2, -0.3]
    s[6] = 1.0  # identity quat
    s[10:13] = [0.01, 0.02, 0.03]
    return s


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

def test_csv_logger_writes_header_and_rows():
    state = _example_state()
    motors = np.array([100.0, 200.0, 300.0, 400.0])
    est = np.array([1.01, 2.02, 2.98])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name

    try:
        logger = CsvLogger(path)
        logger.log(0.0, state, estimate=est, motors=motors)
        logger.log(0.1, state, estimate=est, motors=motors)
        logger.log(0.2, state)
        logger.close()

        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        assert len(rows) == 4  # header + 3 data rows
        assert rows[0] == CsvLogger.HEADER

        # First data row — check t and position
        assert abs(float(rows[1][0]) - 0.0) < 1e-9
        assert abs(float(rows[1][1]) - 1.0) < 1e-9
        assert abs(float(rows[1][2]) - 2.0) < 1e-9
        assert abs(float(rows[1][3]) - 3.0) < 1e-9

        # Motor columns (0-indexed: columns 14-17)
        assert abs(float(rows[1][14]) - 100.0) < 1e-9
        assert abs(float(rows[1][15]) - 200.0) < 1e-9
        assert abs(float(rows[1][16]) - 300.0) < 1e-9
        assert abs(float(rows[1][17]) - 400.0) < 1e-9

        # Estimate columns (18-20)
        assert abs(float(rows[1][18]) - 1.01) < 1e-9
        assert abs(float(rows[1][19]) - 2.02) < 1e-9
        assert abs(float(rows[1][20]) - 2.98) < 1e-9

        # Third row — no motors, no estimate → zeros and NaN
        assert float(rows[3][14]) == 0.0
        assert np.isnan(float(rows[3][18]))
    finally:

        os.unlink(path)


def test_csv_logger_context_manager():
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        state = _example_state()
        with CsvLogger(path) as logger:
            logger.log(0.0, state)
        # File should be closed and have content
        with open(path) as f:
            content = f.read()
        assert len(content) > 0
    finally:

        os.unlink(path)


# ---------------------------------------------------------------------------
# JSON logger
# ---------------------------------------------------------------------------

def test_json_logger_writes_valid_jsonl():
    state = _example_state()
    motors = np.array([100.0, 200.0, 300.0, 400.0])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        logger = JsonLogger(path)
        logger.log(0.0, state, motors=motors)
        logger.log(0.1, state, estimate=np.array([1.0, 2.0, 3.0]))
        logger.close()

        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 2
        assert lines[0]["t"] == 0.0
        assert lines[0]["position"] == [1.0, 2.0, 3.0]
        assert lines[0]["quaternion"] == [1.0, 0.0, 0.0, 0.0]
        assert lines[0]["motors"] == [100.0, 200.0, 300.0, 400.0]
        assert lines[0]["estimate"] is None

        # Second row — estimate present
        assert lines[1]["estimate"] == [1.0, 2.0, 3.0]
        assert lines[1]["motors"] == []
    finally:

        os.unlink(path)


def test_logger_handles_missing_estimate():
    state = _example_state()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        logger = JsonLogger(path)
        logger.log(0.0, state)  # no estimate, no motors
        logger.close()
        with open(path) as f:
            rec = json.loads(f.readline())
        assert rec["estimate"] is None
        assert rec["motors"] == []
    finally:

        os.unlink(path)
