"""Tests for math_utils quaternion operations."""

import numpy as np
import pytest

from drones_sim.math_utils import (
    quat_normalize,
    quat_multiply,
    quat_conjugate,
    quat_to_rotation_matrix,
    quat_from_euler,
    quat_to_euler,
    quat_derivative,
    euler_to_rotation_matrix,
)


def test_quat_normalize():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    qn = quat_normalize(q)
    assert np.allclose(np.linalg.norm(qn), 1.0)
    assert np.allclose(qn, [1, 0, 0, 0])


def test_quat_multiply_identity():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(quat_multiply(identity, q), q)
    assert np.allclose(quat_multiply(q, identity), q)


def test_quat_conjugate_inverse():
    q = quat_normalize(np.array([1.0, 2.0, 3.0, 4.0]))
    qc = quat_conjugate(q)
    product = quat_multiply(q, qc)
    assert np.allclose(product, [1, 0, 0, 0], atol=1e-10)


def test_quat_to_rotation_identity():
    R = quat_to_rotation_matrix(np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(R, np.eye(3))


def test_quat_euler_roundtrip():
    roll, pitch, yaw = 0.3, -0.2, 0.5
    q = quat_from_euler(roll, pitch, yaw)
    rpy = quat_to_euler(q)
    assert np.allclose(rpy, [roll, pitch, yaw], atol=1e-10)


def test_euler_rotation_matrix_identity():
    R = euler_to_rotation_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3))


def test_quat_derivative_zero_omega():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    qd = quat_derivative(q, np.zeros(3))
    assert np.allclose(qd, 0.0)
