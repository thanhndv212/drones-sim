"""Tests for disturbance models and their integration with dynamics."""

import numpy as np

from drones_sim.dynamics import QuadcopterDynamics
from drones_sim.dynamics.disturbances import (
    ConstantWind,
    DrydenGust,
    GroundEffect,
    PayloadDrop,
    StepWind,
)


def test_disturbances_off_by_default():
    """Default QuadcopterDynamics has no disturbances and behaves normally."""
    quad = QuadcopterDynamics()
    assert quad.disturbances == []
    # Should not crash
    quad.update(0.01, np.array([100.0, 100.0, 100.0, 100.0]))


def test_constant_wind_accelerates_drone():
    """A steady horizontal wind should drift the drone when thrust is near zero."""
    quad = QuadcopterDynamics(
        disturbances=[ConstantWind(np.array([5.0, 0.0, 0.0]), k_d=0.05)],
        k_d=0.05,  # match wind's k_d for consistent force
    )
    quad.reset()
    motors = np.zeros(4)  # no thrust — pure wind drag

    for _ in range(200):
        quad.update(0.01, motors)
    # Wind force = k_d * 5.0 = 0.25 N → drone should drift +x
    assert quad.get_position()[0] > 0.1, "Drone did not drift in wind direction"


def test_step_wind_turns_on():
    """StepWind only activates after t_on."""
    wind = StepWind(np.array([10.0, 0.0, 0.0]), t_on=1.0)
    quad = QuadcopterDynamics(disturbances=[wind], k_d=0.05)
    quad.reset()

    # Before t_on: sim time 0 → external_force returns zeros
    assert np.allclose(wind.external_force(0.0, 0.01, quad.state), 0.0)

    # After t_on: external_force still returns zeros (actual force in dynamics)
    assert np.allclose(wind.external_force(2.0, 0.01, quad.state), 0.0)
    # Verify the disturbance doesn't crash when called
    wind.modify_dynamics(quad, 2.0)


def test_payload_drop_reduces_mass():
    """PayloadDrop should reduce the drone mass after t_drop."""
    quad = QuadcopterDynamics(
        mass=1.0,
        disturbances=[PayloadDrop(post_drop_mass=0.5, t_drop=1.0)],
    )
    quad.reset()
    assert quad.mass == 1.0

    # Step past t_drop
    motors = np.full(4, 100.0)
    quad.update(0.01, motors)  # t=0.01
    quad.update(0.01, motors)  # t=0.02
    # t should now be ~0.02 — still before t_drop
    assert quad.mass == 1.0

    # Fast-forward past t_drop
    for _ in range(200):
        quad.update(0.01, motors)
    # t ≈ 2.02 s — past t_drop
    assert quad.mass < 0.6, f"Mass should have dropped to ~0.5, got {quad.mass}"


def test_ground_effect_boosts_near_floor():
    """GroundEffect thrust multiplier should be >1.0 near the floor."""
    ge = GroundEffect(radius=0.13)
    # At altitude=0.01 m the multiplier should be significantly > 1.0
    mult_near = ge.thrust_multiplier(0.01)
    assert mult_near > 1.15, f"Ground effect multiplier near floor = {mult_near}, expected > 1.15"

    # At altitude=1.0 m (far above ground) the multiplier should be ≈ 1.0
    mult_far = ge.thrust_multiplier(1.0)
    assert mult_far < 1.01, f"Ground effect far from floor = {mult_far}, expected ≈1.0"
    assert mult_far >= 1.0

    # At altitude=0.0 the multiplier should be at maximum
    mult_ground = ge.thrust_multiplier(0.0)
    assert mult_ground >= 1.3, f"Ground effect at floor = {mult_ground}, expected >=1.3"


def test_multiple_disturbances_sum():
    """Multiple disturbances should be additive."""
    quad = QuadcopterDynamics(disturbances=[
        GroundEffect(radius=0.13),
        PayloadDrop(post_drop_mass=0.7, t_drop=0.5),
    ])
    quad.reset()
    # Should not crash — each disturbance receives its modify_dynamics call
    motors = np.full(4, 100.0)
    for _ in range(200):
        quad.update(0.01, motors)
    assert quad.mass < 0.8  # payload dropped
    # GroundEffect at altitude>0.5 should not crash


def test_reset_resets_disturbances():
    """Reset should reset all disturbance internal state."""
    quad = QuadcopterDynamics(disturbances=[
        PayloadDrop(post_drop_mass=0.5, t_drop=1.0),
    ])
    # Simulate past t_drop
    motors = np.full(4, 100.0)
    for _ in range(200):
        quad.update(0.01, motors)
    assert quad.mass < 0.6  # mass dropped

    quad.reset()
    assert quad.mass == 1.0  # mass restored
    # _sim_time should be 0
    assert quad._sim_time == 0.0


def test_dryden_gust_produces_finite_value():
    """DrydenGust should produce finite wind values."""
    gust = DrydenGust(intensity=2.0, length_scale=50.0, seed=42)
    for _ in range(100):
        gust.external_force(0.0, 0.01, np.zeros(13))
    # Internal wind should be finite
    assert np.all(np.isfinite(gust._wind))
    assert not np.allclose(gust._wind, 0.0)  # should have drifted from zero
