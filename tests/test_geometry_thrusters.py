"""Check geometry and actual Bullet force moments, independently of PPO."""
import math

import numpy as np
import pybullet as p
import pytest

from teslafliprl import EnvConfig
from teslafliprl.ramp import ramp_normal, ramp_point, ramp_tangent
from teslafliprl.vehicle import THRUSTERS, apply_thrusters, torque_allocation_matrix


def test_reverse_c_enters_right_and_exits_left_upward():
    config = EnvConfig()
    np.testing.assert_allclose(ramp_point(0, config.ramp_radius), [0, 0, 0])
    np.testing.assert_allclose(ramp_tangent(0), [1, 0, 0])
    tangent = ramp_tangent(config.ramp_angle)
    assert tangent[0] < 0 and tangent[2] > 0
    assert ramp_point(config.ramp_angle, config.ramp_radius)[2] > config.ramp_radius
    for theta in np.linspace(0, config.ramp_angle, 20):
        epsilon = 1e-6
        derivative = (ramp_point(theta + epsilon) - ramp_point(theta - epsilon)) / (2 * epsilon)
        np.testing.assert_allclose(derivative / np.linalg.norm(derivative), ramp_tangent(theta), atol=1e-8)
        assert np.dot(ramp_tangent(theta), ramp_normal(theta)) == pytest.approx(0, abs=1e-12)


def test_actuators_cover_all_rotational_axes_and_required_regions():
    allocation = torque_allocation_matrix()
    assert allocation.shape == (3, len(THRUSTERS))
    assert np.linalg.matrix_rank(allocation) == 3
    # Each rotational axis has physically available positive and negative moments.
    assert np.all(np.min(allocation, axis=1) < 0)
    assert np.all(np.max(allocation, axis=1) > 0)
    assert any(t.position[0] > 0 for t in THRUSTERS)
    assert any(t.position[0] < 0 for t in THRUSTERS)
    assert any(t.position[1] > 0 for t in THRUSTERS)
    assert any(t.position[1] < 0 for t in THRUSTERS)
    assert any(t.position[2] < 0 for t in THRUSTERS)
    for nozzle in THRUSTERS:
        assert np.linalg.norm(nozzle.direction) == pytest.approx(1)


@pytest.fixture
def free_box():
    client = p.connect(p.DIRECT)
    p.setGravity(0, 0, 0, physicsClientId=client)
    p.setTimeStep(1 / 240, physicsClientId=client)
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.4, 0.2], physicsClientId=client)
    body = p.createMultiBody(baseMass=10, baseCollisionShapeIndex=shape, basePosition=[0, 0, 5], physicsClientId=client)
    p.changeDynamics(body, -1, linearDamping=0, angularDamping=0, physicsClientId=client)
    try:
        yield client, body
    finally:
        p.disconnect(physicsClientId=client)


def test_off_center_thruster_generates_physical_pitch(free_box):
    client, body = free_box
    nozzle = next(t for t in THRUSTERS if t.name == "front_up")
    apply_thrusters(client, body, [nozzle], [1.0], 100)
    p.stepSimulation(physicsClientId=client)
    velocity, omega = p.getBaseVelocity(body, physicsClientId=client)
    assert velocity[2] > 0
    assert omega[1] < 0  # +X cross +Z = -Y.
    assert abs(omega[0]) < 1e-10 and abs(omega[2]) < 1e-10


def test_bottom_lift_through_mass_center_does_not_invent_torque(free_box):
    client, body = free_box
    nozzle = next(t for t in THRUSTERS if t.name == "bottom_lift")
    apply_thrusters(client, body, [nozzle], [1.0], 100)
    p.stepSimulation(physicsClientId=client)
    velocity, omega = p.getBaseVelocity(body, physicsClientId=client)
    assert velocity[2] > 0
    np.testing.assert_allclose(omega, [0, 0, 0], atol=1e-10)


def test_thruster_rotates_with_vehicle(free_box):
    client, body = free_box
    orientation = p.getQuaternionFromEuler([0, math.pi / 2, 0])
    p.resetBasePositionAndOrientation(body, [0, 0, 5], orientation, physicsClientId=client)
    nozzle = next(t for t in THRUSTERS if t.name == "bottom_lift")
    apply_thrusters(client, body, [nozzle], [1.0], 100)
    p.stepSimulation(physicsClientId=client)
    velocity, _ = p.getBaseVelocity(body, physicsClientId=client)
    assert velocity[0] > 0
    assert abs(velocity[1]) < 1e-10 and abs(velocity[2]) < 1e-10


@pytest.mark.parametrize("override", [
    {"physics_hz": 240, "control_hz": 70},
    {"ramp_angle": math.pi / 3},
    {"ramp_angle": math.pi},
    {"ramp_radius": 0},
    {"ramp_segments": 2},
    {"vehicle_mass": -1},
    {"thruster_force": float("nan")},
    {"max_episode_seconds": 0},
    {"start_x": 0},
])
def test_invalid_physics_configuration_rejected(override):
    with pytest.raises(ValueError):
        EnvConfig(**override)
