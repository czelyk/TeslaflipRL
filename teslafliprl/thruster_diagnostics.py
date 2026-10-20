"""Measure the complete car's response to individual nozzles, without contacts."""
import math

import numpy as np
import pybullet as p

from .config import EnvConfig
from .vehicle import THRUSTERS, apply_thrusters, build_vehicle


def pulse_response(index, level=1., duration=.05, orientation=(0., 0., 0.)):
    config = EnvConfig(randomize=False)
    client = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, 0, physicsClientId=client)
        p.setTimeStep(1 / config.physics_hz, physicsClientId=client)
        car, wheels = build_vehicle(client, config, [0, 0, 5])
        quat = p.getQuaternionFromEuler(orientation)
        p.resetBasePositionAndOrientation(car, [0, 0, 5], quat, physicsClientId=client)
        p.setJointMotorControlArray(car, wheels, p.VELOCITY_CONTROL, forces=[0] * 4, physicsClientId=client)
        levels = np.zeros(len(THRUSTERS))
        levels[index] = level
        steps = round(duration * config.physics_hz)
        for _ in range(steps):
            apply_thrusters(client, car, THRUSTERS, levels, config.thruster_force)
            p.stepSimulation(physicsClientId=client)
        velocity, omega = p.getBaseVelocity(car, physicsClientId=client)
        rotation = np.asarray(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        direction = rotation @ np.asarray(THRUSTERS[index].direction)
        torque = rotation @ np.cross(THRUSTERS[index].position, np.asarray(THRUSTERS[index].direction) * config.thruster_force * level)
        mass = config.vehicle_mass + len(wheels) * config.wheel_mass
        expected = config.thruster_force * level * (steps / config.physics_hz) / mass
        projected = float(np.dot(velocity, direction))
        moment_response = float(np.dot(omega, torque))
        passed = math.isclose(projected, expected, rel_tol=.06, abs_tol=.0001)
        passed = passed and (moment_response > 0 if np.linalg.norm(torque) > 1e-8 else np.linalg.norm(omega) < 1e-7)
        return {'name': THRUSTERS[index].name, 'level': level, 'force_n': config.thruster_force * level,
                'duration_s': steps / config.physics_hz, 'expected_delta_v_m_s': expected,
                'measured_delta_v_along_force_m_s': projected,
                'linear_velocity_m_s': list(velocity), 'angular_velocity_rad_s': list(omega),
                'expected_torque_nm': torque.tolist(), 'passed': bool(passed)}
    finally:
        p.disconnect(physicsClientId=client)


def measure_all():
    rows = []
    for index in range(len(THRUSTERS)):
        row = pulse_response(index)
        half = pulse_response(index, level=.5)
        off = pulse_response(index, level=0)
        row['half_level_delta_v_m_s'] = half['measured_delta_v_along_force_m_s']
        row['off_delta_v_m_s'] = off['measured_delta_v_along_force_m_s']
        row['passed'] = row['passed'] and half['passed'] and off['passed']
        rows.append(row)
    return rows
