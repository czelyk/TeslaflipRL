"""Static, concave reverse-C track, sampled into tangent collision boxes."""
import math
import numpy as np
import pybullet as p


def ramp_point(theta, radius=9.0):
    return np.array([radius * math.sin(theta), 0.0, radius * (1 - math.cos(theta))], dtype=float)


def ramp_tangent(theta):
    return np.array([math.cos(theta), 0.0, math.sin(theta)], dtype=float)


def ramp_normal(theta):
    return np.array([-math.sin(theta), 0.0, math.cos(theta)], dtype=float)


def build_ramp(client_id, config):
    ids = []
    delta = config.ramp_angle / config.ramp_segments
    half_thickness = 0.16
    collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[config.ramp_radius * delta / 2 + 0.012, config.ramp_width / 2, half_thickness], physicsClientId=client_id)
    visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[config.ramp_radius * delta / 2 + 0.008, config.ramp_width / 2, half_thickness], rgbaColor=[0.17, 0.24, 0.32, 1], physicsClientId=client_id)
    for i in range(config.ramp_segments):
        theta = (i + 0.5) * delta
        center = ramp_point(theta, config.ramp_radius) - ramp_normal(theta) * half_thickness
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual, basePosition=center, baseOrientation=p.getQuaternionFromEuler([0, -theta, 0]), physicsClientId=client_id)
        p.changeDynamics(body, -1, lateralFriction=1.3, restitution=0, physicsClientId=client_id)
        ids.append(body)
    return ids
