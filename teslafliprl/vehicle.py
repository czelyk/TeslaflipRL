"""Four real revolute wheels and spatially separated, bounded force actuators."""
from dataclasses import dataclass
import math
import numpy as np
import pybullet as p


@dataclass(frozen=True)
class Thruster:
    name: str
    position: tuple
    direction: tuple


THRUSTERS = (
    Thruster("front_up", (1.05, 0, 0.05), (0, 0, 1)),
    Thruster("front_down", (1.05, 0, 0.05), (0, 0, -1)),
    Thruster("rear_up", (-1.05, 0, 0.05), (0, 0, 1)),
    Thruster("rear_down", (-1.05, 0, 0.05), (0, 0, -1)),
    Thruster("left_up", (0, 0.68, 0), (0, 0, 1)),
    Thruster("left_down", (0, 0.68, 0), (0, 0, -1)),
    Thruster("right_up", (0, -0.68, 0), (0, 0, 1)),
    Thruster("right_down", (0, -0.68, 0), (0, 0, -1)),
    Thruster("front_left", (1.05, 0, 0), (0, 1, 0)),
    Thruster("front_right", (1.05, 0, 0), (0, -1, 0)),
    Thruster("rear_left", (-1.05, 0, 0), (0, 1, 0)),
    Thruster("rear_right", (-1.05, 0, 0), (0, -1, 0)),
    Thruster("bottom_lift", (0, 0, -0.28), (0, 0, 1)),
)


def torque_allocation_matrix(thrusters=THRUSTERS):
    return np.stack([np.cross(t.position, t.direction) for t in thrusters], axis=1)


def nozzle_position(thruster):
    """Move along the force's line to an exposed outlet; r x F is unchanged."""
    point = np.asarray(thruster.position, dtype=float).copy()
    direction = np.asarray(thruster.direction, dtype=float)
    axis = int(np.argmax(np.abs(direction)))
    point[axis] = -direction[axis] * ([1.22, .64, .25][axis] + .08)
    return point


def _outlet_orientation(direction):
    exhaust = -np.asarray(direction, dtype=float)
    if exhaust[2] < -.999:
        return p.getQuaternionFromEuler([math.pi, 0, 0])
    if exhaust[2] > .999:
        return [0, 0, 0, 1]
    cross = np.cross([0, 0, 1], exhaust)
    quat = np.r_[cross, 1 + exhaust[2]]
    return (quat / np.linalg.norm(quat)).tolist()


def update_thruster_visuals(client_id, car_id, levels, previous=None):
    """Fixed, massless visual links: real outlets and orange exhaust cones."""
    state = tuple(max(1, int(round(float(level) * 10))) if level > .001 else 0 for level in levels)
    for index, value in enumerate(state):
        if previous is not None and value == previous[index]:
            continue
        p.changeVisualShape(car_id, 4 + index, rgbaColor=[1, .55, .05, 1] if value else [.1, .85, .85, 1], physicsClientId=client_id)
        p.changeVisualShape(car_id, 4 + len(THRUSTERS) + index,
                            rgbaColor=[1, .25 + .04 * value, .02, 1 if value else 0], physicsClientId=client_id)
    return state


def build_vehicle(client_id, config, position):
    chassis_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.22, 0.64, 0.25], physicsClientId=client_id)
    chassis_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.22, 0.64, 0.25], rgbaColor=[0.88, 0.12, 0.13, 1], physicsClientId=client_id)
    wheel_orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    wheel_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=config.wheel_radius, height=0.22, collisionFrameOrientation=wheel_orientation, physicsClientId=client_id)
    wheel_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=config.wheel_radius, length=0.22, visualFrameOrientation=wheel_orientation, rgbaColor=[0.035, 0.035, 0.045, 1], physicsClientId=client_id)
    wheel_positions = [[x, y, -0.24] for x in (0.90, -0.90) for y in (0.78, -0.78)]
    nozzle_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=.13, length=.16,
                                   rgbaColor=[.1, .85, .85, 1], physicsClientId=client_id)
    vertices = [[0, 0, 1.0], [0, 0, 0]] + [[.13 * math.cos(i * math.pi / 4), .13 * math.sin(i * math.pi / 4), 0] for i in range(8)]
    triangles = []
    for i in range(8):
        a, b = 2 + i, 2 + (i + 1) % 8
        triangles.extend([0, a, b, 1, b, a])
    plume_vis = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=triangles,
                                  rgbaColor=[1, .5, .02, 0], physicsClientId=client_id)
    points = [nozzle_position(t).tolist() for t in THRUSTERS]
    plume_points = [(nozzle_position(t) - np.asarray(t.direction) * .08).tolist() for t in THRUSTERS]
    orientations = [_outlet_orientation(t.direction) for t in THRUSTERS]
    count = len(THRUSTERS) * 2
    car = p.createMultiBody(baseMass=config.vehicle_mass, baseCollisionShapeIndex=chassis_col, baseVisualShapeIndex=chassis_vis, basePosition=position,
        linkMasses=[config.wheel_mass] * 4 + [0.] * count,
        linkCollisionShapeIndices=[wheel_col] * 4 + [-1] * count,
        linkVisualShapeIndices=[wheel_vis] * 4 + [nozzle_vis] * len(THRUSTERS) + [plume_vis] * len(THRUSTERS),
        linkPositions=wheel_positions + points + plume_points,
        linkOrientations=[[0, 0, 0, 1]] * 4 + orientations + orientations,
        linkInertialFramePositions=[[0, 0, 0]] * (4 + count),
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * (4 + count), linkParentIndices=[0] * (4 + count),
        linkJointTypes=[p.JOINT_REVOLUTE] * 4 + [p.JOINT_FIXED] * count,
        linkJointAxis=[[0, 1, 0]] * 4 + [[0, 0, 0]] * count, physicsClientId=client_id)
    p.changeDynamics(car, -1, lateralFriction=0.45, restitution=0.02, linearDamping=0.008, angularDamping=0.02, physicsClientId=client_id)
    for wheel in range(4):
        p.changeDynamics(car, wheel, lateralFriction=1.3, spinningFriction=0.001, rollingFriction=0.001, restitution=0.02, jointDamping=0.05, physicsClientId=client_id)
    return car, list(range(4))


def apply_thrusters(client_id, car_id, thrusters, levels, maximum_force):
    """Apply each force at its actual body-frame nozzle position, in world coordinates."""
    position, orientation = p.getBasePositionAndOrientation(car_id, physicsClientId=client_id)
    rotation = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
    origin = np.asarray(position)
    for nozzle, level in zip(thrusters, levels):
        if level > 0:
            force = rotation @ np.asarray(nozzle.direction) * (maximum_force * float(level))
            point = origin + rotation @ nozzle_position(nozzle)
            p.applyExternalForce(car_id, -1, force.tolist(), point.tolist(), p.WORLD_FRAME, physicsClientId=client_id)
