"""Simulation settings in SI units. This is a fictional rigid-body vehicle."""
from dataclasses import dataclass
import math


@dataclass
class EnvConfig:
    physics_hz: int = 240
    control_hz: int = 60
    ramp_radius: float = 9.0
    ramp_angle: float = 2.55
    ramp_width: float = 6.0
    ramp_segments: int = 100
    approach_speed: float = 25.0
    start_x: float = -12.0
    vehicle_mass: float = 650.0
    wheel_mass: float = 12.0
    wheel_radius: float = 0.34
    wheel_motor_torque: float = 900.0
    thruster_force: float = 2000.0
    max_episode_seconds: float = 14.0
    randomize: bool = True
    max_landing_speed: float = 4.0
    max_landing_angular_speed: float = 2.5
    settle_seconds: float = 0.20

    def __post_init__(self):
        if not isinstance(self.physics_hz, int) or not isinstance(self.control_hz, int) or self.physics_hz <= 0 or self.control_hz <= 0 or self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be a positive integer multiple of control_hz")
        if not math.pi / 2 < self.ramp_angle < math.pi:
            raise ValueError("ramp_angle must lie between pi/2 and pi for a leftward exit")
        if not isinstance(self.ramp_segments, int) or self.ramp_radius < 4 or self.ramp_width < 2.5 or self.ramp_segments < 20:
            raise ValueError("Ramp dimensions or segment count are too small")
        for name in ("approach_speed", "vehicle_mass", "wheel_mass", "wheel_radius", "thruster_force", "wheel_motor_torque", "max_episode_seconds", "settle_seconds", "max_landing_speed", "max_landing_angular_speed"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.start_x > -3:
            raise ValueError("start_x must provide a left-to-right ground approach")
