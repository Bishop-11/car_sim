import math
from dataclasses import dataclass


@dataclass
class CarState:
    """Point-mass kinematic car: constant forward speed, commanded yaw rate."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # radians, CCW from +X
    speed: float = 0.0
    distance_traveled: float = 0.0

    def step(self, control, speed, yaw_rate, dt):
        """control in {'left', 'right', 'straight'}."""
        if control == 'left':
            omega = yaw_rate
        elif control == 'right':
            omega = -yaw_rate
        else:
            omega = 0.0

        self.speed = speed
        self.yaw += omega * dt
        self.x += speed * math.cos(self.yaw) * dt
        self.y += speed * math.sin(self.yaw) * dt
        self.distance_traveled += speed * dt

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = 0.0
        self.distance_traveled = 0.0
