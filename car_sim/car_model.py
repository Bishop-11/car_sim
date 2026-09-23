import math
from dataclasses import dataclass


@dataclass
class CarState:
    """Point-mass kinematic car. Both yaw and speed are real integrated
    state, driven by two independent continuous commands each tick:

    - steering in [-1, 1]: -1 = full left, +1 = full right, proportional in
      between. omega = -steering * yaw_rate (yaw_rate is the max/full-lock
      turn rate).
    - throttle in [-1, 1]: >0 accelerates (proportional up to max_accel),
      <0 brakes (proportional up to max_decel, stronger than max_accel,
      same as a real car). At throttle == 0, a small constant
      friction_decel coasts the car down, like rolling resistance/drag.
      Speed is clamped to [0, max_speed] - no reverse.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0  # radians, CCW from +X
    speed: float = 0.0
    distance_traveled: float = 0.0
    omega: float = 0.0  # rad/s, last-applied yaw rate (read-only, set by step())
    accel: float = 0.0  # m/s^2, last-applied acceleration (read-only, set by step())

    yaw_rate: float = 1.0        # rad/s, max turn rate
    max_accel: float = 3.0       # m/s^2
    max_decel: float = 6.0       # m/s^2, braking
    friction_decel: float = 1.0  # m/s^2, passive coast-down
    max_speed: float = 15.0      # m/s

    def step(self, steering, throttle, dt):
        steering = max(-1.0, min(1.0, steering))
        throttle = max(-1.0, min(1.0, throttle))

        self.omega = -steering * self.yaw_rate
        self.yaw += self.omega * dt

        if throttle > 0.0:
            self.accel = throttle * self.max_accel
        elif throttle < 0.0:
            self.accel = throttle * self.max_decel
        else:
            self.accel = -self.friction_decel
        self.speed = max(0.0, min(self.max_speed, self.speed + self.accel * dt))

        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt
        self.distance_traveled += self.speed * dt

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.speed = 0.0
        self.distance_traveled = 0.0
        self.omega = 0.0
        self.accel = 0.0
