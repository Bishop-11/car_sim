import math
from dataclasses import dataclass


@dataclass
class CarState:
    """Kinematic bicycle-model car: a rigid body with a wheelbase, front
    wheels that steer, rear wheels that only roll (never slip sideways).
    Both yaw and speed are real integrated state, driven by two independent
    continuous commands each tick:

    - steering in [-1, 1]: -1 = full left, +1 = full right, proportional in
      between, mapped to a front-wheel angle delta = -steering * max_steer_angle.
      Turn rate omega = (speed / wheelbase) * tan(delta) - NOT commanded
      directly. At speed == 0, omega == 0 regardless of steering: turning
      the wheel while parked doesn't spin the car, same as a real car - you
      have to be rolling. omega is then capped so the implied lateral
      acceleration (speed * omega) never exceeds max_lat_accel: a tire can
      only generate so much sideways force before it slides, which is what
      actually stops a real car's turn rate from growing without bound as
      speed increases (the bicycle-model formula alone has no such limit).
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

    # Defaults here match sim_node's current config (config/sim_params.yaml);
    # kept in sync so a standalone CarState() behaves the same as the sim.
    wheelbase: float = 2.7           # meters, front-to-rear axle distance
    max_steer_angle: float = math.radians(35.0)  # radians, front wheel limit
    max_lat_accel: float = 14.71     # m/s^2, tire grip limit (~1.5 g)
    max_accel: float = 5.0           # m/s^2 (~0.5 g, realistic road car)
    max_decel: float = 10.0          # m/s^2, braking (~1.0 g, realistic road car)
    friction_decel: float = 2.0      # m/s^2, passive coast-down (~0.2 g)
    max_speed: float = 50.0          # m/s

    def step(self, steering, throttle, dt):
        steering = max(-1.0, min(1.0, steering))
        throttle = max(-1.0, min(1.0, throttle))

        delta = -steering * self.max_steer_angle
        omega = (self.speed / self.wheelbase) * math.tan(delta)
        lat_accel = self.speed * omega
        if abs(lat_accel) > self.max_lat_accel and self.speed > 1e-6:
            omega = math.copysign(self.max_lat_accel / self.speed, omega)
        self.omega = omega
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
