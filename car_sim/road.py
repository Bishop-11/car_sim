import math
import random


class RoadGenerator:
    """Procedurally generates an endless road centerline ahead of the car.

    Curvature follows a rate-limited random walk: every `segment_length`
    meters a new target curvature is sampled, and the current curvature is
    steered toward it at most `curvature_rate` per meter, so the road never
    kinks. Waypoints are stored as (x, y, heading, curvature, s).
    """

    def __init__(self, seed, width=6.0, waypoint_spacing=0.5,
                 segment_length=40.0, max_curvature=0.06,
                 curvature_rate=0.02, lookahead=150.0, keep_behind=20.0,
                 finish_distance=None, straight_start=0.0):
        self.width = width
        self.ds = waypoint_spacing
        self.segment_length = segment_length
        self.max_curvature = max_curvature
        self.curvature_rate = curvature_rate
        self.lookahead = lookahead
        self.keep_behind = keep_behind
        self.finish_distance = finish_distance
        self.straight_start = straight_start
        self.seed = seed

        self._rng = random.Random(seed)
        self._x = 0.0
        self._y = 0.0
        self._heading = 0.0
        self._s = 0.0
        self._curvature = 0.0
        self._target_curvature = 0.0
        self._dist_to_next_target = 0.0

        self.waypoints = [(0.0, 0.0, 0.0, 0.0, 0.0)]
        self.extend_to(self.lookahead)

    def reset(self, seed=None):
        if seed is None:
            seed = self.seed
        self.__init__(seed, self.width, self.ds, self.segment_length,
                       self.max_curvature, self.curvature_rate,
                       self.lookahead, self.keep_behind, self.finish_distance,
                       self.straight_start)

    def _step(self):
        if self._s >= self.straight_start and self._dist_to_next_target <= 0.0:
            self._target_curvature = self._rng.uniform(
                -self.max_curvature, self.max_curvature)
            self._dist_to_next_target = self.segment_length

        max_delta = self.curvature_rate * self.ds
        diff = self._target_curvature - self._curvature
        step = max(-max_delta, min(max_delta, diff))
        self._curvature += step

        self._heading += self._curvature * self.ds
        self._x += math.cos(self._heading) * self.ds
        self._y += math.sin(self._heading) * self.ds
        self._s += self.ds
        self._dist_to_next_target -= self.ds

        self.waypoints.append(
            (self._x, self._y, self._heading, self._curvature, self._s))

    def extend_to(self, s_target):
        if self.finish_distance is not None:
            s_target = min(s_target, self.finish_distance)
        while self._s < s_target:
            self._step()

    def update(self, car_s):
        """Ensure generated road covers [car_s - keep_behind, car_s + lookahead]."""
        self.extend_to(car_s + self.lookahead)
        cutoff = car_s - self.keep_behind
        while len(self.waypoints) > 2 and self.waypoints[1][4] < cutoff:
            self.waypoints.pop(0)

    def waypoint_at_s(self, s):
        """Nearest generated waypoint to arc-length s (list is short and s-ordered)."""
        return min(self.waypoints, key=lambda wp: abs(wp[4] - s))

    def reaches(self, s):
        """Whether the road has been generated at least as far as s."""
        return self.waypoints[-1][4] >= s - 1e-6

    def nearest_waypoint(self, x, y):
        """Brute-force nearest waypoint to a world point (list is short)."""
        best = None
        best_d2 = None
        for wp in self.waypoints:
            d2 = (wp[0] - x) ** 2 + (wp[1] - y) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best = wp
        return best, (math.sqrt(best_d2) if best_d2 is not None else None)

    def lateral_offset(self, x, y):
        """Signed lateral distance of (x, y) from the centerline: +left, -right."""
        wp, _ = self.nearest_waypoint(x, y)
        if wp is None:
            return 0.0
        wx, wy, wheading, _, _ = wp
        dx = x - wx
        dy = y - wy
        nx, ny = -math.sin(wheading), math.cos(wheading)
        return dx * nx + dy * ny
