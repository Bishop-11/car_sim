import math
import numpy as np
import cv2

ROAD_COLOR = (60, 60, 60)        # BGR dark gray
GRASS_COLOR = (40, 90, 40)       # BGR green
LANE_COLOR = (30, 220, 230)      # BGR yellow-ish
EDGE_COLOR = (255, 255, 255)     # white
CAR_COLOR = (0, 0, 220)          # red
SKY_COLOR = (180, 140, 90)       # BGR light blue-ish

_PIX_CLIP = 6000  # guard against overflow when drawing far/behind-camera points
_MAX_SEG_PX = 120  # skip segments whose projected length explodes (near-horizon grazing angle)


def _seg_ok(p0, p1, max_px=_MAX_SEG_PX):
    return abs(int(p0[0]) - int(p1[0])) < max_px and abs(int(p0[1]) - int(p1[1])) < max_px


def _edges(waypoints, half_width):
    lefts, rights = [], []
    for x, y, heading, _, _ in waypoints:
        nx, ny = -math.sin(heading), math.cos(heading)
        lefts.append((x + nx * half_width, y + ny * half_width))
        rights.append((x - nx * half_width, y - ny * half_width))
    return lefts, rights


def _car_outline_world(car, length=1.6, width=0.9):
    """Local car-frame outline (x-forward, y-left) transformed to world coords."""
    local_pts = [(length * 0.55, 0.0),
                 (-length * 0.45, width / 2.0),
                 (-length * 0.45, -width / 2.0)]
    cos_y, sin_y = math.cos(car.yaw), math.sin(car.yaw)
    world_pts = []
    for lx, ly in local_pts:
        wx = car.x + lx * cos_y - ly * sin_y
        wy = car.y + lx * sin_y + ly * cos_y
        world_pts.append((wx, wy))
    return world_pts


def render_topdown(car, road, width_px=500, height_px=500, scale=6.0,
                    rotate_with_car=False, forward_offset_m=0.0):
    """Render a map view of the road + car.

    rotate_with_car=False -> north-up minimap, car centered (top-down state view).
    rotate_with_car=True  -> car-relative chase view: car always points
                              up-screen, placed `forward_offset_m` (negative
                              = behind the anchor) from image center.
    """
    img = np.empty((height_px, width_px, 3), dtype=np.uint8)
    img[:] = GRASS_COLOR

    theta = -(car.yaw - math.pi / 2.0) if rotate_with_car else 0.0
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    anchor_x = width_px / 2.0
    anchor_y = height_px / 2.0 + forward_offset_m * scale

    def world_to_screen(x, y):
        dx, dy = x - car.x, y - car.y
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        sx = anchor_x + rx * scale
        sy = anchor_y - ry * scale
        return (int(round(sx)), int(round(sy)))

    half_width = road.width / 2.0
    lefts, rights = _edges(road.waypoints, half_width)

    poly = [world_to_screen(x, y) for x, y in lefts] + \
           [world_to_screen(x, y) for x, y in reversed(rights)]
    cv2.fillPoly(img, [np.array(poly, dtype=np.int32)], ROAD_COLOR)

    left_pts = np.array([world_to_screen(x, y) for x, y in lefts], dtype=np.int32)
    right_pts = np.array([world_to_screen(x, y) for x, y in rights], dtype=np.int32)
    cv2.polylines(img, [left_pts], False, EDGE_COLOR, 2)
    cv2.polylines(img, [right_pts], False, EDGE_COLOR, 2)

    center_pts = [world_to_screen(x, y) for x, y, *_ in road.waypoints]
    for i in range(0, len(center_pts) - 3, 6):
        cv2.line(img, center_pts[i], center_pts[i + 3], LANE_COLOR, 2)

    car_screen = np.array([world_to_screen(x, y) for x, y in _car_outline_world(car)],
                           dtype=np.int32)
    cv2.fillConvexPoly(img, car_screen, CAR_COLOR)

    return img


def render_camera(car, road, camera, width_px, height_px, render_distance=80.0):
    """Render the front-facing perspective camera view via pinhole projection."""
    img = np.empty((height_px, width_px, 3), dtype=np.uint8)
    img[:] = SKY_COLOR
    horizon = camera.horizon_row
    img[horizon:, :] = GRASS_COLOR

    half_width = road.width / 2.0
    car_wp, _ = road.nearest_waypoint(car.x, car.y)
    car_s = car_wp[4] if car_wp else 0.0
    wps = [wp for wp in road.waypoints if car_s - 5.0 <= wp[4] <= car_s + render_distance]
    if len(wps) < 2:
        return img

    lefts, rights = _edges(wps, half_width)
    lefts_3d = np.array([[x, y, 0.0] for x, y in lefts])
    rights_3d = np.array([[x, y, 0.0] for x, y in rights])
    centers_3d = np.array([[wp[0], wp[1], 0.0] for wp in wps])

    lp, lv = camera.project(lefts_3d, car.x, car.y, car.yaw)
    rp, rv = camera.project(rights_3d, car.x, car.y, car.yaw)
    cp, cv_valid = camera.project(centers_3d, car.x, car.y, car.yaw)

    lp = np.clip(lp, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)
    rp = np.clip(rp, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)
    cp = np.clip(cp, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)

    # points that map at/above the vanishing row are numerically unstable
    # (ground-plane depth -> infinity); drop them rather than draw streaks
    min_y = camera.horizon_row + 3
    lv = lv & (lp[:, 1] > min_y)
    rv = rv & (rp[:, 1] > min_y)
    cv_valid = cv_valid & (cp[:, 1] > min_y)

    for i in range(len(wps) - 1):
        if lv[i] and lv[i + 1] and rv[i] and rv[i + 1] \
                and _seg_ok(lp[i], lp[i + 1]) and _seg_ok(rp[i], rp[i + 1]):
            quad = np.array([lp[i], rp[i], rp[i + 1], lp[i + 1]], dtype=np.int32)
            cv2.fillConvexPoly(img, quad, ROAD_COLOR)

    for i in range(len(wps) - 1):
        if lv[i] and lv[i + 1] and _seg_ok(lp[i], lp[i + 1]):
            cv2.line(img, tuple(lp[i]), tuple(lp[i + 1]), EDGE_COLOR, 2)
        if rv[i] and rv[i + 1] and _seg_ok(rp[i], rp[i + 1]):
            cv2.line(img, tuple(rp[i]), tuple(rp[i + 1]), EDGE_COLOR, 2)

    for i in range(0, len(wps) - 2, 4):
        j = i + 2
        if cv_valid[i] and cv_valid[j] and _seg_ok(cp[i], cp[j]):
            cv2.line(img, tuple(cp[i]), tuple(cp[j]), LANE_COLOR, 3)

    return img
