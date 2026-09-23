import math
from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import cv2


@runtime_checkable
class RoadLike(Protocol):
    """The road interface every render function in this module depends on.

    This file never imports road.py or RoadGenerator - it only calls methods
    on whatever `road` object is passed in. Any object implementing this
    interface (a simpler road, a more complex one, a fixed hand-authored
    track, ...) can be rendered by this module unmodified; nothing here
    needs to change.

    Waypoints are 5-tuples (x, y, heading, curvature, s) in world
    coordinates: x/y is the centerline position, heading is used to find the
    left/right perpendicular direction (for road edges, poles, the finish
    checker), curvature is unused by rendering, and s is arc-length from the
    road's start (used for the visible-window cutoff, the dash pattern, and
    pole/finish placement).
    """

    width: float
    waypoints: List[Tuple[float, float, float, float, float]]

    def nearest_waypoint(
        self, x: float, y: float
    ) -> Tuple[Optional[Tuple[float, float, float, float, float]], Optional[float]]:
        """Nearest waypoint to a world point (x, y), and the distance to it."""
        ...

    def waypoint_at_s(self, s: float) -> Tuple[float, float, float, float, float]:
        """Nearest generated waypoint to arc-length s."""
        ...

    def reaches(self, s: float) -> bool:
        """Whether the road has been generated at least as far as s."""
        ...

    def poles_in_range(self, s_start: float, s_end: float) -> List[Tuple[float, float]]:
        """(x, y) of roadside poles with arc-length in [s_start, s_end]."""
        ...

    def lane_dash_on(self, s: float) -> bool:
        """Whether arc-length s is in the 'on' part of the centerline dash pattern."""
        ...


ROAD_COLOR = (60, 60, 60)        # BGR dark gray
GRASS_COLOR = (40, 90, 40)       # BGR green
LANE_COLOR = (30, 220, 230)      # BGR yellow-ish
EDGE_COLOR = (255, 255, 255)     # white
CAR_COLOR = (0, 0, 220)          # red
CAR_TOP_COLOR = (0, 0, 150)      # darker red, shading cue for the car's roof
SKY_COLOR = (180, 140, 90)       # BGR light blue-ish
POLE_COLOR = (110, 110, 120)     # gray
LAMP_COLOR = (60, 200, 255)      # warm yellow-orange

_PIX_CLIP = 6000  # guard against overflow when drawing far/behind-camera points


def _clip_polygon_gt(points, coord_index, threshold):
    """Sutherland-Hodgman clip: keep the part of a convex polygon where
    points[:, coord_index] > threshold. `points` is a list of 1D arrays
    (any dimension >= coord_index+1); works for both 3D camera-space
    near-plane clipping and 2D pixel-space horizon clipping."""
    n = len(points)
    if n == 0:
        return []
    out = []
    for i in range(n):
        curr = points[i]
        nxt = points[(i + 1) % n]
        curr_in = curr[coord_index] > threshold
        next_in = nxt[coord_index] > threshold
        if curr_in:
            out.append(curr)
        if curr_in != next_in:
            t = (threshold - curr[coord_index]) / (nxt[coord_index] - curr[coord_index])
            out.append(curr + t * (nxt - curr))
    return out


def _clip_segment_gt(p0, p1, coord_index, threshold):
    """Clip a 2-point segment to the region points[coord_index] > threshold."""
    in0 = p0[coord_index] > threshold
    in1 = p1[coord_index] > threshold
    if not in0 and not in1:
        return None
    if in0 and in1:
        return p0, p1
    t = (threshold - p0[coord_index]) / (p1[coord_index] - p0[coord_index])
    ip = p0 + t * (p1 - p0)
    return (p0, ip) if in0 else (ip, p1)


def _clipped_quad_pixels(camera, R, t, world_quad, near, min_y):
    """World-space quad -> pixel polygon, correctly clipped against the
    camera near plane and the horizon row. Returns an int32 Nx2 array
    (N may be 0, 3, 4, or 5) ready for cv2.fillConvexPoly, or None.

    Takes a precomputed (R, t) extrinsics pair (see PinholeCamera.extrinsics)
    instead of recomputing it - this is only called for the handful of
    boundary primitives per frame, so the batch fast path in render_camera
    stays vectorized and this stays cheap.
    """
    cam_pts = np.array(world_quad) @ R.T + t
    clipped = _clip_polygon_gt(list(cam_pts), 2, near)
    if len(clipped) < 3:
        return None
    proj = camera.project_camera_points(np.array(clipped))
    proj_clipped = _clip_polygon_gt(list(proj), 1, min_y)
    if len(proj_clipped) < 3:
        return None
    pts = np.clip(np.array(proj_clipped), -_PIX_CLIP, _PIX_CLIP).astype(np.int32)
    return pts


def _clipped_segment_pixels(camera, R, t, p0_world, p1_world, near, min_y):
    """World-space segment -> pixel segment, clipped against the near plane
    and horizon row. Returns (px0, px1) int tuples or None. Takes a
    precomputed (R, t) extrinsics pair; see _clipped_quad_pixels."""
    cam_pts = np.array([p0_world, p1_world]) @ R.T + t
    clipped = _clip_segment_gt(cam_pts[0], cam_pts[1], 2, near)
    if clipped is None:
        return None
    proj = camera.project_camera_points(np.array(clipped))
    proj_clipped = _clip_segment_gt(proj[0], proj[1], 1, min_y)
    if proj_clipped is None:
        return None
    p0, p1 = proj_clipped
    p0 = np.clip(p0, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)
    p1 = np.clip(p1, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)
    return tuple(p0), tuple(p1)


def _edges(waypoints, half_width):
    lefts, rights = [], []
    for x, y, heading, _, _ in waypoints:
        nx, ny = -math.sin(heading), math.cos(heading)
        lefts.append((x + nx * half_width, y + ny * half_width))
        rights.append((x - nx * half_width, y - ny * half_width))
    return lefts, rights


def _car_outline_world(car, length=4.8, width=2.7):
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


def _car_box_world(car, length=4.2, width=2.0, height=1.4):
    """8 world-space corners of a simple box car model."""
    cos_y, sin_y = math.cos(car.yaw), math.sin(car.yaw)

    def to_world(lx, ly, lz):
        wx = car.x + lx * cos_y - ly * sin_y
        wy = car.y + lx * sin_y + ly * cos_y
        return (wx, wy, lz)

    hl, hw = length / 2.0, width / 2.0
    return dict(
        rbl=to_world(-hl, hw, 0.0), rbr=to_world(-hl, -hw, 0.0),
        rtl=to_world(-hl, hw, height), rtr=to_world(-hl, -hw, height),
        ftl=to_world(hl, hw, height), ftr=to_world(hl, -hw, height),
    )


def _draw_car_box(img, camera, R, t, car, near=0.05):
    """Draw a simple 3D box car model (rear + top faces) for third-person
    views. Near-plane clipped only - the horizon cut doesn't apply to a
    nearby elevated object like the car's roof, which can legitimately
    project above the ground-vanishing row."""
    c = _car_box_world(car)
    faces = [
        ([c['rtl'], c['rtr'], c['rbr'], c['rbl']], CAR_COLOR),      # rear face
        ([c['rtl'], c['rtr'], c['ftr'], c['ftl']], CAR_TOP_COLOR),  # top face
    ]
    for corners, color in faces:
        cam_pts = np.array(corners) @ R.T + t
        clipped = _clip_polygon_gt(list(cam_pts), 2, near)
        if len(clipped) < 3:
            continue
        proj = camera.project_camera_points(np.array(clipped))
        pts = np.clip(proj, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)
        cv2.fillConvexPoly(img, pts, color)


def _draw_pole(img, camera, R, t, base_xy, height, near=0.05):
    """Draw one vertical pole (shaft + lamp cap). Near-plane clipped only -
    like the car box, a pole is a real object that can legitimately project
    above the ground-vanishing row, so the horizon cut doesn't apply here."""
    base = (base_xy[0], base_xy[1], 0.0)
    top = (base_xy[0], base_xy[1], height)
    cam_pts = np.array([base, top]) @ R.T + t

    clipped = _clip_segment_gt(cam_pts[0], cam_pts[1], 2, near)
    if clipped is not None:
        proj = camera.project_camera_points(np.array(clipped))
        p0 = tuple(np.clip(proj[0], -_PIX_CLIP, _PIX_CLIP).astype(np.int32))
        p1 = tuple(np.clip(proj[1], -_PIX_CLIP, _PIX_CLIP).astype(np.int32))
        cv2.line(img, p0, p1, POLE_COLOR, 3)

    if cam_pts[1, 2] > near:  # top point itself is in front of the camera
        top_proj = camera.project_camera_points(cam_pts[1:2])[0]
        tp = tuple(np.clip(top_proj, -_PIX_CLIP, _PIX_CLIP).astype(np.int32))
        cv2.circle(img, tp, 5, LAMP_COLOR, -1)


def _checker_quads(road: RoadLike, finish_s, rows=2, cols=10, square_len=0.8):
    """Checkered finish-line pattern as world-space quads centered at arc-length finish_s."""
    fx, fy, fh, _, _ = road.waypoint_at_s(finish_s)
    fwd = (math.cos(fh), math.sin(fh))
    left = (-math.sin(fh), math.cos(fh))
    half_w = road.width / 2.0
    row_h = road.width / rows
    total_len = cols * square_len

    quads = []
    for r in range(rows):
        w0 = -half_w + r * row_h
        w1 = w0 + row_h
        for c in range(cols):
            l0 = -total_len / 2.0 + c * square_len
            l1 = l0 + square_len
            color = (255, 255, 255) if (r + c) % 2 == 0 else (20, 20, 20)
            local_corners = [(l0, w0), (l1, w0), (l1, w1), (l0, w1)]
            world_corners = [
                (fx + lo * fwd[0] + wo * left[0], fy + lo * fwd[1] + wo * left[1])
                for lo, wo in local_corners
            ]
            quads.append((world_corners, color))
    return quads


def render_chase(car, road: RoadLike, width_px=500, height_px=500, scale=6.0,
                  anchor_x_frac=0.5, anchor_y_frac=0.5, finish_s=None):
    """Render the third-person chase-cam map view of the road + car.

    Car-relative: the car always points up-screen, placed at
    (anchor_x_frac, anchor_y_frac) of the canvas (fractions of width/height,
    0-1), so the layout scales seamlessly with width_px/height_px.
    finish_s -> if the road has been generated that far, draw a checkered
                finish-line band at that arc length.
    """
    img = np.empty((height_px, width_px, 3), dtype=np.uint8)
    img[:] = GRASS_COLOR

    theta = -(car.yaw - math.pi / 2.0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    anchor_x = width_px * anchor_x_frac
    anchor_y = height_px * anchor_y_frac

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

    for i in range(len(road.waypoints) - 1):
        s0, s1 = road.waypoints[i][4], road.waypoints[i + 1][4]
        if road.lane_dash_on(s0) and road.lane_dash_on(s1):
            p0 = world_to_screen(road.waypoints[i][0], road.waypoints[i][1])
            p1 = world_to_screen(road.waypoints[i + 1][0], road.waypoints[i + 1][1])
            cv2.line(img, p0, p1, LANE_COLOR, 2)

    if finish_s is not None and road.reaches(finish_s):
        for world_corners, color in _checker_quads(road, finish_s):
            poly_pts = np.array([world_to_screen(x, y) for x, y in world_corners],
                                 dtype=np.int32)
            cv2.fillConvexPoly(img, poly_pts, color)

    car_wp, _ = road.nearest_waypoint(car.x, car.y)
    car_s = car_wp[4] if car_wp else 0.0
    view_reach = max(width_px, height_px) / scale  # meters visible at this zoom
    for bx, by in road.poles_in_range(car_s - view_reach, car_s + view_reach):
        px, py = world_to_screen(bx, by)
        cv2.circle(img, (px, py), 4, POLE_COLOR, -1)
        cv2.circle(img, (px, py), 2, LAMP_COLOR, -1)

    car_screen = np.array([world_to_screen(x, y) for x, y in _car_outline_world(car)],
                           dtype=np.int32)
    cv2.fillConvexPoly(img, car_screen, CAR_COLOR)

    return img


def _scale_for_text_height(text, font, target_h_px, min_scale=0.3):
    """Pick a cv2 font scale so the rendered text is ~target_h_px tall."""
    (_, trial_h), _ = cv2.getTextSize(text, font, 1.0, 2)
    if trial_h <= 0:
        return max(min_scale, 1.0)
    return max(min_scale, target_h_px / trial_h)


def draw_distance_box(img, distance_m, margin_frac=0.02, text_height_frac=0.035):
    """Draw a bottom-left info box showing distance traveled.

    margin_frac/text_height_frac are fractions of min(width, height), so the
    box scales seamlessly with the canvas size.
    """
    out = img.copy()
    h, w = out.shape[:2]
    text = f"Distance: {distance_m:.1f} m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    ref = min(w, h)
    scale = _scale_for_text_height(text, font, text_height_frac * ref)
    thickness = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pad = int(0.5 * th)
    margin = int(margin_frac * ref)
    x0, y1 = margin, h - margin
    x1, y0 = x0 + tw + 2 * pad, y1 - (th + 2 * pad)

    band = out.copy()
    cv2.rectangle(band, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.55, out, 0.45, 0, out)
    cv2.putText(out, text, (x0 + pad, y1 - pad), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def draw_end_state_overlay(img, text, color, band_height_frac=0.22, text_height_frac=0.12):
    """Return a copy of img with a banner (e.g. GAME OVER / VICTORY!) over the middle.

    band_height_frac/text_height_frac are fractions of the canvas height.
    """
    out = img.copy()
    h, w = out.shape[:2]

    band_h = int(h * band_height_frac)
    y0, y1 = h // 2 - band_h // 2, h // 2 + band_h // 2
    band = out.copy()
    cv2.rectangle(band, (0, y0), (w, y1), (0, 0, 0), -1)
    cv2.addWeighted(band, 0.55, out, 0.45, 0, out)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _scale_for_text_height(text, font, text_height_frac * h)
    thickness = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    tx, ty = (w - tw) // 2, h // 2 + th // 2
    cv2.putText(out, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)
    return out


def _render_road_scene(img, camera, R, t, road: RoadLike, render_distance, finish_s,
                        pole_height=3.5):
    """Draw the road surface/edges/lane-markings/finish-checker/poles into
    img, using the given camera intrinsics and precomputed (R, t)
    extrinsics. Shared by render_camera (camera mounted on the car) and
    render_player_view (camera pulled back behind the car).

    The visible waypoint window is centered on the CAMERA's own position
    (recovered from R, t), not the car's - for render_player_view the
    camera sits meters behind the car, so windowing around the car's
    position instead would miss the near-field ground right under the
    camera and leave an undrawn gap there.
    """
    half_width = road.width / 2.0
    cam_pos = -R.T @ t
    cam_wp, _ = road.nearest_waypoint(cam_pos[0], cam_pos[1])
    cam_s = cam_wp[4] if cam_wp else 0.0
    wps = [wp for wp in road.waypoints if cam_s - 5.0 <= wp[4] <= cam_s + render_distance]
    if len(wps) < 2:
        return

    lefts, rights = _edges(wps, half_width)
    near = 0.05
    min_y = camera.horizon_row + 3

    # One batched matmul per point set - the expensive Sutherland-Hodgman
    # clip only runs for the few primitives that actually straddle the
    # near-plane/horizon boundary.
    def batch_project(points_3d):
        cam_pts = points_3d @ R.T + t
        depth = cam_pts[:, 2]
        near_valid = depth > near
        safe_depth = np.where(near_valid, depth, 1.0)
        u = camera.fx * (cam_pts[:, 0] / safe_depth) + camera.cx
        v = camera.fy * (cam_pts[:, 1] / safe_depth) + camera.cy
        pix = np.stack([u, v], axis=1)
        valid = near_valid & (v > min_y)
        return pix, valid

    lefts_3d = np.array([[x, y, 0.0] for x, y in lefts])
    rights_3d = np.array([[x, y, 0.0] for x, y in rights])
    centers_3d = np.array([[wp[0], wp[1], 0.0] for wp in wps])

    lp, lv = batch_project(lefts_3d)
    rp, rv = batch_project(rights_3d)
    cp, cv_valid = batch_project(centers_3d)

    lp_i = np.clip(lp, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)
    rp_i = np.clip(rp, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)
    cp_i = np.clip(cp, -_PIX_CLIP, _PIX_CLIP).astype(np.int32)

    for i in range(len(wps) - 1):
        flags = (lv[i], rv[i], rv[i + 1], lv[i + 1])
        if all(flags):
            quad = np.array([lp_i[i], rp_i[i], rp_i[i + 1], lp_i[i + 1]], dtype=np.int32)
            cv2.fillConvexPoly(img, quad, ROAD_COLOR)
        elif any(flags):
            world_quad = [(lefts[i][0], lefts[i][1], 0.0), (rights[i][0], rights[i][1], 0.0),
                          (rights[i + 1][0], rights[i + 1][1], 0.0), (lefts[i + 1][0], lefts[i + 1][1], 0.0)]
            pts = _clipped_quad_pixels(camera, R, t, world_quad, near, min_y)
            if pts is not None:
                cv2.fillConvexPoly(img, pts, ROAD_COLOR)

    for i in range(len(wps) - 1):
        if lv[i] and lv[i + 1]:
            cv2.line(img, tuple(lp_i[i]), tuple(lp_i[i + 1]), EDGE_COLOR, 2)
        elif lv[i] or lv[i + 1]:
            l0 = (lefts[i][0], lefts[i][1], 0.0)
            l1 = (lefts[i + 1][0], lefts[i + 1][1], 0.0)
            seg = _clipped_segment_pixels(camera, R, t, l0, l1, near, min_y)
            if seg is not None:
                cv2.line(img, seg[0], seg[1], EDGE_COLOR, 2)

        if rv[i] and rv[i + 1]:
            cv2.line(img, tuple(rp_i[i]), tuple(rp_i[i + 1]), EDGE_COLOR, 2)
        elif rv[i] or rv[i + 1]:
            r0 = (rights[i][0], rights[i][1], 0.0)
            r1 = (rights[i + 1][0], rights[i + 1][1], 0.0)
            seg = _clipped_segment_pixels(camera, R, t, r0, r1, near, min_y)
            if seg is not None:
                cv2.line(img, seg[0], seg[1], EDGE_COLOR, 2)

    for i in range(len(wps) - 1):
        if not (road.lane_dash_on(wps[i][4]) and road.lane_dash_on(wps[i + 1][4])):
            continue
        if cv_valid[i] and cv_valid[i + 1]:
            cv2.line(img, tuple(cp_i[i]), tuple(cp_i[i + 1]), LANE_COLOR, 3)
        elif cv_valid[i] or cv_valid[i + 1]:
            c0 = (wps[i][0], wps[i][1], 0.0)
            c1 = (wps[i + 1][0], wps[i + 1][1], 0.0)
            seg = _clipped_segment_pixels(camera, R, t, c0, c1, near, min_y)
            if seg is not None:
                cv2.line(img, seg[0], seg[1], LANE_COLOR, 3)

    if finish_s is not None and road.reaches(finish_s) \
            and cam_s - 5.0 <= finish_s <= cam_s + render_distance + 10.0:
        for world_corners, color in _checker_quads(road, finish_s):
            world_quad = [(x, y, 0.0) for x, y in world_corners]
            pts = _clipped_quad_pixels(camera, R, t, world_quad, near, min_y)
            if pts is not None:
                cv2.fillConvexPoly(img, pts, color)

    for base_xy in road.poles_in_range(cam_s - 5.0, cam_s + render_distance):
        _draw_pole(img, camera, R, t, base_xy, pole_height, near)


def render_camera(car, road: RoadLike, camera, width_px, height_px, render_distance=80.0,
                   finish_s=None, pole_height=3.5):
    """Render the front-facing perspective camera view via pinhole projection.
    Camera is mounted directly on the car - the car itself is invisible
    (it's the point of view), only the road ahead is drawn."""
    img = np.empty((height_px, width_px, 3), dtype=np.uint8)
    img[:] = SKY_COLOR
    img[camera.horizon_row:, :] = GRASS_COLOR

    R, t = camera.extrinsics(car.x, car.y, car.yaw)
    _render_road_scene(img, camera, R, t, road, render_distance, finish_s, pole_height)
    return img


def render_player_view(car, road: RoadLike, camera, width_px, height_px, render_distance,
                        back_offset, cam_height, finish_s=None, pole_height=3.5):
    """Render a pulled-back third-person perspective view (Need-For-Speed
    style chase cam): a real pinhole camera positioned `back_offset` meters
    behind and `cam_height` meters above the car, yawed to match the car's
    heading, looking forward at both the car and the road ahead."""
    img = np.empty((height_px, width_px, 3), dtype=np.uint8)
    img[:] = SKY_COLOR
    img[camera.horizon_row:, :] = GRASS_COLOR

    cam_x = car.x - back_offset * math.cos(car.yaw)
    cam_y = car.y - back_offset * math.sin(car.yaw)
    R, t = camera.extrinsics_at(cam_x, cam_y, cam_height, car.yaw)

    _render_road_scene(img, camera, R, t, road, render_distance, finish_s, pole_height)
    _draw_car_box(img, camera, R, t, car)
    return img


def _draw_bar_gauge(img, x, y, w, h, value, vmin, vmax, fill_color,
                     bg_color=(60, 60, 60), border_color=(210, 210, 210)):
    """Horizontal bar gauge in [vmin, vmax] at pixel rect (x, y, w, h).
    If vmin < 0 < vmax, fills from the zero line (with a zero marker)
    instead of from the left - correct reading for signed quantities like
    steering, acceleration, or yaw rate."""
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    frac = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
    frac = max(0.0, min(1.0, frac))
    fill_x = x + int(round(w * frac))
    if vmin < 0.0 < vmax:
        zero_x = x + int(round(w * (-vmin) / (vmax - vmin)))
        lo, hi = (zero_x, fill_x) if fill_x >= zero_x else (fill_x, zero_x)
        cv2.rectangle(img, (lo, y), (hi, y + h), fill_color, -1)
        cv2.line(img, (zero_x, y), (zero_x, y + h), (255, 255, 255), 1)
    else:
        cv2.rectangle(img, (x, y), (fill_x, y + h), fill_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 1)


def draw_telemetry_hud(img, *, throttle, accel, max_accel, max_decel,
                        speed, max_speed, steering, omega, max_omega, yaw,
                        distance_traveled, target_distance, elapsed_s):
    """Draw two HUD panels on img (a copy is returned, img is untouched):

    top-left  - throttle, acceleration, speed, steering, yaw rate, yaw
    top-right - elapsed time, distance traveled vs. target

    All sizing is fraction-based (of min(width, height)) so the HUD scales
    seamlessly with the canvas, same convention as draw_distance_box/
    draw_end_state_overlay.
    """
    out = img.copy()
    h, w = out.shape[:2]
    ref = min(w, h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (235, 235, 235)
    GREEN, RED, CYAN, BLUE = (60, 200, 60), (50, 50, 230), (230, 210, 60), (230, 160, 60)

    def panel_bg(x0, y0, x1, y1):
        band = out.copy()
        cv2.rectangle(band, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(band, 0.5, out, 0.5, 0, out)

    def make_metrics(panel_scale):
        """Panel sizing (margin/text/bars/row height/width), all derived
        from `ref` and scaled by `panel_scale` - lets one panel (e.g. the
        top-left gauges) be sized independently of another (top-right)."""
        margin = max(2, int(0.018 * ref * panel_scale))
        text_h = max(8, int(0.024 * ref * panel_scale))
        scale = _scale_for_text_height("Hg", font, text_h)
        thickness = max(1, int(round(scale * 1.6)))
        bar_h = max(3, int(text_h * 0.55))
        gap = max(2, int(text_h * 0.35))
        row_h = text_h + gap + bar_h + gap
        panel_w = int(0.32 * w * panel_scale)
        return dict(margin=margin, text_h=text_h, scale=scale, thickness=thickness,
                    bar_h=bar_h, gap=gap, row_h=row_h, panel_w=panel_w)

    def bar_row(m, x0, y0, i, label_value, value, vmin, vmax, color):
        y = y0 + m['margin'] + i * m['row_h']
        cv2.putText(out, label_value, (x0 + m['margin'], y + m['text_h']), font, m['scale'],
                    text_color, m['thickness'], cv2.LINE_AA)
        _draw_bar_gauge(out, x0 + m['margin'], y + m['text_h'] + m['gap'],
                         m['panel_w'] - 2 * m['margin'], m['bar_h'], value, vmin, vmax, color)

    # ---- top-left: throttle, accel, speed, steering, omega, yaw (30% smaller) ----
    lm = make_metrics(panel_scale=0.7)
    lx0, ly0 = lm['margin'], lm['margin']
    n_left = 6
    panel_bg(lx0, ly0, lx0 + lm['panel_w'], ly0 + n_left * lm['row_h'] + lm['margin'])

    bar_row(lm, lx0, ly0, 0, f"THROTTLE  {throttle:+.2f}", throttle, -1.0, 1.0,
            GREEN if throttle >= 0 else RED)
    bar_row(lm, lx0, ly0, 1, f"ACCEL     {accel:+.2f} m/s2", accel, -max_decel, max_accel,
            GREEN if accel >= 0 else RED)
    bar_row(lm, lx0, ly0, 2, f"SPEED     {speed:.2f} m/s", speed, 0.0, max_speed, CYAN)
    bar_row(lm, lx0, ly0, 3, f"STEERING  {steering:+.2f}", steering, -1.0, 1.0, BLUE)
    bar_row(lm, lx0, ly0, 4, f"OMEGA     {omega:+.2f} rad/s", omega, -max_omega, max_omega, BLUE)
    yaw_deg = math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))
    bar_row(lm, lx0, ly0, 5, f"YAW       {yaw_deg:+.1f} deg", yaw_deg, -180.0, 180.0, BLUE)

    # ---- top-right: elapsed time, distance vs. target ----
    rm = make_metrics(panel_scale=1.0)
    rx1 = w - rm['margin']
    rx0 = rx1 - rm['panel_w']
    ry0 = rm['margin']
    n_right = 2
    panel_bg(rx0, ry0, rx1, ry0 + n_right * rm['row_h'] + rm['margin'])

    mm, ss = divmod(max(0.0, elapsed_s), 60.0)
    time_text = f"TIME      {int(mm):02d}:{ss:04.1f}"
    y = ry0 + rm['margin']
    cv2.putText(out, time_text, (rx0 + rm['margin'], y + rm['text_h']), font, rm['scale'],
                text_color, rm['thickness'], cv2.LINE_AA)

    bar_row(rm, rx0, ry0, 1, f"DIST  {distance_traveled:.0f} / {target_distance:.0f} m",
            distance_traveled, 0.0, max(target_distance, 1.0), CYAN)

    return out
