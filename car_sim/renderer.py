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


def _checker_quads(road, finish_s, rows=2, cols=10, square_len=0.8):
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


def render_chase(car, road, width_px=500, height_px=500, scale=6.0,
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

    center_pts = [world_to_screen(x, y) for x, y, *_ in road.waypoints]
    for i in range(0, len(center_pts) - 3, 6):
        cv2.line(img, center_pts[i], center_pts[i + 3], LANE_COLOR, 2)

    if finish_s is not None and road.reaches(finish_s):
        for world_corners, color in _checker_quads(road, finish_s):
            poly_pts = np.array([world_to_screen(x, y) for x, y in world_corners],
                                 dtype=np.int32)
            cv2.fillConvexPoly(img, poly_pts, color)

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


def render_camera(car, road, camera, width_px, height_px, render_distance=80.0,
                   finish_s=None):
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
    near = 0.05
    min_y = camera.horizon_row + 3

    # One extrinsics computation per frame, one batched matmul per point set -
    # the expensive Sutherland-Hodgman clip only runs for the few primitives
    # that actually straddle the near-plane/horizon boundary.
    R, t = camera.extrinsics(car.x, car.y, car.yaw)

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

    for i in range(0, len(wps) - 2, 4):
        j = i + 2
        if cv_valid[i] and cv_valid[j]:
            cv2.line(img, tuple(cp_i[i]), tuple(cp_i[j]), LANE_COLOR, 3)
        elif cv_valid[i] or cv_valid[j]:
            c0 = (wps[i][0], wps[i][1], 0.0)
            c1 = (wps[j][0], wps[j][1], 0.0)
            seg = _clipped_segment_pixels(camera, R, t, c0, c1, near, min_y)
            if seg is not None:
                cv2.line(img, seg[0], seg[1], LANE_COLOR, 3)

    if finish_s is not None and road.reaches(finish_s) \
            and car_s - 5.0 <= finish_s <= car_s + render_distance + 10.0:
        for world_corners, color in _checker_quads(road, finish_s):
            world_quad = [(x, y, 0.0) for x, y in world_corners]
            pts = _clipped_quad_pixels(camera, R, t, world_quad, near, min_y)
            if pts is not None:
                cv2.fillConvexPoly(img, pts, color)

    return img
