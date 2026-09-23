import math
import numpy as np


def quaternion_from_matrix(R):
    """Rotation matrix (3,3) -> quaternion (x, y, z, w). Standard robust
    (Shepperd's) method - safe for any rotation, not just small angles."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return x, y, z, w


class PinholeCamera:
    """A forward-looking pinhole camera, tilted down by `pitch_deg` from
    horizontal. `extrinsics()` mounts it directly on the car (at the car's
    x,y and this camera's mount_height, yawed with the car); `extrinsics_at()`
    places it at an arbitrary world pose instead, e.g. pulled back and raised
    for a third-person chase view.
    """

    def __init__(self, width_px, height_px, hfov_deg, mount_height, pitch_deg):
        self.width_px = width_px
        self.height_px = height_px
        hfov = math.radians(hfov_deg)
        self.fx = (width_px / 2.0) / math.tan(hfov / 2.0)
        self.fy = self.fx
        self.cx = width_px / 2.0
        self.cy = height_px / 2.0
        self.mount_height = mount_height
        self.pitch = math.radians(pitch_deg)
        # row where a level (z=0) ground plane vanishes to infinity, derived
        # from the camera tilt: v = cy - fy*tan(pitch)
        self.horizon_row = int(round(
            max(0, min(height_px, self.cy - self.fy * math.tan(self.pitch)))))

    def extrinsics(self, car_x, car_y, car_yaw):
        """Extrinsics for the camera mounted directly on the car (at the
        car's own x,y, at this camera's mount_height)."""
        return self.extrinsics_at(car_x, car_y, self.mount_height, car_yaw)

    def extrinsics_at(self, cam_x, cam_y, cam_z, yaw):
        """Rotation/translation mapping a world point p to camera frame: p_cam = R @ p + t,
        for a camera at an arbitrary world position (cam_x, cam_y, cam_z) yawed by `yaw`.

        Camera frame follows OpenCV convention: X-right, Y-down, Z-forward.
        """
        pitch = self.pitch
        forward = np.array([
            math.cos(yaw) * math.cos(pitch),
            math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
        ])
        world_up = np.array([0.0, 0.0, 1.0])
        x_cam = np.cross(forward, world_up)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(forward, x_cam)  # "down" in camera frame

        R = np.stack([x_cam, y_cam, forward], axis=0)
        cam_pos = np.array([cam_x, cam_y, cam_z])
        t = -R @ cam_pos
        return R, t

    def project(self, points_world, car_x, car_y, car_yaw, near=0.05):
        """Project Nx3 world points to pixel coords.

        Returns (pixels Nx2 float array, valid Nx bool array). Points behind
        the near plane are marked invalid and must not be trusted/drawn.
        """
        R, t = self.extrinsics(car_x, car_y, car_yaw)
        pts = np.asarray(points_world, dtype=np.float64)
        cam_pts = pts @ R.T + t
        depth = cam_pts[:, 2]
        valid = depth > near
        safe_depth = np.where(valid, depth, 1.0)
        u = self.fx * (cam_pts[:, 0] / safe_depth) + self.cx
        v = self.fy * (cam_pts[:, 1] / safe_depth) + self.cy
        pixels = np.stack([u, v], axis=1)
        return pixels, valid

    def project_camera_points(self, cam_pts):
        """Project already-camera-space points (Nx3, z assumed > 0) to pixel coords."""
        depth = cam_pts[:, 2]
        u = self.fx * (cam_pts[:, 0] / depth) + self.cx
        v = self.fy * (cam_pts[:, 1] / depth) + self.cy
        return np.stack([u, v], axis=1)

    def body_frame_pose(self):
        """Camera position + orientation in the car's own body frame
        (x-forward, y-left, z-up, origin at the car), independent of the
        car's world pose - what a perception node would receive as
        extrinsics on a real vehicle (no global localization assumed).

        Returns (position (3,), R_body_from_cam (3,3)) where a camera-frame
        vector v_cam transforms to the body frame via R_body_from_cam @ v_cam.
        """
        # extrinsics_at with car pose (0,0,0)/yaw=0 makes "world" and "body
        # frame" coincide, giving R mapping body -> camera; we want the
        # inverse (camera -> body), which for a rotation matrix is just
        # the transpose.
        R_cam_from_body, _ = self.extrinsics_at(0.0, 0.0, self.mount_height, 0.0)
        R_body_from_cam = R_cam_from_body.T
        position = np.array([0.0, 0.0, self.mount_height])
        return position, R_body_from_cam
