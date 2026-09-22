import math
import numpy as np


class PinholeCamera:
    """Forward-facing pinhole camera rigidly mounted on the car.

    Mounted at a fixed height above the car's (x, y) position, tilted down
    by `pitch_deg` from horizontal, yawed with the car's heading.
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
        """Rotation/translation mapping a world point p to camera frame: p_cam = R @ p + t.

        Camera frame follows OpenCV convention: X-right, Y-down, Z-forward.
        """
        yaw = car_yaw
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
        cam_pos = np.array([car_x, car_y, self.mount_height])
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
