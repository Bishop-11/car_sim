import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32, Bool
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

from car_sim.road import RoadGenerator
from car_sim.car_model import CarState
from car_sim.camera_model import PinholeCamera, quaternion_from_matrix
from car_sim import render

# Difficulty is a fixed, code-level setting (not a launch parameter yet):
# difficulty N means the finish line is N km down the road.
DIFFICULTY = 1
FINISH_DISTANCE_M = DIFFICULTY * 1000.0


class SimNode(Node):
    """Owns car kinematics + the procedural road, and renders three views:

    /car/camera/image_raw      - front pinhole camera (for lane detection)
    /car/camera/camera_info    - front camera intrinsics (sensor_msgs/CameraInfo,
                                  like a real camera driver would publish)
    /car/camera/extrinsics     - front camera pose relative to the car's own
                                  body frame (geometry_msgs/PoseStamped) - not
                                  world pose; a perception node has no global
                                  localization, only "where am I on the car"
    /car/player_view/image_raw - pulled-back 3rd-person perspective chase cam
                                  (Need-For-Speed style); this is what manual
                                  play's player window displays and drives from
    /car/chase/image_raw       - car-relative map view; visualization only
    /car/game_over             - latched off-road/victory flag
    /car/reset (service)       - respawn car + regenerate road

    /car/steering (std_msgs/Float32) - steering command in [-1, 1]:
    -1 = full left, +1 = full right, proportional in between.
    /car/throttle (std_msgs/Float32) - throttle command in [-1, 1]:
    >0 accelerates, <0 brakes, 0 coasts down under constant friction.
    Two independent axes, both continuous - any controller (manual, PID,
    RL) drives the car the same way, just by publishing to both.
    """

    def __init__(self):
        super().__init__('sim_node')
        self._declare_params()
        self._load_params()

        self.bridge = CvBridge()
        self.finish_distance_m = FINISH_DISTANCE_M
        self.road = RoadGenerator(
            seed=self.road_seed, width=self.road_width,
            waypoint_spacing=self.waypoint_spacing,
            segment_length=self.segment_length,
            max_curvature=self.max_curvature,
            curvature_rate=self.curvature_rate,
            lookahead=self.lookahead, keep_behind=self.keep_behind,
            finish_distance=self.finish_distance_m,
            straight_start=self.straight_start_m,
            pole_spacing=self.pole_spacing, pole_side_offset=self.pole_side_offset,
            lane_dash_length=self.lane_dash_length, lane_dash_gap=self.lane_dash_gap)
        self.car = CarState(
            wheelbase=self.wheelbase, max_steer_angle=math.radians(self.max_steer_angle_deg),
            max_lat_accel=self.max_lat_accel, max_accel=self.max_accel, max_decel=self.max_decel,
            friction_decel=self.friction_decel, max_speed=self.max_speed)
        # Peak achievable |omega| given the wheelbase/steering/grip limits -
        # used only as the HUD gauge's display range (omega itself still
        # comes straight from CarState each tick, this is just a sensible
        # full-scale for the bar). Occurs where the raw bicycle-model omega
        # equals the grip-limited omega: v* = sqrt(max_lat_accel*L/tan(delta_max)).
        delta_max = math.radians(self.max_steer_angle_deg)
        self.omega_display_range = math.sqrt(
            self.max_lat_accel * math.tan(delta_max) / self.wheelbase)
        self.camera = PinholeCamera(
            self.cam_width, self.cam_height, self.cam_hfov_deg,
            self.cam_mount_height, self.cam_pitch_deg)
        self.player_camera = PinholeCamera(
            self.player_width_px, self.player_height_px, self.player_hfov_deg,
            self.player_cam_height, self.player_pitch_deg)
        # Static (camera never moves relative to the car) - built once,
        # republished each tick with a fresh header stamp.
        self.camera_info_msg = self._build_camera_info(self.camera, 'car_camera')
        self.extrinsics_msg = self._build_extrinsics(self.camera, 'car_base_link')

        self.steering = 0.0  # [-1, 1]
        self.throttle = 0.0  # [-1, 1]
        self.elapsed_s = 0.0
        self.game_over = False
        self.victory = False
        self._frozen_frames = None

        self.camera_pub = self.create_publisher(Image, '/car/camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/car/camera/camera_info', 10)
        self.extrinsics_pub = self.create_publisher(PoseStamped, '/car/camera/extrinsics', 10)
        self.player_view_pub = self.create_publisher(Image, '/car/player_view/image_raw', 10)
        self.chase_pub = self.create_publisher(Image, '/car/chase/image_raw', 10)
        self.game_over_pub = self.create_publisher(Bool, '/car/game_over', 10)

        self.create_subscription(Float32, '/car/steering', self._on_steering, 10)
        self.create_subscription(Float32, '/car/throttle', self._on_throttle, 10)
        self.create_service(Trigger, '/car/reset', self._on_reset)

        self.dt = 1.0 / self.tick_rate
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f'car_sim ready (seed={self.road_seed}, '
            f'difficulty={DIFFICULTY} -> finish={self.finish_distance_m:.0f}m)')

    def _declare_params(self):
        p = self.declare_parameter
        p('wheelbase', 2.7)         # meters, front-to-rear axle distance
        p('max_steer_angle_deg', 35.0)  # degrees, front wheel limit
        p('max_lat_accel', 14.71)  # m/s^2, tire grip limit (~1.5 g), caps omega at speed
        p('max_accel', 5.0)        # m/s^2, up-arrow acceleration
        p('max_decel', 10.0)       # m/s^2, down-arrow braking (stronger than accel)
        p('friction_decel', 2.0)   # m/s^2, passive coast-down with no throttle input
        p('max_speed', 50.0)       # m/s
        p('road_seed', 42)
        p('road_width', 16.2)
        p('waypoint_spacing', 0.5)
        p('segment_length', 40.0)
        p('max_curvature', 0.02)
        p('curvature_rate', 0.01)
        p('lookahead', 150.0)
        p('keep_behind', 65.0)
        p('straight_start_m', 30.0)  # road stays straight for this many meters at the start
        p('pole_spacing', 20.0)      # meters between roadside poles (landmarks); <=0 disables
        p('pole_side_offset', 1.5)   # meters beyond the road edge
        p('lane_dash_length', 1.0)   # meters, centerline dash "on" length
        p('lane_dash_gap', 1.0)      # meters, centerline dash "off" length
        p('pole_height', 3.5)        # meters
        p('tick_rate', 20.0)
        p('camera_width', 640)
        p('camera_height', 360)
        p('camera_hfov_deg', 120.0)
        p('camera_mount_height', 1.5)
        p('camera_pitch_deg', 12.0)
        p('camera_render_distance', 50.0)
        p('player_width_px', 800)
        p('player_height_px', 600)
        p('player_hfov_deg', 90.0)
        p('player_pitch_deg', 15.0)
        p('player_cam_height', 3.5)     # meters above ground
        p('player_back_offset', 8.0)    # meters behind the car
        p('player_render_distance', 60.0)
        p('chase_width_px', 800)
        p('chase_height_px', 600)
        p('chase_scale', 8.0)
        p('chase_car_x_frac', 0.5)   # car's horizontal position, fraction of width
        p('chase_car_y_frac', 0.667)  # car's vertical position, fraction of height from top

    def _load_params(self):
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.wheelbase = g('wheelbase')
        self.max_steer_angle_deg = g('max_steer_angle_deg')
        self.max_lat_accel = g('max_lat_accel')
        self.max_accel = g('max_accel')
        self.max_decel = g('max_decel')
        self.friction_decel = g('friction_decel')
        self.max_speed = g('max_speed')
        self.road_seed = g('road_seed')
        self.road_width = g('road_width')
        self.waypoint_spacing = g('waypoint_spacing')
        self.segment_length = g('segment_length')
        self.max_curvature = g('max_curvature')
        self.curvature_rate = g('curvature_rate')
        self.lookahead = g('lookahead')
        self.keep_behind = g('keep_behind')
        self.straight_start_m = g('straight_start_m')
        self.pole_spacing = g('pole_spacing')
        self.pole_side_offset = g('pole_side_offset')
        self.lane_dash_length = g('lane_dash_length')
        self.lane_dash_gap = g('lane_dash_gap')
        self.pole_height = g('pole_height')
        self.tick_rate = g('tick_rate')
        self.cam_width = g('camera_width')
        self.cam_height = g('camera_height')
        self.cam_hfov_deg = g('camera_hfov_deg')
        self.cam_mount_height = g('camera_mount_height')
        self.cam_pitch_deg = g('camera_pitch_deg')
        self.cam_render_distance = g('camera_render_distance')
        self.player_width_px = g('player_width_px')
        self.player_height_px = g('player_height_px')
        self.player_hfov_deg = g('player_hfov_deg')
        self.player_pitch_deg = g('player_pitch_deg')
        self.player_cam_height = g('player_cam_height')
        self.player_back_offset = g('player_back_offset')
        self.player_render_distance = g('player_render_distance')
        self.chase_width_px = g('chase_width_px')
        self.chase_height_px = g('chase_height_px')
        self.chase_scale = g('chase_scale')
        self.chase_car_x_frac = g('chase_car_x_frac')
        self.chase_car_y_frac = g('chase_car_y_frac')

    def _on_steering(self, msg):
        self.steering = max(-1.0, min(1.0, msg.data))

    def _on_throttle(self, msg):
        self.throttle = max(-1.0, min(1.0, msg.data))

    def _on_reset(self, request, response):
        self.road.reset(seed=self.road_seed)
        self.car.reset()
        self.game_over = False
        self.victory = False
        self.steering = 0.0
        self.throttle = 0.0
        self.elapsed_s = 0.0
        self._frozen_frames = None
        response.success = True
        response.message = 'reset'
        self.get_logger().info('Simulation reset')
        return response

    def _tick(self):
        episode_over = self.game_over or self.victory

        if episode_over:
            # Frozen: no kinematics, no road generation, no re-rendering -
            # just republish the single cached end-state frame so late
            # subscribers still see current state.
            if self._frozen_frames is None:
                self._frozen_frames = self._render_frames()
            cam_img, player_img, chase_img = self._frozen_frames
        else:
            self.car.step(self.steering, self.throttle, self.dt)
            self.elapsed_s += self.dt
            self.road.update(self.car.distance_traveled)
            offset = self.road.lateral_offset(self.car.x, self.car.y)
            if abs(offset) > self.road.width / 2.0:
                self.game_over = True
                self.get_logger().warn('Car went off road - game over')
            elif self.car.distance_traveled >= self.finish_distance_m:
                self.victory = True
                self.get_logger().info('Finish line reached - victory!')

            cam_img, player_img, chase_img = self._render_frames()
            if self.game_over or self.victory:
                self._frozen_frames = (cam_img, player_img, chase_img)

        stamp = self.get_clock().now().to_msg()
        self._publish_image(self.camera_pub, cam_img, stamp, 'car_camera')
        self.camera_info_msg.header.stamp = stamp
        self.camera_info_pub.publish(self.camera_info_msg)
        self.extrinsics_msg.header.stamp = stamp
        self.extrinsics_pub.publish(self.extrinsics_msg)
        self._publish_image(self.player_view_pub, player_img, stamp, 'car_player_view')
        self._publish_image(self.chase_pub, chase_img, stamp, 'car_chase')

        go_msg = Bool()
        go_msg.data = self.game_over or self.victory
        self.game_over_pub.publish(go_msg)

    def _render_frames(self):
        cam_img = render.render_camera(
            self.car, self.road, self.camera,
            self.cam_width, self.cam_height, self.cam_render_distance,
            finish_s=self.finish_distance_m, pole_height=self.pole_height)

        player_img = render.render_player_view(
            self.car, self.road, self.player_camera,
            self.player_width_px, self.player_height_px, self.player_render_distance,
            self.player_back_offset, self.player_cam_height,
            finish_s=self.finish_distance_m, pole_height=self.pole_height)
        player_img = render.draw_telemetry_hud(
            player_img,
            throttle=self.throttle, accel=self.car.accel,
            max_accel=self.max_accel, max_decel=self.max_decel,
            speed=self.car.speed, max_speed=self.max_speed,
            steering=self.steering, omega=self.car.omega, max_omega=self.omega_display_range,
            yaw=self.car.yaw,
            distance_traveled=self.car.distance_traveled,
            target_distance=self.finish_distance_m, elapsed_s=self.elapsed_s)

        chase_img = render.render_chase(
            self.car, self.road, self.chase_width_px, self.chase_height_px,
            self.chase_scale,
            anchor_x_frac=self.chase_car_x_frac, anchor_y_frac=self.chase_car_y_frac,
            finish_s=self.finish_distance_m)
        chase_img = render.draw_distance_box(chase_img, self.car.distance_traveled)

        if self.game_over:
            cam_img = render.draw_end_state_overlay(cam_img, 'GAME OVER', (0, 0, 255))
            player_img = render.draw_end_state_overlay(player_img, 'GAME OVER', (0, 0, 255))
            chase_img = render.draw_end_state_overlay(chase_img, 'GAME OVER', (0, 0, 255))
        elif self.victory:
            cam_img = render.draw_end_state_overlay(cam_img, 'VICTORY!', (0, 200, 0))
            player_img = render.draw_end_state_overlay(player_img, 'VICTORY!', (0, 200, 0))
            chase_img = render.draw_end_state_overlay(chase_img, 'VICTORY!', (0, 200, 0))

        return cam_img, player_img, chase_img

    def _publish_image(self, pub, img_bgr, stamp, frame_id):
        msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        pub.publish(msg)

    def _build_camera_info(self, camera, frame_id):
        """Ideal-pinhole CameraInfo (zero distortion) - same intrinsics
        render.py actually uses, so a perception node reprojecting pixels
        with these numbers gets exact, not approximate, results."""
        msg = CameraInfo()
        msg.header.frame_id = frame_id
        msg.width = camera.width_px
        msg.height = camera.height_px
        msg.distortion_model = 'plumb_bob'
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.k = [camera.fx, 0.0, camera.cx,
                 0.0, camera.fy, camera.cy,
                 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [camera.fx, 0.0, camera.cx, 0.0,
                 0.0, camera.fy, camera.cy, 0.0,
                 0.0, 0.0, 1.0, 0.0]
        return msg

    def _build_extrinsics(self, camera, frame_id):
        """Camera pose relative to the car's own body frame (see
        PinholeCamera.body_frame_pose - x-forward, y-left, z-up, origin at
        the car). Fixed for the whole run since the camera is rigidly
        mounted; only the header stamp changes tick to tick."""
        position, R_body_from_cam = camera.body_frame_pose()
        qx, qy, qz, qw = quaternion_from_matrix(R_body_from_cam)
        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
