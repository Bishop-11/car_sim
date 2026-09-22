import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Bool
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

from car_sim.road import RoadGenerator
from car_sim.kinematics import CarState
from car_sim.camera_model import PinholeCamera
from car_sim import renderer

# Difficulty is a fixed, code-level setting (not a launch parameter yet):
# difficulty N means the finish line is N km down the road.
DIFFICULTY = 1
FINISH_DISTANCE_M = DIFFICULTY * 1000.0


class SimNode(Node):
    """Owns car kinematics + the procedural road, and renders both views:

    /car/camera/image_raw  - front pinhole camera (for lane detection)
    /car/chase/image_raw   - third-person chase cam (the game view: used as
                              manual play's player window and available for
                              visualization in automatic-control mode too)
    /car/game_over         - latched off-road/victory flag
    /car/reset (service)   - respawn car + regenerate road
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
            straight_start=self.straight_start_m)
        self.car = CarState()
        self.camera = PinholeCamera(
            self.cam_width, self.cam_height, self.cam_hfov_deg,
            self.cam_mount_height, self.cam_pitch_deg)

        self.control = 'straight'
        self.game_over = False
        self.victory = False
        self._frozen_frames = None

        self.camera_pub = self.create_publisher(Image, '/car/camera/image_raw', 10)
        self.chase_pub = self.create_publisher(Image, '/car/chase/image_raw', 10)
        self.game_over_pub = self.create_publisher(Bool, '/car/game_over', 10)

        self.create_subscription(String, '/car/control', self._on_control, 10)
        self.create_service(Trigger, '/car/reset', self._on_reset)

        self.dt = 1.0 / self.tick_rate
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f'car_sim ready (seed={self.road_seed}, '
            f'difficulty={DIFFICULTY} -> finish={self.finish_distance_m:.0f}m)')

    def _declare_params(self):
        p = self.declare_parameter
        p('car_speed', 7.5)
        p('yaw_rate', 1.0)
        p('road_seed', 42)
        p('road_width', 16.2)
        p('waypoint_spacing', 0.5)
        p('segment_length', 40.0)
        p('max_curvature', 0.02)
        p('curvature_rate', 0.01)
        p('lookahead', 150.0)
        p('keep_behind', 65.0)
        p('straight_start_m', 30.0)  # road stays straight for this many meters at the start
        p('tick_rate', 20.0)
        p('camera_width', 640)
        p('camera_height', 360)
        p('camera_hfov_deg', 120.0)
        p('camera_mount_height', 1.5)
        p('camera_pitch_deg', 12.0)
        p('camera_render_distance', 50.0)
        p('chase_width_px', 800)
        p('chase_height_px', 600)
        p('chase_scale', 8.0)
        p('chase_car_x_frac', 0.5)   # car's horizontal position, fraction of width
        p('chase_car_y_frac', 0.667)  # car's vertical position, fraction of height from top

    def _load_params(self):
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.car_speed = g('car_speed')
        self.yaw_rate = g('yaw_rate')
        self.road_seed = g('road_seed')
        self.road_width = g('road_width')
        self.waypoint_spacing = g('waypoint_spacing')
        self.segment_length = g('segment_length')
        self.max_curvature = g('max_curvature')
        self.curvature_rate = g('curvature_rate')
        self.lookahead = g('lookahead')
        self.keep_behind = g('keep_behind')
        self.straight_start_m = g('straight_start_m')
        self.tick_rate = g('tick_rate')
        self.cam_width = g('camera_width')
        self.cam_height = g('camera_height')
        self.cam_hfov_deg = g('camera_hfov_deg')
        self.cam_mount_height = g('camera_mount_height')
        self.cam_pitch_deg = g('camera_pitch_deg')
        self.cam_render_distance = g('camera_render_distance')
        self.chase_width_px = g('chase_width_px')
        self.chase_height_px = g('chase_height_px')
        self.chase_scale = g('chase_scale')
        self.chase_car_x_frac = g('chase_car_x_frac')
        self.chase_car_y_frac = g('chase_car_y_frac')

    def _on_control(self, msg):
        if msg.data in ('left', 'right', 'straight'):
            self.control = msg.data

    def _on_reset(self, request, response):
        self.road.reset(seed=self.road_seed)
        self.car.reset()
        self.game_over = False
        self.victory = False
        self.control = 'straight'
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
            cam_img, chase_img = self._frozen_frames
        else:
            self.car.step(self.control, self.car_speed, self.yaw_rate, self.dt)
            self.road.update(self.car.distance_traveled)
            offset = self.road.lateral_offset(self.car.x, self.car.y)
            if abs(offset) > self.road.width / 2.0:
                self.game_over = True
                self.get_logger().warn('Car went off road - game over')
            elif self.car.distance_traveled >= self.finish_distance_m:
                self.victory = True
                self.get_logger().info('Finish line reached - victory!')

            cam_img, chase_img = self._render_frames()
            if self.game_over or self.victory:
                self._frozen_frames = (cam_img, chase_img)

        stamp = self.get_clock().now().to_msg()
        self._publish_image(self.camera_pub, cam_img, stamp, 'car_camera')
        self._publish_image(self.chase_pub, chase_img, stamp, 'car_chase')

        go_msg = Bool()
        go_msg.data = self.game_over or self.victory
        self.game_over_pub.publish(go_msg)

    def _render_frames(self):
        cam_img = renderer.render_camera(
            self.car, self.road, self.camera,
            self.cam_width, self.cam_height, self.cam_render_distance,
            finish_s=self.finish_distance_m)

        chase_img = renderer.render_chase(
            self.car, self.road, self.chase_width_px, self.chase_height_px,
            self.chase_scale,
            anchor_x_frac=self.chase_car_x_frac, anchor_y_frac=self.chase_car_y_frac,
            finish_s=self.finish_distance_m)
        chase_img = renderer.draw_distance_box(chase_img, self.car.distance_traveled)

        if self.game_over:
            cam_img = renderer.draw_end_state_overlay(cam_img, 'GAME OVER', (0, 0, 255))
            chase_img = renderer.draw_end_state_overlay(chase_img, 'GAME OVER', (0, 0, 255))
        elif self.victory:
            cam_img = renderer.draw_end_state_overlay(cam_img, 'VICTORY!', (0, 200, 0))
            chase_img = renderer.draw_end_state_overlay(chase_img, 'VICTORY!', (0, 200, 0))

        return cam_img, chase_img

    def _publish_image(self, pub, img_bgr, stamp, frame_id):
        msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding='bgr8')
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        pub.publish(msg)


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
