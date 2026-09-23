import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge


def _ramp_axis(value, positive_held, negative_held, ramp_rate, center_rate, dt):
    """One tick of a ramped control axis in [-1, 1]. Holding the positive/
    negative key ramps the value toward +1/-1 at `ramp_rate` units/s;
    holding neither (or both) self-centers it back toward 0 at
    `center_rate` units/s. Same trick for steering and throttle - it's what
    turns two digital keys into a continuous, proportional command."""
    if positive_held and not negative_held:
        value += ramp_rate * dt
    elif negative_held and not positive_held:
        value -= ramp_rate * dt
    elif value > 0.0:
        value = max(0.0, value - center_rate * dt)
    elif value < 0.0:
        value = min(0.0, value + center_rate * dt)
    return max(-1.0, min(1.0, value))


class ManualControlNode(Node):
    """Displays the pulled-back third-person player view and publishes
    continuous steering/throttle commands from arrow keys.

    Keyboards are digital, so proportional control ("how much", not just
    direction) comes from ramping: holding a key ramps that axis's value
    toward full deflection; releasing it self-centers back toward 0. A
    quick tap gives a small input, holding builds up to full - the analog
    feel of a real controller out of four digital keys. Left/right drive
    steering (/car/steering), up/down drive throttle (/car/throttle) -
    two independent axes, published every tick.

    Only run during manual play; not used when an automatic controller drives.
    """

    def __init__(self):
        super().__init__('manual_control_node')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('window_width', 800)
        self.declare_parameter('window_height', 600)
        self.declare_parameter('steer_ramp_rate', 2.5)      # units/s toward full lock
        self.declare_parameter('steer_center_rate', 4.0)    # units/s toward center when released
        self.declare_parameter('throttle_ramp_rate', 2.5)   # units/s toward full throttle/brake
        self.declare_parameter('throttle_center_rate', 4.0)  # units/s toward center when released

        self.publish_rate = self.get_parameter('publish_rate').value
        self.window_width = self.get_parameter('window_width').value
        self.window_height = self.get_parameter('window_height').value
        self.steer_ramp_rate = self.get_parameter('steer_ramp_rate').value
        self.steer_center_rate = self.get_parameter('steer_center_rate').value
        self.throttle_ramp_rate = self.get_parameter('throttle_ramp_rate').value
        self.throttle_center_rate = self.get_parameter('throttle_center_rate').value

        self.bridge = CvBridge()
        self.latest_frame = None
        self.steering = 0.0  # [-1, 1]: -1 = full left, +1 = full right
        self.throttle = 0.0  # [-1, 1]: +1 = full accel, -1 = full brake

        self.steering_pub = self.create_publisher(Float32, '/car/steering', 10)
        self.throttle_pub = self.create_publisher(Float32, '/car/throttle', 10)
        self.create_subscription(Image, '/car/player_view/image_raw', self._on_frame, 10)

        pygame.init()
        pygame.display.set_caption('Lane Drive - Manual Control')
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.clock = pygame.time.Clock()

        self.create_timer(1.0 / self.publish_rate, self._tick)

    def _on_frame(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_frame = cv_img[:, :, ::-1]  # BGR -> RGB for pygame

    def _tick(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.get_logger().info('Window closed, shutting down manual control')
                rclpy.shutdown()
                return

        if self.latest_frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(self.latest_frame, (1, 0, 2)))
            if surf.get_size() != (self.window_width, self.window_height):
                surf = pygame.transform.scale(surf, (self.window_width, self.window_height))
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()

        dt = 1.0 / self.publish_rate
        keys = pygame.key.get_pressed()

        self.steering = _ramp_axis(
            self.steering, keys[pygame.K_RIGHT], keys[pygame.K_LEFT],
            self.steer_ramp_rate, self.steer_center_rate, dt)
        self.throttle = _ramp_axis(
            self.throttle, keys[pygame.K_UP], keys[pygame.K_DOWN],
            self.throttle_ramp_rate, self.throttle_center_rate, dt)

        self.steering_pub.publish(Float32(data=self.steering))
        self.throttle_pub.publish(Float32(data=self.throttle))
        self.clock.tick(self.publish_rate)


def main(args=None):
    rclpy.init(args=args)
    node = ManualControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        pygame.quit()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
