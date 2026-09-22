import numpy as np
import pygame
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class ManualControlNode(Node):
    """Displays the chase-cam player view and publishes /car/control from arrow keys.

    Only run during manual play; not used when an automatic controller drives.
    """

    def __init__(self):
        super().__init__('manual_control_node')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('window_width', 500)
        self.declare_parameter('window_height', 500)

        self.publish_rate = self.get_parameter('publish_rate').value
        self.window_width = self.get_parameter('window_width').value
        self.window_height = self.get_parameter('window_height').value

        self.bridge = CvBridge()
        self.latest_frame = None

        self.control_pub = self.create_publisher(String, '/car/control', 10)
        self.create_subscription(Image, '/car/chase/image_raw', self._on_frame, 10)

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

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            control = 'left'
        elif keys[pygame.K_RIGHT]:
            control = 'right'
        else:
            control = 'straight'

        msg = String()
        msg.data = control
        self.control_pub.publish(msg)
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
