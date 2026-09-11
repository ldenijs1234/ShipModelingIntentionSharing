import rclpy
from rclpy.node import Node
from collections import deque
from maritime_interfaces.msg import RouteIntent


class LatencyBridgeNode(Node):
    def __init__(self):
        super().__init__('latency_bridge_node')
        self.declare_parameter('latency', 2.0)
        
        self.sub = self.create_subscription(
            RouteIntent,
            '/ts/route_true',
            self.route_callback,
            10
        )
        self.pub = self.create_publisher(RouteIntent, '/ts/route_delayed', 10)
        self.queue = deque()
        self.timer = self.create_timer(0.02, self.check_queue)
        self.get_logger().info('Latency bridge initialized.')

    def route_callback(self, msg: RouteIntent):
        current_time = self.get_clock().now().nanoseconds / 1e9
        self.queue.append((current_time, msg))

    def check_queue(self):
        latency = self.get_parameter('latency').get_parameter_value().double_value
        current_time = self.get_clock().now().nanoseconds / 1e9

        while self.queue and (current_time - self.queue[0][0]) >= latency:
            _, msg = self.queue.popleft()
            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LatencyBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
