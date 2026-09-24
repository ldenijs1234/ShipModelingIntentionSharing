#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time

class SimulationClockNode(Node):
    def __init__(self):
        super().__init__('sim_clock_node')
        self.declare_parameter('speed_factor', 1.0)
        self.declare_parameter('step_size', 0.05)  # 20 Hz simulation clock resolution

        self.speed_factor = float(self.get_parameter('speed_factor').value)
        self.step_size = float(self.get_parameter('step_size').value)
        
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.sim_time = 0.0

        # Pacing timer: wall interval scaled by desired speed factor
        wall_timer_period = max(1e-4, self.step_size / self.speed_factor)
        self.timer = self.create_timer(wall_timer_period, self.tick)
        self.get_logger().info(f"Sim Clock running: step={self.step_size*1000:.1f}ms, speed={self.speed_factor}x")

    def tick(self):
        self.sim_time += self.step_size
        
        sec = int(self.sim_time)
        nanosec = int((self.sim_time - sec) * 1e9)
        
        msg = Clock()
        msg.clock = Time(sec=sec, nanosec=nanosec)
        self.clock_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimulationClockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()