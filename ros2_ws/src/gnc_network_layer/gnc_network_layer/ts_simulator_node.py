#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point
from maritime_interfaces.msg import RouteIntent, VesselKinematics


class TSSimulatorNode(Node):
    def __init__(self):
        super().__init__('ts_simulator_node')
        
        # Parameter to toggle sharing the route intention
        self.declare_parameter('share_intent', True)

        self.x = 25.0
        self.y = 0.0
        self.psi = np.pi
        self.u = 0.5
        self.r = 0.0
        self.dt = 0.02

        self.state_pub = self.create_publisher(VesselKinematics, '/ts/state_vector', 10)
        self.route_pub = self.create_publisher(RouteIntent, '/ts/route_true', 10)

        self.state_timer = self.create_timer(self.dt, self.step_kinematics)
        self.route_timer = self.create_timer(1.0, self.publish_route)
        
        mode = "WITH intent" if self.get_parameter('share_intent').value else "WITHOUT intent"
        self.get_logger().info(f'TS Simulator initialized ({mode}).')

    def step_kinematics(self):
        self.x += self.u * np.cos(self.psi) * self.dt
        self.y += self.u * np.sin(self.psi) * self.dt

        msg = VesselKinematics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vessel_mmsi = 244000002
        msg.x = float(self.x)
        msg.y = float(self.y)
        msg.psi = float(self.psi)
        msg.u = float(self.u)
        msg.v = 0.0
        msg.r = float(self.r)
        self.state_pub.publish(msg)

    def publish_route(self):
        share_intent = self.get_parameter('share_intent').value
        
        route_msg = RouteIntent()
        route_msg.header.stamp = self.get_clock().now().to_msg()
        route_msg.vessel_mmsi = 244000002
        route_msg.planned_speed = 0.5
        
        if share_intent:
            wps = [[25.0, 0.0], [12.5, 0.0], [0.0, 0.0]]
            for wp in wps:
                p = Point()
                p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.0
                route_msg.route.append(p)
        else:
            # Broadcast empty route if intent sharing is disabled
            route_msg.route = []

        self.route_pub.publish(route_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TSSimulatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()