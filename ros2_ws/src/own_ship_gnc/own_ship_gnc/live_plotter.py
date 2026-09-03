#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from maritime_interfaces.msg import VesselKinematics, RouteIntent
import matplotlib.pyplot as plt
import numpy as np
import threading
import math

class LivePlotterNode(Node):
    def __init__(self):
        super().__init__('live_plotter')
        
        # Current states
        self.os_state = [0.0, 0.0, 0.0]
        self.ts_state = [30.0, 0.0, math.pi]
        
        # Route intention arrays (dynamic)
        self.ts_route_x = []
        self.ts_route_y = []

        # History arrays for the sailed route imprint
        self.os_history_x = []
        self.os_history_y = []
        self.ts_history_x = []
        self.ts_history_y = []
        
        # Subscriptions
        self.create_subscription(Float64MultiArray, '/os/state_vector', self.os_callback, 10)
        self.create_subscription(VesselKinematics, '/ts/state_vector', self.ts_callback, 10)
        self.create_subscription(RouteIntent, '/ts/route_true', self.ts_route_callback, 10)
        
        self.get_logger().info("Live Plotter listening to OS and TS states...")

    def os_callback(self, msg):
        # Reset history if the ship "teleports" (indicates a simulation restart)
        if self.os_history_x:
            dist = math.hypot(msg.data[0] - self.os_history_x[-1], msg.data[1] - self.os_history_y[-1])
            if dist > 5.0:  # Huge jump = restart
                self.os_history_x.clear()
                self.os_history_y.clear()

        self.os_state = [msg.data[0], msg.data[1], msg.data[2]]
        self.os_history_x.append(msg.data[0])
        self.os_history_y.append(msg.data[1])

    def ts_callback(self, msg):
        # Reset history if the ship "teleports"
        if self.ts_history_x:
            dist = math.hypot(msg.x - self.ts_history_x[-1], msg.y - self.ts_history_y[-1])
            if dist > 5.0:
                self.ts_history_x.clear()
                self.ts_history_y.clear()

        self.ts_state = [msg.x, msg.y, msg.psi]
        self.ts_history_x.append(msg.x)
        self.ts_history_y.append(msg.y)

    def ts_route_callback(self, msg):
        # Dynamically update the intention line based on network payload
        self.ts_route_x = [pt.x for pt in msg.route]
        self.ts_route_y = [pt.y for pt in msg.route]

def draw_ship(ax, x, y, psi, color, label):
    length = 2.0
    width = 1.0
    
    p1 = np.array([0, length])
    p2 = np.array([-width, -length])
    p3 = np.array([width, -length])
    
    rot = np.array([
        [np.cos(-psi), -np.sin(-psi)],
        [np.sin(-psi),  np.cos(-psi)]
    ])
    
    p1 = rot.dot(p1) + np.array([y, x])
    p2 = rot.dot(p2) + np.array([y, x])
    p3 = rot.dot(p3) + np.array([y, x])
    
    triangle = plt.Polygon([p1, p2, p3], color=color, label=label)
    ax.add_patch(triangle)

def main(args=None):
    rclpy.init(args=args)
    node = LivePlotterNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title('Imazu Case 01: Head-On')

    try:
        while rclpy.ok():
            ax.clear()
            
            ax.set_xlim(-20, 20)
            ax.set_ylim(-10, 50)
            ax.set_xlabel("East (y) [m]")
            ax.set_ylabel("North (x) [m]")
            ax.set_title("Live GNC Simulation")
            ax.grid(True, linestyle='--', alpha=0.6)

            # 1. Draw OS Mission Intention (Remains static as OS always has a plan)
            ax.plot([0, 0], [0, 40], 'b--', alpha=0.3, label='OS Mission')
            
            # 2. Draw TS Route Intent ONLY if the array is populated (share_intent=True)
            if node.ts_route_x and node.ts_route_y:
                ax.plot(node.ts_route_y, node.ts_route_x, 'r--', alpha=0.3, label='TS Route Intent')

            # 3. Draw Sailed Route History (Solid lines)
            ax.plot(list(node.os_history_y), list(node.os_history_x), 'b-', linewidth=2, alpha=0.7, label='OS Sailed')
            ax.plot(list(node.ts_history_y), list(node.ts_history_x), 'r-', linewidth=2, alpha=0.7, label='TS Sailed')

            # 4. Draw Live Ship Positions
            draw_ship(ax, node.os_state[0], node.os_state[1], node.os_state[2], 'blue', 'Own Ship')
            draw_ship(ax, node.ts_state[0], node.ts_state[1], node.ts_state[2], 'red', 'Target Ship')

            ax.legend(loc='upper right')
            plt.pause(0.1) 

    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        plt.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()