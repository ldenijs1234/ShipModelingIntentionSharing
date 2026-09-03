#!/usr/bin/env python3
import sys
from pathlib import Path

# Connect to the external core Python library
parent_repo = Path('/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs')
if str(parent_repo) not in sys.path:
    sys.path.insert(0, str(parent_repo))

import rclpy
from rclpy.node import Node
import numpy as np

# Message definitions
from maritime_interfaces.msg import RouteIntent, VesselKinematics
from std_msgs.msg import Float64MultiArray

from gnc_core.simulation.pipeline import SynchronousPipeline

class OSTransceiverNode(Node):
    def __init__(self):
        super().__init__('os_transceiver_node')

        self.dt = 0.1  # 10 Hz
        self.u_nominal = 0.5

        # 1. IMAZU CASE 01 CONFIGURATION (Directly loaded)
        self.internal_state = np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]) 
        self.w_mission_os = np.array([[0.0, 0.0], [40.0, 0.0]])

        # 2. Network Subscriptions
        self.route_sub = self.create_subscription(
            RouteIntent, '/ts/route_delayed', self.ts_route_callback, 10
        )
        self.state_sub = self.create_subscription(
            VesselKinematics, '/ts/state_vector', self.ts_state_callback, 10
        )

        # 3. Output Publishers
        self.os_state_pub = self.create_publisher(Float64MultiArray, '/os/state_vector', 10)
        self.telemetry_pub = self.create_publisher(Float64MultiArray, '/os/telemetry', 10)

        # 4. Latent Data & State Caching
        self.x_ts_raw = np.zeros(6) 
        self.w_ts_delayed = None
        self.cached = {
            "time": 0.0,
            "dcpa": float("inf"),
            "tcpa": 0.0,
            "w_active": self.w_mission_os.copy(),
            "psi_ca": 0.0,
            "state": "STAND_ON",
            "psi_wp": 0.0,
            "wp_idx": 0
        }

        # 5. Start the synchronous pipeline loop
        self.timer = self.create_timer(self.dt, self.step_gnc_pipeline)
        self.get_logger().info("OS Transceiver initialized. 10 Hz Pipeline running.")

    def ts_route_callback(self, msg: RouteIntent):
        self.w_ts_delayed = np.array([[pt.x, pt.y] for pt in msg.route])

    def ts_state_callback(self, msg: VesselKinematics):
        self.x_ts_raw = np.array([msg.x, msg.y, msg.psi, msg.u, msg.v, msg.r])

    def step_gnc_pipeline(self):
        # NOTE: We call .step() directly on the class, no instantiation!
        self.internal_state, self.cached, telemetry = SynchronousPipeline.step(
            internal_state=self.internal_state,
            w_mission_os=self.w_mission_os,
            x_ts_raw=self.x_ts_raw,
            w_ts_delayed=self.w_ts_delayed,
            cached=self.cached,
            dt=self.dt,
            u_nominal=self.u_nominal
        )

        # Publish Own Ship State
        os_msg = Float64MultiArray()
        os_msg.data = self.internal_state.tolist()
        self.os_state_pub.publish(os_msg)

        # Publish Telemetry
        telem_msg = Float64MultiArray()
        telem_msg.data = [
            telemetry["time"], telemetry["dcpa"], telemetry["tcpa"],
            telemetry["u_c"], telemetry["tau_c"], telemetry["psi_cmd"]
        ]
        self.telemetry_pub.publish(telem_msg)

def main(args=None):
    rclpy.init(args=args)
    node = OSTransceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()