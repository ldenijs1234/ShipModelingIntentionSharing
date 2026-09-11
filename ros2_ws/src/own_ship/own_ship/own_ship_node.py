#!/usr/bin/env python3
import sys
from pathlib import Path
from gnc_core.imazu_cases.scenario_loader import load_scenario

parent_repo = Path('/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs')
if str(parent_repo) not in sys.path:
    sys.path.insert(0, str(parent_repo))

import rclpy
from rclpy.node import Node
import numpy as np

from maritime_interfaces.msg import RouteIntent, VesselKinematics
from std_msgs.msg import Float64MultiArray

from gnc_core.simulation.pipeline import SynchronousPipeline

class OSTransceiverNode(Node):
    def __init__(self):
        super().__init__('os_transceiver_node')

        self.declare_parameter('scenario', 'case01')
        scenario_name = self.get_parameter('scenario').value
        config = load_scenario(scenario_name)

        self.dt = 0.1
        self.u_nominal = float(config['os_nominal_speed'])

        self.w_mission_os = config['os_mission_wps'].copy()
        self.internal_state = config['os_initial_state'].copy()

        self.route_sub = self.create_subscription(
            RouteIntent, '/ts/route_delayed', self.ts_route_callback, 10
        )
        self.state_sub = self.create_subscription(
            VesselKinematics, '/ts/state_vector', self.ts_state_callback, 10
        )

        self.os_state_pub = self.create_publisher(Float64MultiArray, '/os/state_vector', 10)
        self.telemetry_pub = self.create_publisher(Float64MultiArray, '/os/telemetry', 10)
        self.w_active_pub = self.create_publisher(Float64MultiArray, '/os/active_waypoints', 10)

        self.x_ts_raw = None
        self.w_ts_delayed = None
        self.cached = {
            "time": 0.0,
            "dcpa": float("inf"),
            "tcpa": float("inf"),
            "w_active": self.w_mission_os.copy(),
            "psi_ca": 0.0,
            "state": "STAND_ON",
            "psi_wp": 0.0,
            "wp_idx": 0
        }

        self.timer = self.create_timer(self.dt, self.step_gnc_pipeline)
        self.get_logger().info(f"OS Transceiver initialized for [{scenario_name}]. 10 Hz Pipeline running.")

    def ts_route_callback(self, msg: RouteIntent):
        self.w_ts_delayed = np.array([[pt.x, pt.y] for pt in msg.route])
        #self.get_logger().info(f"[OS] >>> Received TS Route Intent with {len(self.w_ts_delayed)} waypoints!")

    def ts_state_callback(self, msg: VesselKinematics):
        self.x_ts_raw = np.array([msg.x, msg.y, msg.psi, msg.u, msg.v, msg.r])

    def step_gnc_pipeline(self):
        # Do not run risk evaluation until Target Ship state has been received
        if self.x_ts_raw is None:
            return

        self.internal_state, self.cached, telemetry = SynchronousPipeline.step(
            internal_state=self.internal_state,
            w_mission_os=self.w_mission_os,
            x_ts_raw=self.x_ts_raw,
            w_ts_delayed=self.w_ts_delayed,
            cached=self.cached,
            dt=self.dt,
            u_nominal=self.u_nominal
        )

        # if telemetry.get("active_state") in ["State A.1", "State B.1"]:
        #     self.get_logger().info(
        #         f"[GNC Loop] State: {telemetry['active_state']} | DCPA: {telemetry['dcpa']:.2f} | TCPA: {telemetry['tcpa']:.2f} | psi_ca: {np.degrees(self.cached['psi_ca']):.1f} deg"
        #     )

        # Publish Own Ship State
        os_msg = Float64MultiArray()
        os_msg.data = self.internal_state.tolist()
        self.os_state_pub.publish(os_msg)

        # Publish Active Waypoints for Plotter
        w_msg = Float64MultiArray()
        w_msg.data = self.cached["w_active"].flatten().tolist()
        self.w_active_pub.publish(w_msg)

        # Publish Telemetry
        telem_msg = Float64MultiArray()
        telem_msg.data = [
            telemetry["time"], telemetry["dcpa"], telemetry["tcpa"],
            telemetry["u_c"], telemetry["tau_c"], telemetry["psi_cmd"]
        ]
        self.telemetry_pub.publish(telem_msg)

        if telemetry.get("comp_time_ms") is not None:
            self.get_logger().info(
                f"\033[92m[DECISION ENGINE] Mode A.1 waypoints generated in: {telemetry['comp_time_ms']:.4f} ms\033[0m"
            )

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