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
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
from maritime_interfaces.msg import RouteIntent, VesselKinematics

from gnc_core.config.vessel_params import VesselParams
from gnc_core.imazu_cases.scenario_loader import load_scenario
from gnc_core.guidance.los import LOSGuidance
from gnc_core.control.autopilot import Autopilot
from gnc_core.models.vessel_dynamics import VesselDynamics


class TSSimulatorNode(Node):
    def __init__(self):
        super().__init__('ts_simulator_node')

        # 1. ROS Parameters
        self.declare_parameter('scenario', 'case01')
        self.declare_parameter('share_intent', True)
        self.declare_parameter('t_advance', 15.0)
        self.declare_parameter('speed_factor', 1.0)

        scenario_name = self.get_parameter('scenario').value
        config = load_scenario(scenario_name)

        self.share_intent = bool(self.get_parameter('share_intent').value)
        self.t_advance = float(self.get_parameter('t_advance').value)
        self.speed_factor = max(float(self.get_parameter('speed_factor').value), 0.1)

        self.dt = 0.05  # 20 Hz integration step
        self.u_nominal = float(config['ts_nominal_speed'])
        self.w_mission_ts = config['ts_mission_wps'].copy()
        self.wp_idx = 1
        self.internal_state = config['ts_initial_state'].copy()

        # 2. Tracking State
        self.os_state = None
        self.intent_broadcasted = False

        # 3. ROS Publishers and Subscriptions
        self.state_pub = self.create_publisher(VesselKinematics, '/ts/state_vector', 10)
        self.route_pub = self.create_publisher(RouteIntent, '/ts/route_true', 10)
        
        self.os_sub = self.create_subscription(
            Float64MultiArray, '/os/state_vector', self.os_state_callback, 10
        )

        # 4. Timers scaled by speed_factor
        dt_state_timer = self.dt / self.speed_factor
        dt_route_timer = 0.1 / self.speed_factor

        self.state_timer = self.create_timer(dt_state_timer, self.step_gnc_pipeline)
        self.route_timer = self.create_timer(dt_route_timer, self.publish_route)

        mode = f"WITH intent (t_advance = {self.t_advance}s)" if self.share_intent else "WITHOUT intent"
        self.get_logger().info(
            f"TS Simulator running for [{scenario_name}] ({mode}) at {self.speed_factor}x speed."
        )

    def os_state_callback(self, msg: Float64MultiArray):
        # OS state vector: [x, y, psi, r, b, u]
        self.os_state = np.array(msg.data, dtype=np.float64)

    def calculate_tcpa(self) -> float:
        """Calculates current TCPA against Own Ship."""
        if self.os_state is None:
            return float('inf')

        p_ts = self.internal_state[0:2]
        p_os = self.os_state[0:2]
        r_pos = p_os - p_ts

        v_ts = np.array([
            self.internal_state[5] * np.cos(self.internal_state[2]),
            self.internal_state[5] * np.sin(self.internal_state[2])
        ])
        v_os = np.array([
            self.os_state[5] * np.cos(self.os_state[2]),
            self.os_state[5] * np.sin(self.os_state[2])
        ])
        v_rel = v_os - v_ts

        v_rel_sq = np.dot(v_rel, v_rel)
        if v_rel_sq < 1e-6:
            return float('inf')

        # TCPA = - (r_rel . v_rel) / ||v_rel||^2
        tcpa = -np.dot(r_pos, v_rel) / v_rel_sq
        return float(tcpa)

    def step_gnc_pipeline(self):
        # A. Guidance Layer (LOS)
        psi_los, _, self.wp_idx = LOSGuidance.compute_heading_reference(
            x_os=self.internal_state,
            w_active=self.w_mission_ts,
            current_wp_idx=self.wp_idx
        )

        # B. Check distance to final destination and command zero speed on arrival
        dist_to_final = float(np.linalg.norm(self.internal_state[0:2] - self.w_mission_ts[-1, 0:2]))
        u_target = 0.0 if dist_to_final <= VesselParams.D_m else self.u_nominal

        # C. Control Layer (Autopilot)
        x_ctrl = self.internal_state.copy()
        x_ctrl[5] = self.internal_state[3]  # Map r to index 5 for Autopilot

        u_c, tau_c, _ = Autopilot.compute_control(
            x_os=x_ctrl,
            psi_wp=psi_los,
            psi_ca_reactive=0.0,
            u_nominal=u_target
        )

        # D. Vessel Dynamics
        inputs = np.array([tau_c, u_c], dtype=np.float64)
        self.internal_state = VesselDynamics.rk4(
            x=self.internal_state,
            inputs=inputs,
            dt=self.dt
        )

        # E. Publish true TS kinematics
        msg = VesselKinematics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vessel_mmsi = 244000002
        msg.x = float(self.internal_state[0])
        msg.y = float(self.internal_state[1])
        msg.psi = float(self.internal_state[2])
        msg.u = float(self.internal_state[5])
        msg.v = 0.0
        msg.r = float(self.internal_state[3])
        self.state_pub.publish(msg)

    def publish_route(self):
        if not self.share_intent:
            return

        tcpa = self.calculate_tcpa()

        # Broadcast intent once when TCPA drops below t_advance
        if not self.intent_broadcasted and tcpa <= self.t_advance:
            route_msg = RouteIntent()
            route_msg.header.stamp = self.get_clock().now().to_msg()
            route_msg.vessel_mmsi = 244000002
            route_msg.planned_speed = float(self.u_nominal)

            for wp in self.w_mission_ts:
                p = Point()
                p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.0
                route_msg.route.append(p)

            self.route_pub.publish(route_msg)
            self.intent_broadcasted = True
            self.get_logger().info(
                f">>> INTENT BROADCAST: Triggered at TCPA = {tcpa:.2f}s (Threshold: {self.t_advance}s)"
            )
        elif self.intent_broadcasted:
            # Continually publish to ensure active reception
            route_msg = RouteIntent()
            route_msg.header.stamp = self.get_clock().now().to_msg()
            route_msg.vessel_mmsi = 244000002
            route_msg.planned_speed = float(self.u_nominal)
            for wp in self.w_mission_ts:
                p = Point()
                p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.0
                route_msg.route.append(p)
            self.route_pub.publish(route_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TSSimulatorNode()
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