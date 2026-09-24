#!/usr/bin/env python3
import sys
from pathlib import Path
from collections import deque

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
from gnc_core.tests.kpi_evaluator import resolve_effective_interval


class TSHorizonSlicer:
    def __init__(self, full_route_xy: np.ndarray, time_scale: float = 5.4772):
        self.route = full_route_xy
        self.time_scale = time_scale
        # 5 minutes full-scale converted to simulation time (~54.77 s)
        self.t_horizon_sim = 300.0 / self.time_scale

        # Precompute cumulative segment distances along the full route
        dists = np.hypot(np.diff(self.route[:, 0]), np.diff(self.route[:, 1]))
        self.s_cumulative = np.insert(np.cumsum(dists), 0, 0.0)

    def extract_5min_intent(
        self, current_x: float, current_y: float, speed_u: float
    ) -> list[tuple[float, float]]:
        """Extracts all dynamic intention waypoints covering 5 min into the future."""
        if len(self.route) < 2 or speed_u <= 0.01:
            return [(float(current_x), float(current_y))]

        # 1. Project current position onto the route to find true continuous path distance (s_now)
        min_dist = float('inf')
        s_now = 0.0

        for i in range(len(self.route) - 1):
            p0 = self.route[i]
            p1 = self.route[i + 1]

            seg_vec = p1 - p0
            seg_len_sq = np.dot(seg_vec, seg_vec)

            if seg_len_sq < 1e-6:
                continue

            pt_vec = np.array([current_x, current_y]) - p0
            t = np.dot(pt_vec, seg_vec) / seg_len_sq
            t_clamped = np.clip(t, 0.0, 1.0)

            proj_pt = p0 + t_clamped * seg_vec
            dist = np.linalg.norm(np.array([current_x, current_y]) - proj_pt)

            if dist < min_dist:
                min_dist = dist
                s_now = self.s_cumulative[i] + (t_clamped * np.sqrt(seg_len_sq))

        # 2. Compute the 5-minute path distance horizon
        dist_horizon = speed_u * self.t_horizon_sim
        s_end = min(s_now + dist_horizon, self.s_cumulative[-1])

        intent_wps = []

        # 3. The intent MUST start exactly at the vessel's current coordinate
        intent_wps.append((float(current_x), float(current_y)))

        # 4. Gather pre-planned structural waypoints strictly inside the forward window
        in_window_indices = np.where(
            (self.s_cumulative > s_now + 1e-3) & (self.s_cumulative < s_end - 1e-3)
        )[0]

        for idx in in_window_indices:
            intent_wps.append((float(self.route[idx, 0]), float(self.route[idx, 1])))

        # 5. Interpolate the exact 5-min boundary point (if we haven't exhausted the route)
        if s_end > s_now + 1e-3:
            seg_idx = np.searchsorted(self.s_cumulative, s_end) - 1
            seg_idx = np.clip(seg_idx, 0, len(self.route) - 2)

            s0 = self.s_cumulative[seg_idx]
            s1 = self.s_cumulative[seg_idx + 1]
            seg_len = s1 - s0

            ratio = (s_end - s0) / seg_len if seg_len > 1e-5 else 0.0
            p0 = self.route[seg_idx]
            p1 = self.route[seg_idx + 1]

            x_end = p0[0] + ratio * (p1[0] - p0[0])
            y_end = p0[1] + ratio * (p1[1] - p0[1])

            intent_wps.append((float(x_end), float(y_end)))

        return intent_wps


class TSSimulatorNode(Node):
    def __init__(self):
        super().__init__('ts_simulator_node')

        self.declare_parameter('scenario', 'case01')
        self.declare_parameter('share_intent', True)
        self.declare_parameter('route_interval', 3.0)
        self.declare_parameter('speed_factor', 1.0)

        scenario_name = self.get_parameter('scenario').value
        config = load_scenario(scenario_name)

        self.share_intent = bool(self.get_parameter('share_intent').value)
        raw_route_interval = float(self.get_parameter('route_interval').value)

        self.dt = 0.05  # 20 Hz simulation integration step
        self.u_nominal = float(config['ts_nominal_speed'])
        self.w_mission_ts = config['ts_mission_wps'].copy()
        self.wp_idx = 1

        # Internal dynamics state [x, y, psi, r, b, u]
        raw_state = config['ts_initial_state']
        self.internal_state = np.array([
            raw_state[0],  # X
            raw_state[1],  # Y
            raw_state[2],  # psi
            raw_state[5],  # r (yaw rate)
            0.0,           # b (heading bias)
            raw_state[3]   # u (surge velocity)
        ], dtype=np.float64)

        # Initialize the dynamic horizon slicer
        self.ts_slicer = TSHorizonSlicer(self.w_mission_ts)

        # Determine realistic initial waypoint count for the 5-minute window
        initial_intent = self.ts_slicer.extract_5min_intent(
            current_x=self.internal_state[0],
            current_y=self.internal_state[1],
            speed_u=self.u_nominal
        )

        # --- Enforce ITU-R M.1371 Regulatory Clamp ---
        num_waypoints = len(initial_intent)
        self.route_interval, self.min_itu_interval = resolve_effective_interval(
            input_interval=raw_route_interval,
            num_waypoints=num_waypoints
        )

        if self.route_interval > raw_route_interval:
            self.get_logger().warn(
                f"\033[93m[TS Comms] Requested interval ({raw_route_interval:.1f}s) violates "
                f"ITU 20 slots/min limit for {num_waypoints} WPs! "
                f"Clamped to minimum compliant: {self.route_interval:.1f}s\033[0m"
            )

        self.os_state = None
        self.intent_broadcasted = False
        self.effective_t_advance = None
        self.sim_finished = False

        # ITU-R M.1371 Table 1 dynamic AIS interval tracking
        self.sim_time = 0.0
        self.last_ais_tx_time = 0.0
        self.last_route_tx_time = 0.0
        self.heading_history = deque(maxlen=int(30.0 / self.dt))

        # ROS Publishers and Subscribers
        self.state_pub = self.create_publisher(VesselKinematics, '/ts/state_vector', 10)
        self.route_pub = self.create_publisher(RouteIntent, '/ts/route_true', 10)

        self.os_sub = self.create_subscription(
            Float64MultiArray, '/os/state_vector', self.os_state_callback, 10
        )

        self.sim_timer = self.create_timer(self.dt, self.step_gnc_pipeline)

        self.publish_current_state()
        self.publish_route()

        mode = (
            f"WITH intent (Effective Interval = {self.route_interval:.1f}s, ITU min = {self.min_itu_interval:.1f}s)"
            if self.share_intent
            else "WITHOUT intent"
        )
        self.get_logger().info(
            f"TS Simulator running for [{scenario_name}] ({mode}) with sim_time."
        )

    def os_state_callback(self, msg: Float64MultiArray):
        self.os_state = np.array(msg.data, dtype=np.float64)

    def get_itu_reporting_interval(self) -> float:
        current_psi = self.internal_state[2]
        self.heading_history.append(current_psi)

        if len(self.heading_history) > 1:
            mean_psi = float(np.arctan2(
                np.mean(np.sin(self.heading_history)),
                np.mean(np.cos(self.heading_history))
            ))
            delta_course = abs((current_psi - mean_psi + np.pi) % (2.0 * np.pi) - np.pi)
            is_changing_course = delta_course > np.radians(5.0)
        else:
            is_changing_course = False

        sog_knots = float(self.internal_state[5]) * 1.94384

        if sog_knots <= 14.0:
            return 10.0 / 3.0 if is_changing_course else 10.0  # 3 1/3 s vs 10 s
        elif 14.0 < sog_knots <= 23.0:
            return 2.0 if is_changing_course else 6.0
        else:
            return 2.0

    def calculate_tcpa(self) -> float:
        if self.os_state is None:
            return float('inf')

        p_ts = self.internal_state[0:2]
        p_os = self.os_state[0:2]
        r_pos = p_os - p_ts

        u_ts = self.internal_state[5]
        v_ts = np.array([
            u_ts * np.cos(self.internal_state[2]),
            u_ts * np.sin(self.internal_state[2])
        ])

        u_os = self.os_state[3]
        v_os = np.array([
            u_os * np.cos(self.os_state[2]),
            u_os * np.sin(self.os_state[2])
        ])
        v_rel = v_os - v_ts

        v_rel_sq = np.dot(v_rel, v_rel)
        if v_rel_sq < 1e-6:
            return float('inf')

        tcpa = -np.dot(r_pos, v_rel) / v_rel_sq
        return float(tcpa)

    def publish_current_state(self):
        if self.sim_finished:
            return

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

    def step_gnc_pipeline(self):
        # 1. Guidance & Control
        dist_to_final = float(np.linalg.norm(self.internal_state[0:2] - self.w_mission_ts[-1, 0:2]))

        if self.sim_finished:
            # Vessel is docked: clamp velocities to zero
            self.internal_state[3] = 0.0  # r
            self.internal_state[5] = 0.0  # u
        else:
            x_sensor = np.array([
                self.internal_state[0],
                self.internal_state[1],
                self.internal_state[2],
                self.internal_state[5],  # u
                0.0,                     # v
                self.internal_state[3]   # r
            ], dtype=np.float64)

            max_wp_idx = max(len(self.w_mission_ts) - 1, 1)
            self.wp_idx = min(self.wp_idx, max_wp_idx)

            psi_los, _, next_wp_idx = LOSGuidance.compute_heading_reference(
                x_os=x_sensor,
                w_active=self.w_mission_ts,
                current_wp_idx=self.wp_idx
            )
            self.wp_idx = min(next_wp_idx, max_wp_idx)

            # Slow down cleanly near target
            if dist_to_final <= 0.4:
                u_target = 0.0
            elif dist_to_final <= 2.0:
                u_target = self.u_nominal * (dist_to_final / 2.0)
            else:
                u_target = self.u_nominal

            u_c, tau_c, _ = Autopilot.compute_control(
                x_os=x_sensor,
                psi_wp=psi_los,
                psi_ca_reactive=0.0,
                u_nominal=u_target
            )

            inputs = np.array([tau_c, u_c], dtype=np.float64)
            self.internal_state = VesselDynamics.rk4(
                x=self.internal_state,
                inputs=inputs,
                dt=self.dt
            )

            self.sim_time += self.dt

            # Check if TS has arrived and stopped
            if dist_to_final <= 0.4 and abs(self.internal_state[5]) < 0.05:
                self.sim_finished = True
                self.internal_state[5] = 0.0
                self.internal_state[3] = 0.0
                self.publish_current_state()
                self.get_logger().info("\033[94m[TS] Reached terminal waypoint and stopped.\033[0m")

        # 2. Discrete AIS dynamic broadcast (continue transmitting resting state)
        req_interval = self.get_itu_reporting_interval()
        if (self.sim_time - self.last_ais_tx_time) >= req_interval:
            self.publish_current_state()
            self.last_ais_tx_time = self.sim_time

        # 3. Discrete Route Intent broadcast tied to sim_time
        if self.share_intent and not self.sim_finished:
            if (self.sim_time - self.last_route_tx_time) >= (self.route_interval - 1e-5):
                self.publish_route()
                self.last_route_tx_time = self.sim_time

    def publish_route(self):
        if not self.share_intent or self.sim_finished:
            return

        tcpa = self.calculate_tcpa()
        if not self.intent_broadcasted:
            self.effective_t_advance = tcpa
            self.intent_broadcasted = True

        route_msg = RouteIntent()
        route_msg.header.stamp = self.get_clock().now().to_msg()
        route_msg.vessel_mmsi = 244000002
        route_msg.planned_speed = float(self.u_nominal)

        intent_wps = self.ts_slicer.extract_5min_intent(
            current_x=self.internal_state[0],
            current_y=self.internal_state[1],
            speed_u=self.internal_state[5]
        )

        for wp in intent_wps:
            p = Point()
            p.x, p.y, p.z = float(wp[0]), float(wp[1]), 0.0
            route_msg.route.append(p)

        self.route_pub.publish(route_msg)


def main(args=None):
    rclpy.init(args=args)
    node = TSSimulatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()