#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from typing import Optional

# Connect to the external core Python library
parent_repo = Path('/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs')
if str(parent_repo) not in sys.path:
    sys.path.insert(0, str(parent_repo))

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

from std_msgs.msg import Float64MultiArray
from maritime_interfaces.msg import RouteIntent, VesselKinematics

from gnc_core.config.vessel_params import VesselParams
from gnc_core.imazu_cases.scenario_loader import load_scenario
from gnc_core.simulation.pipeline import SynchronousPipeline
from gnc_core.navigation.state_estimation import StateEstimation
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


class OSTransceiverNode(Node):
    def __init__(self):
        super().__init__('own_ship_node')

        self.declare_parameter('scenario', 'case01')
        self.declare_parameter('mode', False)  # True = IS, False = RA
        self.declare_parameter('latency', 0.0)
        self.declare_parameter('interval', 5.0)
        self.declare_parameter('speed_factor', 1.0)
        self.declare_parameter('auto_close', False)
        self.declare_parameter('min_intent_range', 10.0)

        # Parse mode
        mode_val = self.get_parameter('mode').value
        self.sim_mode = "IS" if (str(mode_val).upper() == "TRUE" or mode_val is True) else "RA"
        self.sim_latency = float(self.get_parameter('latency').value)
        self.sim_interval = float(self.get_parameter('interval').value)

        scenario_name = self.get_parameter('scenario').value
        self.auto_close = bool(self.get_parameter('auto_close').value)
        self.min_intent_range = float(self.get_parameter('min_intent_range').value)

        config = load_scenario(scenario_name)

        self.dt = 0.05  # 20 Hz simulation integration step
        self.sim_time = 0.0
        self.u_nominal = float(config['os_nominal_speed'])
        self.w_mission_os = config['os_mission_wps'].copy()
        self.w_mission_ts_nominal = config['ts_mission_wps'].copy()
        self.canal_polygons = config.get('canal_polygons', None)
        self.canal_bounds = config.get('canal_bounds', None)

        # Internal state format [x, y, psi, r, b, u]
        raw_state = config['os_initial_state']
        self.internal_state = np.array([
            raw_state[0],  # X
            raw_state[1],  # Y
            raw_state[2],  # psi
            raw_state[5],  # r (yaw rate)
            0.0,           # b (heading bias)
            raw_state[3]   # u (surge velocity)
        ], dtype=np.float64)

        # Target Ship tracking & dead-reckoning state
        self.x_ts_raw: Optional[np.ndarray] = None

        # Pre-seed TS dead-reckoning estimate with scenario initial state to avoid t=0 glitches
        raw_ts_init = config.get('ts_initial_state', None)
        if raw_ts_init is not None:
            # Format: [X, Y, psi, u, v, r]
            self.x_ts_est = np.array([
                raw_ts_init[0], raw_ts_init[1], raw_ts_init[2],
                raw_ts_init[3], 0.0, raw_ts_init[5]
            ], dtype=np.float64)
        else:
            self.x_ts_est = None

        # Setup dynamic horizon slicer to estimate initial target ship waypoint count
        self.ts_slicer = TSHorizonSlicer(self.w_mission_ts_nominal)
        if self.x_ts_est is not None:
            initial_intent = self.ts_slicer.extract_5min_intent(
                current_x=self.x_ts_est[0],
                current_y=self.x_ts_est[1],
                speed_u=self.x_ts_est[3]
            )
            initial_num_waypoints = len(initial_intent)
        else:
            initial_num_waypoints = len(self.w_mission_ts_nominal)

        num_ts_wps = initial_num_waypoints
        raw_interval = float(self.get_parameter("interval").value)
        self.sim_interval, self.min_itu_interval = resolve_effective_interval(
            input_interval=raw_interval, num_waypoints=num_ts_wps
        )

        if self.sim_interval > raw_interval:
            self.get_logger().warn(
                f"\033[93m[OS] Configured interval ({raw_interval:.1f}s) clamped "
                f"to ITU-compliant interval: {self.sim_interval:.1f}s\033[0m"
            )

        self.w_ts_delayed: Optional[np.ndarray] = None
        self.t_intent_shared: Optional[float] = None
        self.sim_finished = False
        self.in_intent_range = False  # Track comms connectivity

        # --- Intent Timing Diagnostics ---
        self.t_first_intent_rx: Optional[float] = None
        self.t_advance: Optional[float] = None

        StateEstimation.reset()

        # Cache structure for SynchronousPipeline
        self.cached = {
            "w_active": self.w_mission_os.copy(),
            "psi_ca": 0.0,
            "p_ca": 1.0,
            "state": "State B.2",
            "wp_idx": 1,
            "dcpa": 999.0,
            "tcpa": 999.0,
            "time": 0.0
        }

        # Telemetry History Logs
        self.hist_time = []
        self.hist_dcpa = []
        self.hist_tcpa = []
        self.hist_range = []
        self.hist_os_pos = []
        self.hist_os_psi = []
        self.hist_os_u = []
        self.hist_ts_pos = []
        self.hist_r = []

        # ROS Publishers and Subscribers
        self.os_state_pub = self.create_publisher(Float64MultiArray, '/os/state_vector', 10)
        self.w_active_pub = self.create_publisher(Float64MultiArray, '/os/active_waypoints', 10)
        self.telemetry_pub = self.create_publisher(Float64MultiArray, '/os/telemetry', 10)

        # Publisher for the TS intent that OS has legitimately received in range
        self.ts_perceived_route_pub = self.create_publisher(RouteIntent, '/os/perceived_ts_route', 10)

        # Static subscription for global AIS kinematics
        self.ts_state_sub = self.create_subscription(
            VesselKinematics, '/ts/state_vector', self.ts_state_callback, 10
        )

        # Static subscription for VHF routing intent (Prevents DDS startup discovery jitter)
        self.ts_route_sub = self.create_subscription(
            RouteIntent, '/ts/route_delayed', self.ts_route_callback, 10
        )

        self.timer = self.create_timer(self.dt, self.step_gnc_pipeline)

        self.get_logger().info(
            f"OS Transceiver initialized for [{scenario_name}] with sim_time."
        )

    def ts_state_callback(self, msg: VesselKinematics):
        if abs(msg.x) < 1e-4 and abs(msg.y) < 1e-4:
            return

        fresh_state = np.array([msg.x, msg.y, msg.psi, msg.u, msg.v, msg.r], dtype=np.float64)
        self.x_ts_raw = fresh_state
        self.x_ts_est = fresh_state.copy()

    def ts_route_callback(self, msg: RouteIntent):
        # Ignore intent broadcast packets when outside communication range
        if not self.in_intent_range:
            return

        self.w_ts_delayed = np.array([[pt.x, pt.y] for pt in msg.route], dtype=np.float64)

        # First-time reception trigger and t_advance calculation
        if self.t_first_intent_rx is None:
            self.t_first_intent_rx = float(self.sim_time)
            self.t_intent_shared = float(self.sim_time)

            if self.x_ts_est is not None and abs(self.x_ts_est[0]) < 900.0:
                p_os = self.internal_state[:2]
                psi_os = self.internal_state[2]
                u_os = self.internal_state[5]
                v_os_vec = np.array([u_os * np.cos(psi_os), u_os * np.sin(psi_os)])

                p_ts = self.x_ts_est[:2]
                psi_ts = self.x_ts_est[2]
                u_ts = self.x_ts_est[3]
                v_ts_vec = np.array([u_ts * np.cos(psi_ts), u_ts * np.sin(psi_ts)])

                dp = p_ts - p_os
                dv = v_ts_vec - v_os_vec
                dv_sq = float(np.dot(dv, dv))

                if dv_sq > 1e-4:
                    tcpa_relative = -float(np.dot(dp, dv)) / dv_sq
                    self.t_advance = tcpa_relative
                else:
                    self.t_advance = np.nan
            else:
                self.t_advance = np.nan

            t_adv_str = f"{self.t_advance:.2f}s" if np.isfinite(self.t_advance) else "N/A"
            self.get_logger().info(
                f"\033[95m[INTENT RX] First packet received at t={self.t_first_intent_rx:.2f}s | "
                f"t_advance (TCPA margin): {t_adv_str}\033[0m"
            )

        # Forward the perceived intent to the live plotter
        self.ts_perceived_route_pub.publish(msg)

    def step_gnc_pipeline(self):
        if self.sim_finished:
            return

        self.sim_time += self.dt

        # Check OS distance to destination
        target_wp = self.cached["w_active"][-1, :2]
        dist_to_final = float(np.linalg.norm(self.internal_state[:2] - target_wp))
        u_target = 0.0 if dist_to_final <= 0.2 else self.u_nominal

        # 1. Kinematic Dead-Reckoning of TS between AIS Broadcasts (20 Hz)
        if self.x_ts_est is not None:
            dist_to_ts = float(np.linalg.norm(self.internal_state[:2] - self.x_ts_est[:2]))

            # --- Dynamic Subscription Range R_IS(t) ---
            u_os = self.internal_state[5]
            psi_os = self.internal_state[2]
            v_os = np.array([u_os * np.cos(psi_os), u_os * np.sin(psi_os)])

            u_ts = self.x_ts_est[3]
            psi_ts = self.x_ts_est[2]
            v_ts = np.array([u_ts * np.cos(psi_ts), u_ts * np.sin(psi_ts)])

            v_rel_norm = float(np.linalg.norm(v_os - v_ts))
            t_tactical_sim = 300.0 / np.sqrt(30.0)  # 5 min full-scale to Froude sim time
            dynamic_r_is = max(self.min_intent_range, v_rel_norm * t_tactical_sim)

            r_connect = max(float(self.min_intent_range), float(dynamic_r_is))
            r_disconnect = max(float(self.min_intent_range) * 1.30, float(dynamic_r_is) * 1.30)

            # Determine whether OS is actively maneuvering
            current_state_str = str(self.cached.get("state", ""))
            is_maneuvering = ("A.1" in current_state_str or "A.2" in current_state_str)
            tcpa_cleared = (self.cached.get("tcpa", 999.0) < 0.0)

            if dist_to_ts <= r_connect:
                if not self.in_intent_range:
                    self.in_intent_range = True
                    self.get_logger().info(
                        f"\033[93m[OS] TS entered intent comms range ({dist_to_ts:.1f}m <= {r_connect:.1f}m). Subscribing to Intent.\033[0m"
                    )

            elif dist_to_ts > r_disconnect and (not is_maneuvering or tcpa_cleared):
                if self.in_intent_range:
                    self.in_intent_range = False
                    self.get_logger().info(
                        f"\033[91m[OS] TS left intent comms range ({dist_to_ts:.1f}m > {r_disconnect:.1f}m). Dropping Intent.\033[0m"
                    )
                    self.w_ts_delayed = None
                    self.t_intent_shared = None

                    empty_msg = RouteIntent()
                    self.ts_perceived_route_pub.publish(empty_msg)

            psi_ts = self.x_ts_est[2]
            u_ts = self.x_ts_est[3]
            r_ts = self.x_ts_est[5]

            if self.w_ts_delayed is not None and len(self.w_ts_delayed) > 0:
                ts_goal = self.w_ts_delayed[-1]
            else:
                ts_goal = self.w_mission_ts_nominal[-1, :2]

            # Integrate dead-reckoning position forward if TS has not arrived
            if np.linalg.norm(self.x_ts_est[:2] - ts_goal[:2]) > 0.4:
                psi_ts_next = (psi_ts + r_ts * self.dt + np.pi) % (2.0 * np.pi) - np.pi
                self.x_ts_est[2] = psi_ts_next
                self.x_ts_est[0] += u_ts * np.cos(psi_ts_next) * self.dt
                self.x_ts_est[1] += u_ts * np.sin(psi_ts_next) * self.dt
            else:
                self.x_ts_est[3] = 0.0
                self.x_ts_est[5] = 0.0

            x_ts_for_gnc = self.x_ts_est
        else:
            x_ts_for_gnc = np.array([999.0, 999.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        prev_state = self.cached["state"]

        # 2. Step Synchronous Pipeline
        self.internal_state, self.cached, telemetry = SynchronousPipeline.step(
            internal_state=self.internal_state,
            w_mission_os=self.w_mission_os,
            x_ts_raw=x_ts_for_gnc,
            w_ts_delayed=self.w_ts_delayed,
            cached=self.cached,
            dt=self.dt,
            u_nominal=u_target,
            canal_polygons=self.canal_polygons,
        )

        if self.cached["state"] != prev_state:
            self.get_logger().info(f"[OS] Transitioned to {self.cached['state']}")

        # 3. Range calculation & Telemetry History Logging
        if self.x_ts_est is not None and abs(self.x_ts_est[0]) < 900.0:
            current_range = float(np.linalg.norm(self.internal_state[:2] - self.x_ts_est[:2]))
            ts_pos = self.x_ts_est[:2].copy()
            dcpa_val = telemetry["dcpa"]
            tcpa_val = telemetry["tcpa"]
        else:
            current_range = np.nan
            ts_pos = np.array([np.nan, np.nan])
            dcpa_val = np.nan
            tcpa_val = np.nan

        self.hist_time.append(self.sim_time)
        self.hist_dcpa.append(dcpa_val)
        self.hist_tcpa.append(tcpa_val)
        self.hist_range.append(current_range)
        self.hist_os_pos.append(self.internal_state[:2].copy())
        self.hist_os_psi.append(float(self.internal_state[2]))
        self.hist_os_u.append(float(self.internal_state[5]))
        self.hist_ts_pos.append(ts_pos)
        self.hist_r.append(float(self.internal_state[3]))

        # 4. Publish Topics (Ensures live_plotter receives coordinates from t = 0)
        os_msg = Float64MultiArray(data=telemetry["x_os"].tolist())
        self.os_state_pub.publish(os_msg)

        w_msg = Float64MultiArray(data=self.cached["w_active"].flatten().tolist())
        self.w_active_pub.publish(w_msg)

        telem_msg = Float64MultiArray(
            data=[
                self.sim_time,
                telemetry["dcpa"],
                telemetry["tcpa"],
                telemetry["u_c"],
                telemetry["tau_c"],
                telemetry["psi_cmd"],
            ]
        )
        self.telemetry_pub.publish(telem_msg)

        # 5. Stop Condition & Completion Check
        final_wp = self.w_mission_os[-1, :2]
        prev_wp = self.w_mission_os[-2, :2] if len(self.w_mission_os) >= 2 else (final_wp - np.array([1.0, 0.0]))

        leg_vec = final_wp - prev_wp
        leg_len = np.linalg.norm(leg_vec)
        leg_unit = leg_vec / max(leg_len, 1e-3)

        dist_to_final = float(np.linalg.norm(self.internal_state[:2] - final_wp))
        dist_past_goal = float(np.dot(self.internal_state[:2] - final_wp, leg_unit))

        os_reached_and_stopped = (dist_to_final <= 0.6) and (abs(self.internal_state[5]) < 0.05)
        os_overshot = dist_past_goal > 1.0
        os_at_goal = os_reached_and_stopped or os_overshot

        if self.w_ts_delayed is not None and len(self.w_ts_delayed) > 0:
            ts_goal = self.w_ts_delayed[-1]
        else:
            ts_goal = self.w_mission_ts_nominal[-1, :2]

        dist_ts_to_goal = float(np.linalg.norm(self.x_ts_est[:2] - ts_goal[:2])) if self.x_ts_est is not None else 0.0
        ts_at_goal = dist_ts_to_goal <= 1.0

        sim_timed_out = getattr(self, "sim_time", 0.0) > 160.0

        if (os_at_goal and ts_at_goal) or sim_timed_out:
            if not self.sim_finished:
                self.sim_finished = True
                if sim_timed_out:
                    self.get_logger().warn("\033[93mRun terminated by failsafe simulation time limit.\033[0m")
                else:
                    self.get_logger().info("\033[92mBoth vessels arrived at terminal waypoints.\033[0m")

                if self.timer:
                    self.timer.cancel()

                self.save_run_log()
                self.plot_encounter_metrics()

                if self.auto_close:
                    self.get_logger().info("[OS] Auto-close active. Triggering clean shutdown.")
                    self.destroy_node()
                    rclpy.shutdown()

    def save_run_log(self):
        if len(self.hist_time) < 2:
            return

        output_dir = os.path.join(str(parent_repo), "simulation_logs")
        os.makedirs(output_dir, exist_ok=True)

        scenario_name = self.get_parameter('scenario').value
        tag = (
            f"{scenario_name}_IS_tau{self.sim_latency:.1f}_dt{self.sim_interval:.1f}"
            if self.sim_mode == "IS"
            else f"{scenario_name}_RA"
        )
        filepath = os.path.join(output_dir, f"{tag}.npz")

        t_arr = np.array(self.hist_time)
        os_pos = np.array(self.hist_os_pos)
        os_psi = np.array(self.hist_os_psi)
        os_u = np.array(self.hist_os_u)
        ts_pos = np.array(self.hist_ts_pos)
        r_arr = np.array(self.hist_r)

        t_adv_val = self.t_advance if self.t_advance is not None else np.nan
        t_rx_val = self.t_first_intent_rx if self.t_first_intent_rx is not None else np.nan

        w_active_val = (
            np.array(self.cached["w_active"])
            if ("w_active" in self.cached and self.cached["w_active"] is not None)
            else np.array([])
        )

        hist_state_arr = np.array(getattr(self, 'hist_state', []))

        np.savez(
            filepath,
            t=t_arr,
            os_pos=os_pos,
            os_psi=os_psi,
            os_r=r_arr,
            os_u=os_u,
            ts_pos=ts_pos,
            nominal_wps=np.array(self.w_mission_os),
            evasive_wps=w_active_val,
            w_ts_delayed=np.array(self.w_ts_delayed) if self.w_ts_delayed is not None else np.array([]),
            hist_state=hist_state_arr,
            scenario=scenario_name,
            canal_polygons=self.canal_polygons,
            canal_bounds=self.canal_bounds,
            mode=self.sim_mode,
            latency=self.sim_latency,
            interval=self.sim_interval,
            t_advance=t_adv_val,
            t_first_intent_rx=t_rx_val,
        )
        self.get_logger().info(
            f"\033[92m[OS] Successfully saved full run log ({len(t_arr)} points, duration: {t_arr[-1]:.1f}s): {filepath}\033[0m"
        )

    def plot_encounter_metrics(self):
        t_arr = np.array(self.hist_time)
        if len(t_arr) < 2:
            return

        range_arr = np.array(self.hist_range)
        dcpa_arr = np.array(self.hist_dcpa)
        tcpa_arr = np.array(self.hist_tcpa)
        r_arr = np.degrees(np.array(self.hist_r))

        os_pos = np.array(self.hist_os_pos)
        ts_pos = np.array(self.hist_ts_pos)
        nom_wps = np.array(self.w_mission_os)
        evasive_wps = (
            np.array(self.cached["w_active"])
            if ("w_active" in self.cached and self.cached["w_active"] is not None)
            else np.array([])
        )

        scenario_name = self.get_parameter('scenario').value
        tag = (
            f"{scenario_name}_IS_tau{self.sim_latency:.1f}_dt{self.sim_interval:.1f}"
            if self.sim_mode == "IS"
            else f"{scenario_name}_RA"
        )
        
        # Target folder: ros2_ws/results_plots
        output_dir = os.path.join(str(parent_repo), "ros2_ws", "results_plots")
        os.makedirs(output_dir, exist_ok=True)

        # ----------------------------------------------------------------------
        # Figure 1: Encounter & Performance Metrics
        # ----------------------------------------------------------------------
        fig1, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig1.canvas.manager.set_window_title("COLREGs Encounter & Performance Metrics")

        # Range
        axs[0].plot(t_arr, range_arr, label="Range (m)", color="#1f77b4", linewidth=1.8)
        axs[0].axhline(y=VesselParams.DCPA_safe, color="r", linestyle="--", label=f"Safe ({VesselParams.DCPA_safe} m)")
        axs[0].set_ylabel("Range [m]")
        axs[0].grid(True, linestyle="--", alpha=0.5)
        axs[0].legend(loc="upper right")

        # DCPA
        axs[1].plot(t_arr, dcpa_arr, label="DCPA (m)", color="#2ca02c", linewidth=1.8)
        axs[1].axhline(y=VesselParams.DCPA_safe, color="r", linestyle="--", label="DCPA Safe")
        axs[1].set_ylabel("DCPA [m]")
        axs[1].grid(True, linestyle="--", alpha=0.5)
        axs[1].legend(loc="upper right")

        # TCPA
        axs[2].plot(t_arr, tcpa_arr, label="TCPA (s)", color="#ff7f0e", linewidth=1.8)
        axs[2].axhline(y=0, color="gray", linestyle=":", label="TCPA = 0")
        axs[2].set_ylabel("TCPA [s]")
        axs[2].grid(True, linestyle="--", alpha=0.5)
        axs[2].legend(loc="upper right")

        # Yaw Rate r
        axs[3].plot(t_arr, r_arr, label="Yaw Rate r [deg/s]", color="#9467bd", linewidth=1.8)
        axs[3].set_ylabel("r [deg/s]")
        axs[3].set_xlabel("Simulation Time [s]")
        axs[3].grid(True, linestyle="--", alpha=0.5)
        axs[3].legend(loc="upper right")

        fig1.tight_layout()
        metrics_png = os.path.join(output_dir, f"{tag}_metrics.png")
        fig1.savefig(metrics_png, dpi=300)
        plt.close(fig1)

        # ----------------------------------------------------------------------
        # Figure 2: 2D Spatial Trajectory & Evasion Route
        # ----------------------------------------------------------------------
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        fig2.canvas.manager.set_window_title("2D Spatial Trajectory Analysis")

        # 1. Canal Banks (Rendered with filled beige polygons matching plot_headless_run)
        from matplotlib.patches import Polygon as MplPolygon
        bank_edge_color = '#8c564b'
        bank_face_color = '#ebdcb9'
        bank_labeled = False

        polys = []
        if self.canal_polygons is not None:
            raw_poly = self.canal_polygons
            if hasattr(raw_poly, 'item'):
                try:
                    raw_poly = raw_poly.item()
                except Exception:
                    pass

            if isinstance(raw_poly, Polygon):
                polys = [raw_poly]
            elif isinstance(raw_poly, MultiPolygon) or hasattr(raw_poly, 'geoms'):
                polys = list(raw_poly.geoms)

        if polys:
            for poly in polys:
                x_pts, y_pts = poly.exterior.xy  # x = North, y = East
                coords = np.column_stack([y_pts, x_pts])  # Plot: [East, North]
                patch = MplPolygon(
                    coords,
                    closed=True,
                    facecolor=bank_face_color,
                    edgecolor=bank_edge_color,
                    linewidth=1.8,
                    alpha=0.65,
                    zorder=1,
                    label="Canal Bank" if not bank_labeled else None
                )
                ax2.add_patch(patch)
                bank_labeled = True
        elif self.canal_bounds is not None:
            y_min = self.canal_bounds.get('y_min', -4.0)
            y_max = self.canal_bounds.get('y_max', 4.0)
            x_min = self.canal_bounds.get('x_min', -10.0)
            x_max = self.canal_bounds.get('x_max', 50.0)
            
            port_coords = np.array([[-15.0, x_min], [y_min, x_min], [y_min, x_max], [-15.0, x_max]])
            ax2.add_patch(MplPolygon(port_coords, closed=True, facecolor=bank_face_color,
                                    edgecolor=bank_edge_color, linewidth=1.8, alpha=0.65, zorder=1, label="Canal Bank"))
            stbd_coords = np.array([[y_max, x_min], [15.0, x_min], [15.0, x_max], [y_max, x_max]])
            ax2.add_patch(MplPolygon(stbd_coords, closed=True, facecolor=bank_face_color,
                                    edgecolor=bank_edge_color, linewidth=1.8, alpha=0.65, zorder=1))

        # 2. Planned routes (East = y, North = x)
        if len(nom_wps) > 0:
            ax2.plot(nom_wps[:, 1], nom_wps[:, 0], 'k--', alpha=0.5, label='Original Mission')
            ax2.scatter(nom_wps[:, 1], nom_wps[:, 0], c='black', s=20, alpha=0.5)

        if evasive_wps.size > 0:
            ax2.plot(evasive_wps[:, 1], evasive_wps[:, 0], color='tab:blue', linestyle='--',
                     marker='o', markersize=4, label='Latched Evasive Route')

        # 3. Actual sailed tracks
        ax2.plot(os_pos[:, 1], os_pos[:, 0], color='tab:blue', linewidth=2.2, label='OS Track')
        ax2.plot(ts_pos[:, 1], ts_pos[:, 0], color='tab:red', linewidth=2.2, label='TS Track')

        # Start and terminal markers
        ax2.scatter(os_pos[0, 1], os_pos[0, 0], color='tab:blue', s=60, marker='o', label='OS Start')
        ax2.scatter(os_pos[-1, 1], os_pos[-1, 0], color='tab:blue', s=80, marker='x', label='OS End')
        ax2.scatter(ts_pos[0, 1], ts_pos[0, 0], color='tab:red', s=60, marker='o', label='TS Start')
        ax2.scatter(ts_pos[-1, 1], ts_pos[-1, 0], color='tab:red', s=80, marker='x', label='TS End')

        # Closest Point of Approach marker
        dists = np.hypot(os_pos[:, 0] - ts_pos[:, 0], os_pos[:, 1] - ts_pos[:, 1])
        min_idx = int(np.argmin(dists))
        ax2.scatter(os_pos[min_idx, 1], os_pos[min_idx, 0], color='darkorange', s=90, marker='*', zorder=5)
        ax2.scatter(ts_pos[min_idx, 1], ts_pos[min_idx, 0], color='darkorange', s=90, marker='*', zorder=5)
        ax2.plot([os_pos[min_idx, 1], ts_pos[min_idx, 1]],
                 [os_pos[min_idx, 0], ts_pos[min_idx, 0]],
                 'k:', linewidth=1.5, label=f'CPA: {dists[min_idx]:.2f} m @ {t_arr[min_idx]:.1f} s')

        title_str = f"Run Trajectory: {scenario_name} ({self.sim_mode})"
        if self.sim_mode == "IS":
            title_str += f" | $\\tau$={self.sim_latency:.1f}s, $\\Delta T$={self.sim_interval:.1f}s"
        ax2.set_title(title_str, fontsize=12, fontweight='bold', pad=10)

        ax2.set_xlabel("East (y) [m]")
        ax2.set_ylabel("North (x) [m]")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.axis("equal")
        ax2.legend(loc="lower left", framealpha=0.9)

        fig2.tight_layout()
        spatial_png = os.path.join(output_dir, f"{tag}_trajectory.png")
        fig2.savefig(spatial_png, dpi=300)
        plt.close(fig2)

        self.get_logger().info(f"[Plotter] Saved figures: {metrics_png} and {spatial_png}")

        if not self.auto_close:
            plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = OSTransceiverNode()
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