#!/usr/bin/env python3
import sys
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

from std_msgs.msg import Float64MultiArray
from maritime_interfaces.msg import RouteIntent, VesselKinematics

from gnc_core.config.vessel_params import VesselParams
from gnc_core.imazu_cases.scenario_loader import load_scenario
from gnc_core.simulation.pipeline import SynchronousPipeline
from gnc_core.navigation.state_estimation import StateEstimation


class OSTransceiverNode(Node):
    def __init__(self):
        super().__init__('own_ship_node')

        self.declare_parameter('scenario', 'case01')
        self.declare_parameter('speed_factor', 1.0)
        self.declare_parameter('auto_close', False)

        scenario_name = self.get_parameter('scenario').value
        self.speed_factor = max(float(self.get_parameter('speed_factor').value), 0.1)
        self.auto_close = bool(self.get_parameter('auto_close').value)

        config = load_scenario(scenario_name)

        self.dt = 0.05  # 20 Hz simulation integration step
        self.sim_time = 0.0
        self.u_nominal = float(config['os_nominal_speed'])
        self.w_mission_os = config['os_mission_wps'].copy()
        self.w_mission_ts_nominal = config['ts_mission_wps'].copy()
        self.canal_polygons = config.get('canal_polygons', None)

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

        self.w_ts_delayed: Optional[np.ndarray] = None
        self.t_intent_shared: Optional[float] = None
        self.sim_finished = False

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
        self.hist_ts_pos = []
        self.hist_r = []

        # ROS Publishers and Subscribers
        self.os_state_pub = self.create_publisher(Float64MultiArray, '/os/state_vector', 10)
        self.w_active_pub = self.create_publisher(Float64MultiArray, '/os/active_waypoints', 10)
        self.telemetry_pub = self.create_publisher(Float64MultiArray, '/os/telemetry', 10)

        self.ts_state_sub = self.create_subscription(
            VesselKinematics, '/ts/state_vector', self.ts_state_callback, 10
        )
        self.ts_route_sub = self.create_subscription(
            RouteIntent, '/ts/route_delayed', self.ts_route_callback, 10
        )

        dt_timer = self.dt / self.speed_factor
        self.timer = self.create_timer(dt_timer, self.step_gnc_pipeline)

        self.get_logger().info(
            f"OS Transceiver initialized for [{scenario_name}] at {self.speed_factor}x speed."
        )

    def ts_state_callback(self, msg: VesselKinematics):
        if abs(msg.x) < 1e-4 and abs(msg.y) < 1e-4:
            return

        fresh_state = np.array([msg.x, msg.y, msg.psi, msg.u, msg.v, msg.r], dtype=np.float64)
        self.x_ts_raw = fresh_state
        self.x_ts_est = fresh_state.copy()

    def ts_route_callback(self, msg: RouteIntent):
        self.w_ts_delayed = np.array([[pt.x, pt.y] for pt in msg.route], dtype=np.float64)
        if self.t_intent_shared is None:
            self.t_intent_shared = self.sim_time

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
            psi_ts = self.x_ts_est[2]
            u_ts = self.x_ts_est[3]
            r_ts = self.x_ts_est[5]

            if self.w_ts_delayed is not None and len(self.w_ts_delayed) > 0:
                ts_goal = self.w_ts_delayed[-1]
            else:
                ts_goal = self.w_mission_ts_nominal[-1, :2]
            
            # Only integrate position forward if TS has not arrived at terminal waypoint
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
            # Safe dummy target outside detection range until first TS packet arrives
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

        # 5. Stop Condition
        if self.x_ts_est is not None:
            os_at_goal = dist_to_final <= 0.5
            os_stopped = abs(self.internal_state[5]) < 0.05 and os_at_goal

            # Use the actual scenario TS destination, not a hardcoded point
            if self.w_ts_delayed is not None and len(self.w_ts_delayed) > 0:
                ts_goal = self.w_ts_delayed[-1]
            else:
                ts_goal = self.w_mission_ts_nominal[-1, :2]

            dist_ts_to_goal = float(np.linalg.norm(self.x_ts_est[:2] - ts_goal[:2]))

            ts_at_goal = dist_ts_to_goal <= 0.6
            ts_stopped = (abs(self.x_ts_est[3]) < 0.08) and ts_at_goal

            if os_stopped and ts_stopped and not self.sim_finished:
                self.sim_finished = True
                self.get_logger().info("\033[92mBoth vessels arrived at terminal waypoints.\033[0m")
                self.plot_encounter_metrics()

                if self.timer:
                    self.timer.cancel()

    def plot_encounter_metrics(self):
        t_arr = np.array(self.hist_time)
        if len(t_arr) < 2:
            return

        range_arr = np.array(self.hist_range)
        dcpa_arr = np.array(self.hist_dcpa)
        tcpa_arr = np.array(self.hist_tcpa)
        r_arr = np.degrees(np.array(self.hist_r))

        fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        fig.canvas.manager.set_window_title("COLREGs Encounter & Performance Metrics")

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

        plt.tight_layout()
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