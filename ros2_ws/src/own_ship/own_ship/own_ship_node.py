#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from gnc_core.imazu_cases.scenario_loader import load_scenario

parent_repo = Path('/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs')
if str(parent_repo) not in sys.path:
    sys.path.insert(0, str(parent_repo))

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt

from maritime_interfaces.msg import RouteIntent, VesselKinematics
from std_msgs.msg import Float64MultiArray
from gnc_core.config.vessel_params import VesselParams
from gnc_core.simulation.pipeline import SynchronousPipeline
from gnc_core.guidance.decision import DecisionLayer


class OSTransceiverNode(Node):
    def __init__(self):
        super().__init__('own_ship_node')

        self.declare_parameter('scenario', 'case01')
        self.declare_parameter('speed_factor', 1.0)

        scenario_name = self.get_parameter('scenario').value
        config = load_scenario(scenario_name)

        self.speed_factor = max(float(self.get_parameter('speed_factor').value), 0.1)

        self.canal_polygons = config.get('canal_polygons', None)

        # Synchronized to 20 Hz (0.05 s) to match TS physics resolution
        self.dt = 0.05
        self.u_nominal = float(config['os_nominal_speed'])
        self.w_mission_os = config['os_mission_wps'].copy()
        
        # Map config [x, y, psi, u, v, r] to internal dynamics state [x, y, psi, r, b, u] (Thesis Section 3.1.1)
        raw_state = config['os_initial_state']
        self.internal_state = np.array([
            raw_state[0],  # X
            raw_state[1],  # Y
            raw_state[2],  # psi
            raw_state[5],  # r (yaw rate)
            0.0,           # b (heading bias)
            raw_state[3]   # u (surge velocity)
        ], dtype=np.float64)

        # Initialize previous heading memory for true yaw rate calculation
        self.prev_psi = float(self.internal_state[2])

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
        self.x_ts_est = None
        self.last_ts_packet_time = None
        self.w_ts_delayed = None
        self.t_intent_shared = None
        self.sim_finished = False

        # Telemetry history buffers for post-run evaluation
        self.hist_time = []
        self.hist_dcpa = []
        self.hist_tcpa = []
        self.hist_range = []
        self.hist_os_pos = []
        self.hist_ts_pos = []
        self.hist_r = []

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

        # Scaled timer: physics integrates by self.dt (0.05s), timer triggers faster
        dt_timer = self.dt / self.speed_factor
        self.timer = self.create_timer(dt_timer, self.step_gnc_pipeline)
        self.get_logger().info(
            f"OS Transceiver initialized for [{scenario_name}] at {self.speed_factor}x speed (20 Hz pipeline)."
        )

        self.declare_parameter('auto_close', False)
        ac_param = self.get_parameter('auto_close').value
        self.auto_close = (
            (ac_param.upper() == 'TRUE')
            if isinstance(ac_param, str)
            else bool(ac_param)
        )

    def ts_route_callback(self, msg: RouteIntent):
        self.w_ts_delayed = np.array([[pt.x, pt.y] for pt in msg.route])
        if self.t_intent_shared is None:
            self.t_intent_shared = self.cached.get("time", 0.0)
            self.get_logger().info(f"[OS] Intent received at t = {self.t_intent_shared:.2f} s")

    def ts_state_callback(self, msg: VesselKinematics):
        # Filter uninitialized coordinates
        if abs(msg.x) < 1e-4 and abs(msg.y) < 1e-4:
            return

        # Raw state vector: [X, Y, psi, u, v, r]
        fresh_state = np.array([msg.x, msg.y, msg.psi, msg.u, msg.v, msg.r], dtype=np.float64)
        self.x_ts_raw = fresh_state

        # Re-anchor the dead-reckoning model to the newly arrived ground truth
        self.x_ts_est = fresh_state.copy()
        self.last_ts_packet_time = self.get_clock().now()

    def step_gnc_pipeline(self):
        if self.x_ts_raw is None or self.sim_finished:
            return

        # Guard against zero or uninitialized position from TS
        if abs(self.x_ts_raw[0]) < 1e-4 and abs(self.x_ts_raw[1]) < 1e-4:
            return

        # Initialize self.x_ts_est if this is the first tick
        if getattr(self, "x_ts_est", None) is None:
            self.x_ts_est = self.x_ts_raw.copy()

        # =============================================================
        # PASTE HERE: 20 Hz Kinematic Interpolation / Dead-Reckoning
        # =============================================================
        psi_ts = self.x_ts_est[2]
        u_ts = self.x_ts_est[3]
        r_ts = self.x_ts_est[5]

        # Integrate yaw rate and position forward by self.dt
        psi_ts_next = (psi_ts + r_ts * self.dt + np.pi) % (2.0 * np.pi) - np.pi
        self.x_ts_est[2] = psi_ts_next
        self.x_ts_est[0] += u_ts * np.cos(psi_ts_next) * self.dt
        self.x_ts_est[1] += u_ts * np.sin(psi_ts_next) * self.dt
        # =============================================================

        # 1. Check if OS has reached the terminal waypoint
        target_wp = self.cached["w_active"][-1, :2]
        dist_to_final = np.linalg.norm(self.internal_state[:2] - target_wp)
        current_u_target = 0.0 if dist_to_final <= VesselParams.D_m else self.u_nominal

        # Capture previous state before the pipeline runs
        prev_state = self.cached["state"]

        # 2. Step synchronous pipeline with the interpolated target state (self.x_ts_est)
        self.internal_state, self.cached, telemetry = SynchronousPipeline.step(
            internal_state=self.internal_state,
            w_mission_os=self.w_mission_os,
            x_ts_raw=self.x_ts_est,  # <-- PASS self.x_ts_est INSTEAD OF self.x_ts_raw
            w_ts_delayed=self.w_ts_delayed,
            cached=self.cached,
            dt=self.dt,
            u_nominal=current_u_target,
            canal_polygons=self.canal_polygons,
        )

        # Log state transition cleanly
        if self.cached["state"] != prev_state:
            self.get_logger().info(f"[OS] Transitioned to {self.cached['state']}")

        # 3. Compute instantaneous Range using interpolated position
        current_range = float(
            np.linalg.norm(self.internal_state[:2] - self.x_ts_est[:2])
        )

        # 4. Log data for analysis
        sim_time = telemetry["time"]
        self.hist_time.append(sim_time)
        self.hist_dcpa.append(telemetry["dcpa"])
        self.hist_tcpa.append(telemetry["tcpa"])
        self.hist_range.append(current_range)
        self.hist_os_pos.append(self.internal_state[:2].copy())
        self.hist_ts_pos.append(self.x_ts_est[:2].copy())  # <-- Log smooth coordinates
        self.hist_r.append(float(self.internal_state[3]))

        # 5. Publish ROS topics
        os_msg = Float64MultiArray(data=telemetry["x_os"].tolist())
        self.os_state_pub.publish(os_msg)

        w_msg = Float64MultiArray(
            data=self.cached["w_active"].flatten().tolist()
        )
        self.w_active_pub.publish(w_msg)

        telem_msg = Float64MultiArray(
            data=[
                sim_time,
                telemetry["dcpa"],
                telemetry["tcpa"],
                telemetry["u_c"],
                telemetry["tau_c"],
                telemetry["psi_cmd"],
            ]
        )
        self.telemetry_pub.publish(telem_msg)

        # 6. Stop condition: internal_state[5] is surge speed u (index 3 is yaw rate r)
        os_stopped = (
            abs(self.internal_state[5]) < 0.05
            and dist_to_final <= (VesselParams.D_m + 0.2)
        )
        ts_stopped = abs(self.x_ts_est[3]) < 0.05

        if os_stopped and ts_stopped:
            self.sim_finished = True
            self.get_logger().info(
                "\033[92mBoth vessels arrived at terminal waypoints.\033[0m"
            )
            self.plot_encounter_metrics()

    
    def plot_encounter_metrics(self):
        t_arr = np.array(self.hist_time)
        if len(t_arr) < 2:
            return

        range_arr = np.array(self.hist_range)
        dcpa_arr = np.array(self.hist_dcpa)
        tcpa_arr = np.array(self.hist_tcpa)

        os_y = np.array([pt[1] for pt in self.hist_os_pos])
        e_cte = np.abs(os_y)
        j_cte_accum = np.cumsum(e_cte) * self.dt

        r_arr = np.array(self.hist_r)
        r_dot = np.gradient(r_arr, self.dt)
        j_ctrl_accum = np.cumsum(r_dot**2) * self.dt

        fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
        fig.canvas.manager.set_window_title("COLREGs Encounter & Performance Metrics")

        # 1. Instantaneous Separation Range
        axes[0].plot(t_arr, range_arr, color="#1f77b4", linewidth=1.8, label="Range (m)")
        axes[0].axhline(VesselParams.DCPA_safe, color="red", linestyle="--", label=f"Safe ({VesselParams.DCPA_safe} m)")
        axes[0].set_ylabel("Range [m]")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right", fontsize=8)

        # 2. DCPA
        axes[1].plot(t_arr, dcpa_arr, color="#2ca02c", linewidth=1.8, label="DCPA (m)")
        axes[1].axhline(VesselParams.DCPA_safe, color="red", linestyle="--", label="DCPA Safe")
        axes[1].set_ylabel("DCPA [m]")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right", fontsize=8)

        # 3. TCPA
        axes[2].plot(t_arr, tcpa_arr, color="#ff7f0e", linewidth=1.8, label="TCPA (s)")
        axes[2].axhline(0.0, color="gray", linestyle=":", label="TCPA = 0")
        axes[2].set_ylabel("TCPA [s]")
        axes[2].grid(True, linestyle=":", alpha=0.6)
        axes[2].legend(loc="upper right", fontsize=8)

        # 4. Instantaneous Yaw Rate
        axes[3].plot(t_arr, np.degrees(r_arr), color="#9467bd", linewidth=1.6, label=r"Yaw Rate $r$ [deg/s]")
        axes[3].set_ylabel(r"$r$ [deg/s]")
        axes[3].grid(True, linestyle=":", alpha=0.6)
        axes[3].legend(loc="upper right", fontsize=8)

        # 5. Cross-Track Error & Cumulative Trajectory Penalty
        axes[4].plot(t_arr, e_cte, color="#d62728", linewidth=1.6, label=r"Instantaneous $|e_{\mathrm{cte}}|$ [m]")
        axes[4].plot(t_arr, j_cte_accum, color="#8c564b", linestyle="--", linewidth=1.4, label=r"Cumulative $J_{\mathrm{cte}}$ [m$\cdot$s]")
        axes[4].set_ylabel(r"$e_{\mathrm{cte}}$ [m]")
        axes[4].set_xlabel("Simulation Time [s]")
        axes[4].grid(True, linestyle=":", alpha=0.6)
        axes[4].legend(loc="upper right", fontsize=8)

        plt.tight_layout()

        if self.auto_close:
            os.makedirs("results_plots", exist_ok=True)
            scenario_name = self.get_parameter("scenario").value
            plt.savefig(f"results_plots/{scenario_name}_encounter_kpi.png", dpi=300)
            plt.close(fig)
        else:
            plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = OSTransceiverNode()
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