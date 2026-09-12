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

        # Synchronized to 20 Hz (0.05 s) to match TS physics resolution
        self.dt = 0.05
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
        self.t_intent_shared = None
        self.sim_finished = False

        # Telemetry history buffers for post-run evaluation
        self.hist_time = []
        self.hist_dcpa = []
        self.hist_tcpa = []
        self.hist_range = []

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

    def ts_route_callback(self, msg: RouteIntent):
        self.w_ts_delayed = np.array([[pt.x, pt.y] for pt in msg.route])
        if self.t_intent_shared is None:
            self.t_intent_shared = self.cached.get("time", 0.0)
            self.get_logger().info(f"[OS] Intent received at t = {self.t_intent_shared:.2f} s")

    def ts_state_callback(self, msg: VesselKinematics):
        self.x_ts_raw = np.array([msg.x, msg.y, msg.psi, msg.u, msg.v, msg.r])

    def step_gnc_pipeline(self):
        if self.x_ts_raw is None or self.sim_finished:
            return

        # 1. Check if OS has reached the terminal waypoint
        target_wp = self.cached["w_active"][-1, :2]
        dist_to_final = np.linalg.norm(self.internal_state[:2] - target_wp)
        current_u_target = 0.0 if dist_to_final <= VesselParams.D_m else self.u_nominal

        # 2. Step synchronous pipeline
        self.internal_state, self.cached, telemetry = SynchronousPipeline.step(
            internal_state=self.internal_state,
            w_mission_os=self.w_mission_os,
            x_ts_raw=self.x_ts_raw,
            w_ts_delayed=self.w_ts_delayed,
            cached=self.cached,
            dt=self.dt,
            u_nominal=current_u_target
        )

        # 3. Compute instantaneous Range
        current_range = float(np.linalg.norm(self.internal_state[:2] - self.x_ts_raw[:2]))

        # 4. Log data for analysis
        sim_time = telemetry["time"]
        self.hist_time.append(sim_time)
        self.hist_dcpa.append(telemetry["dcpa"])
        self.hist_tcpa.append(telemetry["tcpa"])
        self.hist_range.append(current_range)

        # 5. Publish ROS topics
        os_msg = Float64MultiArray(data=self.internal_state.tolist())
        self.os_state_pub.publish(os_msg)

        w_msg = Float64MultiArray(data=self.cached["w_active"].flatten().tolist())
        self.w_active_pub.publish(w_msg)

        telem_msg = Float64MultiArray(data=[
            sim_time, telemetry["dcpa"], telemetry["tcpa"],
            telemetry["u_c"], telemetry["tau_c"], telemetry["psi_cmd"]
        ])
        self.telemetry_pub.publish(telem_msg)

        # 6. Stop condition: both vessels have zero forward velocity
        os_stopped = abs(self.internal_state[3]) < 0.05 and dist_to_final <= (VesselParams.D_m + 0.2)
        ts_stopped = abs(self.x_ts_raw[3]) < 0.05
        if os_stopped and ts_stopped:
            self.sim_finished = True
            self.get_logger().info("\033[92mBoth vessels arrived at terminal waypoints. Plotting metrics...\033[0m")
            self.plot_encounter_metrics()

        prev_state = self.cached["state"]
        self.cached["w_active"], self.cached["psi_ca"], self.cached["state"] = DecisionLayer.evaluate(
            self.internal_state, self.x_ts_raw, self.w_mission_os, self.w_ts_delayed, self.cached["dcpa"], self.cached["tcpa"]
        )

        if self.cached["state"] != prev_state:
            self.get_logger().info(f"[OS] Transitioned to {self.cached['state']}")

    def plot_encounter_metrics(self):
        t_arr = np.array(self.hist_time)
        dcpa_arr = np.array(self.hist_dcpa)
        tcpa_arr = np.array(self.hist_tcpa)
        range_arr = np.array(self.hist_range)

        # Clamp large initial values for readability
        dcpa_arr = np.clip(dcpa_arr, 0.0, 15.0)
        tcpa_arr = np.clip(tcpa_arr, -10.0, 60.0)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        fig.canvas.manager.set_window_title('COLREGs Encounter Metrics Evaluation')

        # 1. Range Subplot
        ax1.plot(t_arr, range_arr, color='#1f77b4', linewidth=2.0, label='Range')
        ax1.axhline(VesselParams.DCPA_safe, color='black', linestyle=':', label='Safe Boundary')
        ax1.set_ylabel('Range [m]')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='upper right')

        # 2. DCPA Subplot
        ax2.plot(t_arr, dcpa_arr, color='#2ca02c', linewidth=2.0, label='DCPA')
        ax2.axhline(VesselParams.DCPA_safe, color='red', linestyle='--', label=f'DCPA Safe ({VesselParams.DCPA_safe} m)')
        ax2.set_ylabel('DCPA [m]')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='upper right')

        # 3. TCPA Subplot
        ax3.plot(t_arr, tcpa_arr, color='#ff7f0e', linewidth=2.0, label='TCPA')
        ax3.axhline(0.0, color='gray', linestyle=':', label='CPA Horizon (TCPA = 0)')
        ax3.set_ylabel('TCPA [s]')
        ax3.set_xlabel('Simulation Time [s]')
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='upper right')

        # Draw vertical hitmarker across all subplots where intention was shared
        if self.t_intent_shared is not None:
            for ax in (ax1, ax2, ax3):
                ax.axvline(
                    self.t_intent_shared, color='crimson', linestyle='-.', linewidth=1.8,
                    label='Intent Received' if ax == ax1 else None
                )
                ax.text(
                    self.t_intent_shared + 0.4, ax.get_ylim()[0] + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"Intent Shared ({self.t_intent_shared:.1f}s)",
                    color='crimson', fontsize=8, fontweight='bold'
                )

        plt.tight_layout()
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()