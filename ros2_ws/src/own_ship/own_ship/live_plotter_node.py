#!/usr/bin/env python3
import math
import os
from pathlib import Path
import sys
import threading

parent_repo = Path(
    "/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs"
)
if str(parent_repo) not in sys.path:
  sys.path.insert(0, str(parent_repo))

from gnc_core.config.vessel_params import VesselParams
from maritime_interfaces.msg import RouteIntent, VesselKinematics
from matplotlib.patches import Ellipse, Polygon
import matplotlib.pyplot as plt
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def draw_tito_neri(
    ax, x, y, psi, color, label=None, draw_domain=False, draw_obb=False
):
  dx_bow = VesselParams.L / 2.0
  dx_shoulder = VesselParams.L / 6.0
  dx_stern = -VesselParams.L / 2.0
  dy_half = VesselParams.B / 2.0

  hull_local = np.array([
      [dx_bow, 0.0],
      [dx_shoulder, dy_half],
      [dx_stern, dy_half],
      [dx_stern, -dy_half],
      [dx_shoulder, -dy_half],
  ])

  c_psi = np.cos(psi)
  s_psi = np.sin(psi)
  r_mat = np.array([[c_psi, -s_psi], [s_psi, c_psi]])

  hull_ned = (r_mat @ hull_local.T).T
  hull_plot = np.column_stack([y + hull_ned[:, 1], x + hull_ned[:, 0]])

  polygon = Polygon(
      hull_plot,
      closed=True,
      facecolor=color,
      edgecolor="#111111",
      linewidth=1.2,
      zorder=5,
      label=label,
  )
  ax.add_patch(polygon)

  if draw_domain:
    domain_ellipse = Ellipse(
        xy=(y, x),
        width=2.0 * VesselParams.R_lateral,
        height=2.0 * VesselParams.R_long,
        angle=-np.degrees(psi),
        edgecolor="#1f77b4",
        facecolor=(0.12, 0.47, 0.71, 0.15),
        linestyle="--",
        linewidth=1.2,
        zorder=4,
        label="OS Domain",
    )
    ax.add_patch(domain_ellipse)

  if draw_obb:
    obb_local = np.array([
        [VesselParams.L / 2.0, VesselParams.B / 2.0],
        [VesselParams.L / 2.0, -VesselParams.B / 2.0],
        [-VesselParams.L / 2.0, -VesselParams.B / 2.0],
        [-VesselParams.L / 2.0, VesselParams.B / 2.0],
    ])
    obb_ned = (r_mat @ obb_local.T).T
    obb_plot = np.column_stack([y + obb_ned[:, 1], x + obb_ned[:, 0]])

    obb_box = Polygon(
        obb_plot,
        closed=True,
        facecolor=(0.85, 0.15, 0.15, 0.10),
        edgecolor="#d62728",
        linewidth=1.0,
        linestyle=":",
        zorder=4,
        label="TS OBB",
    )
    ax.add_patch(obb_box)


class LivePlotterNode(Node):

  def __init__(self):
    super().__init__("live_plotter")

    self.declare_parameter("scenario", "case01")
    self.declare_parameter("mode", False) # Receives string 'True' or 'False'
    self.declare_parameter("latency", 0.0)
    self.declare_parameter("interval", 3.0) 
    self.declare_parameter("auto_close", False) 

    self.scenario_name = str(self.get_parameter("scenario").value).lower()
    
    # Safely handle the mode whether ROS passes it as a bool or a string
    mode_val = self.get_parameter("mode").value
    if isinstance(mode_val, str):
        self.sim_mode = "IS" if mode_val.upper() == "TRUE" else "RA"
    else:
        self.sim_mode = "IS" if mode_val else "RA"
    
    self.sim_latency = float(self.get_parameter("latency").value)
    self.sim_interval = float(self.get_parameter("interval").value)
    
    # Safely handle auto_close
    ac_val = self.get_parameter("auto_close").value
    if isinstance(ac_val, str):
        self.auto_close = (ac_val.upper() == "TRUE")
    else:
        self.auto_close = bool(ac_val)

    from gnc_core.imazu_cases.scenario_loader import load_scenario

    self.config = load_scenario(self.scenario_name)

    all_pts = np.vstack([
        self.config["os_mission_wps"],
        self.config["ts_mission_wps"],
        self.config["os_initial_state"][:2],
        self.config["ts_initial_state"][:2],
    ])

    min_x, max_x = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
    min_y, max_y = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))

    pad_x = max(3.0, (max_x - min_x) * 0.1)
    pad_y = max(4.0, (max_y - min_y) * 0.3)

    self.ylim = (min_x - pad_x, max_x + pad_x)
    self.xlim = (min_y - pad_y, max_y + pad_y)

    self.os_state = self.config["os_initial_state"][:3].tolist()
    self.ts_state = self.config["ts_initial_state"][:3].tolist()
    self.telemetry = {"time": 0.0, "dcpa": 0.0, "tcpa": 0.0}

    self.os_wps_x = self.config["os_mission_wps"][:, 0].tolist()
    self.os_wps_y = self.config["os_mission_wps"][:, 1].tolist()
    self.ts_route_x = []
    self.ts_route_y = []

    self.os_history_x = []
    self.os_history_y = []
    self.os_history_psi = []
    self.os_history_r = []
    self.ts_history_x = []
    self.ts_history_y = []
    self.time_history = []

    self.create_subscription(
        Float64MultiArray, "/os/state_vector", self.os_callback, 10
    )
    self.create_subscription(
        Float64MultiArray, "/os/active_waypoints", self.os_wps_callback, 10
    )
    self.create_subscription(
        Float64MultiArray, "/os/telemetry", self.telem_callback, 10
    )
    self.create_subscription(
        VesselKinematics, "/ts/state_vector", self.ts_callback, 10
    )
    self.create_subscription(
        RouteIntent, "/ts/route_delayed", self.ts_route_callback, 10
    )

    self.get_logger().info(
        f"Live Plotter ready [{self.scenario_name}] | Mode: {self.sim_mode} | "
        f"Latency: {self.sim_latency:.1f}s | AutoClose: {self.auto_close}"
    )

  def telem_callback(self, msg):
    if len(msg.data) >= 3:
      self.telemetry["time"] = msg.data[0]
      self.telemetry["dcpa"] = msg.data[1]
      self.telemetry["tcpa"] = msg.data[2]

  def os_wps_callback(self, msg):
    arr = np.array(msg.data).reshape(-1, 2)
    self.os_wps_x = arr[:, 0].tolist()
    self.os_wps_y = arr[:, 1].tolist()

  def os_callback(self, msg):
    if len(msg.data) < 3:
        return
    x, y, psi = msg.data[0], msg.data[1], msg.data[2]
    
    # Filter origin drops
    if self.os_history_x and abs(x) < 1e-4 and abs(y) < 1e-4:
        return

    self.os_state = [x, y, psi]
    self.os_history_x.append(x)
    self.os_history_y.append(y)
    self.os_history_psi.append(psi)

    r_val = msg.data[5] if len(msg.data) > 5 else 0.0
    self.os_history_r.append(r_val)
    self.time_history.append(self.telemetry["time"])

  def ts_callback(self, msg):
    # Filter origin dropouts
    if abs(msg.x) < 1e-4 and abs(msg.y) < 1e-4:
        return

    # Check for unreasonable jumps without clearing history
    if self.ts_history_x:
        dist = math.hypot(msg.x - self.ts_history_x[-1], msg.y - self.ts_history_y[-1])
        if dist > 2.0:
            return  # Drop outlier point instead of clearing track

    self.ts_state = [msg.x, msg.y, msg.psi]
    self.ts_history_x.append(msg.x)
    self.ts_history_y.append(msg.y)

  def ts_route_callback(self, msg):
    self.ts_route_x = [pt.x for pt in msg.route]
    self.ts_route_y = [pt.y for pt in msg.route]

  def save_run_log(self):
    if not self.os_history_x:
        self.get_logger().warn("No trajectory data recorded to save.")
        return

    output_dir = os.path.join(str(parent_repo), "simulation_logs")
    os.makedirs(output_dir, exist_ok=True)

    if self.sim_mode == "IS":
        tag = f"{self.scenario_name}_IS_tau{self.sim_latency:.1f}_dt{self.sim_interval:.1f}"
    else:
        tag = f"{self.scenario_name}_RA"

    filepath = os.path.join(output_dir, f"{tag}.npz")

    n = min(
        len(self.time_history), len(self.os_history_x), len(self.ts_history_x)
    )
    if n == 0: return

    t_arr = np.array(self.time_history[:n])
    if np.all(t_arr == 0.0) or len(np.unique(t_arr)) <= 1:
        t_arr = np.linspace(0.0, 0.1 * n, n)

    np.savez(
        filepath,
        t=t_arr,
        os_pos=np.column_stack((self.os_history_x[:n], self.os_history_y[:n])),
        os_psi=np.array(self.os_history_psi[:n]),
        os_r=np.array(self.os_history_r[:n]),
        ts_pos=np.column_stack((self.ts_history_x[:n], self.ts_history_y[:n])),
        nominal_wps=np.array(self.config["os_mission_wps"]),
        scenario=self.scenario_name,
        mode=self.sim_mode,
        latency=self.sim_latency,
        interval=self.sim_interval,
    )
    self.get_logger().info(f"Successfully saved run log: {filepath}")


def main(args=None):
  rclpy.init(args=args)
  node = LivePlotterNode()

  spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
  spin_thread.start()

  plt.ion()
  fig, ax = plt.subplots(figsize=(6.5, 9.5))
  fig.canvas.manager.set_window_title(
      f"GNC Simulation: {node.scenario_name.upper()}"
  )

  try:
    while rclpy.ok():
      if not plt.fignum_exists(fig.number):
          break # Exit loop cleanly if user closed the window mid-run
          
      ax.clear()

      ax.set_xlim(node.xlim[0], node.xlim[1])
      ax.set_ylim(node.ylim[0], node.ylim[1])
      ax.set_aspect("equal", adjustable="box")
      ax.set_xlabel("East (y) [m]", fontsize=10, fontweight="medium")
      ax.set_ylabel("North (x) [m]", fontsize=10, fontweight="medium")
      ax.set_title(
          f"Autonomous Collision Avoidance ({node.scenario_name.upper()})",
          fontsize=11, fontweight="bold", pad=10,
      )
      ax.grid(True, linestyle="--", alpha=0.45)
      ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, label="Original Mission")

      if len(node.os_wps_x) >= 2:
        ax.plot(node.os_wps_y, node.os_wps_x, color="#1f77b4", linestyle="--", linewidth=1.4, alpha=0.8, label="OS Active Route")
        ax.scatter(node.os_wps_y, node.os_wps_x, color="#1f77b4", marker="o", s=18, zorder=3)

      if node.ts_route_x and node.ts_route_y:
        ax.plot(node.ts_route_y, node.ts_route_x, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.8, label="TS Route Intent")
        ax.scatter(node.ts_route_y, node.ts_route_x, color="#d62728", marker="s", s=16, zorder=3)

      ax.plot(list(node.os_history_y), list(node.os_history_x), color="#0047ab", linewidth=2.0, alpha=0.9, label="OS Track")
      ax.plot(list(node.ts_history_y), list(node.ts_history_x), color="#b22222", linewidth=2.0, alpha=0.9, label="TS Track")

      draw_tito_neri(ax, node.os_state[0], node.os_state[1], node.os_state[2], color="#2b6cb0", label="Own Ship", draw_domain=True)
      draw_tito_neri(ax, node.ts_state[0], node.ts_state[1], node.ts_state[2], color="#c53030", label="Target Ship")

      hud_text = (
          f"Sim Time: {node.telemetry['time']:4.1f} s\n"
          f"DCPA:     {node.telemetry['dcpa']:4.2f} m\n"
          f"TCPA:     {node.telemetry['tcpa']:4.1f} s"
      )
      ax.text(
          0.03, 0.97, hud_text, transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
          verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.85)
      )

      ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.97), fontsize=8.0, framealpha=0.9, edgecolor="#cccccc")
      plt.tight_layout()
      plt.pause(0.1)

      # Determine if the scenario is physically over
      time_limit = 220.0 
      if isinstance(node.config, dict):
        time_limit = node.config.get("sim_time_limit", node.config.get("t_sim", 220.0))

      encounter_cleared = False
      if len(node.os_history_x) > 20 and len(node.ts_history_x) > 20:
        curr_dist = math.hypot(node.os_state[0] - node.ts_state[0], node.os_state[1] - node.ts_state[1])
        # If they've passed each other (TCPA < 0) and are 200m apart, it's over
        if node.telemetry["tcpa"] < -5.0 and curr_dist > 200.0:
          encounter_cleared = True

      if (node.telemetry["time"] >= time_limit and node.telemetry["time"] > 0.0) or encounter_cleared:
        node.get_logger().info(f"Encounter completed at t={node.telemetry['time']:.1f}s.")
        
        if node.auto_close:
            node.get_logger().info("Auto-close active. Exiting immediately for batch run...")
            break
        else:
            node.get_logger().info("Single-run mode. Plot frozen. Close the window to save and exit.")
            plt.ioff()
            plt.show(block=True)
            break

  except KeyboardInterrupt:
    pass
  finally:
    plt.ioff()
    plt.close('all')
    node.save_run_log()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    os._exit(0)

if __name__ == "__main__":
  main()