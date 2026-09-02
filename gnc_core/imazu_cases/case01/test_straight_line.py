"""
test_straight_line.py
Animated Live-Visualization: Own Ship straight-line tracking.
"""

import sys
from pathlib import Path
root_path = Path(__file__).resolve().parents[3]
sys.path.append(str(root_path))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from gnc_core.simulation.pipeline import SynchronousPipeline
from gnc_core.config.vessel_params import VesselParams


def create_vessel_polygon(x: float, y: float, psi: float, length: float = 0.98, beam: float = 0.30):
    """Generates 2D coordinates for a boat polygon in NED (x=North, y=East)."""
    half_l, half_b = length / 2.0, beam / 2.0
    # Local vertices: [dx (North), dy (East)]
    local_pts = np.array([
        [half_l, 0.0],              # Bow tip
        [half_l * 0.5, half_b],     # Bow starboard
        [-half_l, half_b],          # Stern starboard
        [-half_l, -half_b],         # Stern port
        [half_l * 0.5, -half_b],    # Bow port
    ])
    # Rotate by heading psi (measured clockwise from North)
    R = np.array([
        [np.cos(psi), -np.sin(psi)],
        [np.sin(psi),  np.cos(psi)]
    ])
    rot_pts = (R @ local_pts.T).T
    # Return as [East, North] for Matplotlib (X=East, Y=North)
    return np.column_stack([y + rot_pts[:, 1], x + rot_pts[:, 0]])


def run_animated_straight_line():
    dt = 0.02
    w_mission_os = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0]
    ], dtype=np.float64)

    # Initial state: [X, Y, psi, r, b, u]
    state = np.array([0.0, 0.8, np.radians(-20.0), 0.0, 0.0, 0.0], dtype=np.float64)
    x_ts_dummy = np.array([100.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    cached = {
        "dcpa": float("inf"),
        "tcpa": float("inf"),
        "w_active": np.copy(w_mission_os),
        "psi_ca": 0.0,
        "psi_wp": 0.0,
        "wp_idx": 1,
        "state": "State A.2"
    }

    # Setup plots
    fig, (ax_map, ax_resp) = plt.subplots(1, 2, figsize=(13, 6))

    # Left: Spatial NED Map (Y=East on horizontal, X=North on vertical)
    ax_map.plot(w_mission_os[:, 1], w_mission_os[:, 0], "r--o", label="Planned Track")
    (traj_line,) = ax_map.plot([], [], "b-", linewidth=2, label="Own Ship Trajectory")
    vessel_patch = Polygon([[0, 0]], closed=True, fc="cyan", ec="blue", zorder=5)
    ax_map.add_patch(vessel_patch)
    ax_map.set_xlim(-4, 4)
    ax_map.set_ylim(-1, 22)
    ax_map.set_xlabel("East (Y) [m]")
    ax_map.set_ylabel("North (X) [m]")
    ax_map.set_title("Live Vessel Navigation (NED)")
    ax_map.grid(True)
    ax_map.legend(loc="upper left")

    # Right: Response Plot
    (line_cmd,) = ax_resp.plot([], [], "r--", label="Commanded Heading (psi_cmd)")
    (line_psi,) = ax_resp.plot([], [], "b-", label="Actual Heading (psi)")
    (line_u,) = ax_resp.plot([], [], "g-.", label="Surge Speed (u)")
    ax_resp.set_xlim(0, 30)
    ax_resp.set_ylim(-40, 50)
    ax_resp.set_xlabel("Time [s]")
    ax_resp.set_ylabel("Degrees / [m/s]")
    ax_resp.set_title("Controller Response")
    ax_resp.grid(True)
    ax_resp.legend(loc="lower right")

    # Log buffers
    history = {"t": [], "x": [], "y": [], "psi": [], "cmd": [], "u": []}
    step_container = {"state": state, "cached": cached, "tick": 0}

    # Step multiplier to render smoothly in real time
    sub_steps = 5

    def update(frame):
        for _ in range(sub_steps):
            s = step_container["state"]
            c = step_container["cached"]
            t = step_container["tick"]

            s_next, c_next, telem = SynchronousPipeline.step(
                internal_state=s,
                w_mission_os=w_mission_os,
                x_ts_raw=x_ts_dummy,
                w_ts_delayed=None,
                tick=t,
                cached=c,
                dt=dt,
                u_nominal=0.5
            )

            step_container["state"] = s_next
            step_container["cached"] = c_next
            step_container["tick"] += 1

            history["t"].append(telem["time"])
            history["x"].append(telem["x_os"][0])
            history["y"].append(telem["x_os"][1])
            history["psi"].append(np.degrees(telem["x_os"][2]))
            history["cmd"].append(np.degrees(telem["psi_cmd"]))
            history["u"].append(telem["x_os"][3])

        # Update Map Objects
        traj_line.set_data(history["y"], history["x"])
        poly_coords = create_vessel_polygon(
            history["x"][-1], history["y"][-1], np.radians(history["psi"][-1])
        )
        vessel_patch.set_xy(poly_coords)

        # Update Telemetry Objects
        line_cmd.set_data(history["t"], history["cmd"])
        line_psi.set_data(history["t"], history["psi"])
        line_u.set_data(history["t"], history["u"])

        return traj_line, vessel_patch, line_cmd, line_psi, line_u

    anim = FuncAnimation(fig, update, frames=300, interval=20, blit=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_animated_straight_line()