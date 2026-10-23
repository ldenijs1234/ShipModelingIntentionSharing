#!/usr/bin/env python3
"""
animate_run.py
Generates an MP4 or GIF animation from a saved simulation .npz log file,
matching all visual features of LivePlotterNode:
  - Tito Neri hull geometry for OS and TS
  - OS Ship Domain Ellipse
  - Canal banks and polygon geometry
  - Active OS route / detour updating dynamically frame-by-frame
  - Perceived TS intent route updating dynamically frame-by-frame
  - Monospace HUD text box (Sim Time, DCPA, TCPA)
  - Saves all animations directly to results_plots/
"""

import argparse
from pathlib import Path
import sys

# Ensure parent directory is in sys.path
parent_repo = Path('/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs')
if str(parent_repo) not in sys.path:
    sys.path.insert(0, str(parent_repo))

OUTPUT_DIR = Path("results_plots")

from gnc_core.config.vessel_params import VesselParams
from gnc_core.imazu_cases.scenario_loader import load_scenario
import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon
import matplotlib.pyplot as plt
import numpy as np


def draw_tito_neri(ax, x, y, psi, color, label=None, draw_domain=False):
    """Draws exact Tito Neri vessel polygon and optional OS domain ellipse in NED (x=North, y=East)."""
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

    domain_patch = None
    if draw_domain:
        domain_patch = Ellipse(
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
        ax.add_patch(domain_patch)

    return polygon, domain_patch


def compute_dcpa_tcpa(p_os, v_os, p_ts, v_ts):
    """Computes instantaneous DCPA and TCPA for HUD display."""
    r = p_ts - p_os
    v = v_ts - v_os
    v_sq = np.dot(v, v)
    if v_sq < 1e-6:
        return float(np.linalg.norm(r)), 0.0
    tcpa = -np.dot(r, v) / v_sq
    if tcpa < 0:
        dcpa = np.linalg.norm(r)
    else:
        cpa_pos = r + v * tcpa
        dcpa = np.linalg.norm(cpa_pos)
    return float(dcpa), float(tcpa)


def animate_scenario(log_path: str, output_path: str = None, fps: int = 15, speedup: float = 1.0, fmt: str = "mp4"):
    log_file = Path(log_path)
    if not log_file.exists():
        print(f"Error: Log file '{log_path}' not found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_file = OUTPUT_DIR / f"{log_file.stem}_animation.{fmt}"
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(log_file, allow_pickle=True)
    t = data["t"]
    os_pos = data["os_pos"]
    os_psi = data["os_psi"]
    ts_pos = data["ts_pos"]
    ts_psi = data["ts_psi"] if "ts_psi" in data else np.zeros_like(t)

    # Dynamic Route Histories
    os_wps_hist = data["os_wps_hist"] if "os_wps_hist" in data else None
    ts_route_hist = data["ts_route_hist"] if "ts_route_hist" in data else None

    scenario_name = str(data["scenario"]).lower() if "scenario" in data else "case01"
    sim_mode = str(data["mode"]) if "mode" in data else "Run"

    # Load Scenario Config
    try:
        config = load_scenario(scenario_name)
    except Exception as e:
        print(f"Warning: Could not load scenario config '{scenario_name}': {e}")
        config = {}

    # Calculate Bounds
    all_pts = np.vstack([
        os_pos,
        ts_pos,
        config.get("os_mission_wps", os_pos[:1]),
        config.get("ts_mission_wps", ts_pos[:1]),
    ])
    min_x, max_x = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
    min_y, max_y = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))

    if "canal_bounds" in config:
        cb = config["canal_bounds"]
        xlim = (float(cb.get("y_min", -10.0)) - 4.0, float(cb.get("y_max", 10.0)) + 4.0)
        ylim = (float(cb.get("x_min", min_x)) - 5.0, float(cb.get("x_max", max_x)) + 5.0)
    else:
        pad_x = max(3.0, (max_x - min_x) * 0.1)
        pad_y = max(4.0, (max_y - min_y) * 0.3)
        ylim = (min_x - pad_x, max_x + pad_x)
        xlim = (min_y - pad_y, max_y + pad_y)

    fig, ax = plt.subplots(figsize=(6.5, 9.5), dpi=100)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (y) [m]", fontsize=10, fontweight="medium")
    ax.set_ylabel("North (x) [m]", fontsize=10, fontweight="medium")
    ax.set_title(f"Autonomous Collision Avoidance ({scenario_name.upper()})", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6, label="Original Mission")

    # Static Canal Geometry
    if "left_bank" in config and "right_bank" in config:
        lb, rb = config["left_bank"], config["right_bank"]
        ax.plot(lb[:, 1], lb[:, 0], color="#8b7355", linewidth=2.0)
        ax.plot(rb[:, 1], rb[:, 0], color="#8b7355", linewidth=2.0, label="Canal Bank")
    elif config.get("canal_polygons") is not None:
        geom = config["canal_polygons"]
        geoms = [geom] if geom.geom_type == 'Polygon' else getattr(geom, 'geoms', [geom])
        for i, g in enumerate(geoms):
            coords = np.array(g.exterior.coords)
            plot_coords = np.column_stack([coords[:, 1], coords[:, 0]])
            ax.add_patch(Polygon(plot_coords, closed=True, facecolor="#d2b48c", edgecolor="#8b7355", linewidth=1.2, alpha=0.4, label="Canal Bank" if i == 0 else None))

    # Dynamic Overlays
    (os_track,) = ax.plot([], [], color="#0047ab", linewidth=2.0, alpha=0.9, label="OS Track")
    (ts_track,) = ax.plot([], [], color="#b22222", linewidth=2.0, alpha=0.9, label="TS Track")

    (os_route_line,) = ax.plot([], [], color="#1f77b4", linestyle="--", linewidth=1.4, alpha=0.8, marker="o", markersize=3, label="Active OS Route")
    (ts_route_line,) = ax.plot([], [], color="#d62728", linestyle="--", linewidth=1.4, alpha=0.8, marker="s", markersize=3, label="Perceived TS Intent")

    # Monospace HUD Text Box
    hud_box = ax.text(
        0.03, 0.97, "", transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.85)
    )

    ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.97), fontsize=8.0, framealpha=0.9, edgecolor="#cccccc")

    if len(t) > 1:
        os_vx, os_vy = np.gradient(os_pos[:, 0], t), np.gradient(os_pos[:, 1], t)
        ts_vx, ts_vy = np.gradient(ts_pos[:, 0], t), np.gradient(ts_pos[:, 1], t)
    else:
        os_vx = os_vy = ts_vx = ts_vy = np.zeros_like(t)

    active_patches = []

    dt_sim = t[1] - t[0] if len(t) > 1 else 0.05
    sim_seconds_per_frame = speedup * (1.0 / fps)
    step = max(1, int(round(sim_seconds_per_frame / dt_sim)))
    frame_indices = list(range(0, len(t), step))

    def update(frame_idx):
        nonlocal active_patches
        for patch in active_patches:
            if patch is not None:
                patch.remove()
        active_patches.clear()

        # Update Executed Tracks
        os_track.set_data(os_pos[:frame_idx, 1], os_pos[:frame_idx, 0])
        ts_track.set_data(ts_pos[:frame_idx, 1], ts_pos[:frame_idx, 0])

        # Update Dynamic Active OS Route
        if os_wps_hist is not None and frame_idx < len(os_wps_hist):
            wps_os = np.array(os_wps_hist[frame_idx])
            if wps_os.ndim == 2 and wps_os.shape[0] >= 2 and wps_os.shape[1] == 2:
                os_route_line.set_data(wps_os[:, 1], wps_os[:, 0])
            else:
                os_route_line.set_data([], [])
        elif "evasive_wps" in data and len(data["evasive_wps"]) > 0:
            wps_os = np.array(data["evasive_wps"])
            if wps_os.ndim == 2 and wps_os.shape[1] == 2:
                os_route_line.set_data(wps_os[:, 1], wps_os[:, 0])

        # Update Dynamic Perceived TS Route
        if ts_route_hist is not None and frame_idx < len(ts_route_hist):
            wps_ts = np.array(ts_route_hist[frame_idx])
            if wps_ts.ndim == 2 and wps_ts.shape[0] >= 2 and wps_ts.shape[1] == 2:
                ts_route_line.set_data(wps_ts[:, 1], wps_ts[:, 0])
            else:
                ts_route_line.set_data([], [])
        elif "w_ts_delayed" in data and len(data["w_ts_delayed"]) > 0:
            wps_ts = np.array(data["w_ts_delayed"])
            if wps_ts.ndim == 2 and wps_ts.shape[1] == 2:
                ts_route_line.set_data(wps_ts[:, 1], wps_ts[:, 0])

        # Draw Hulls & OS Domain
        x_os, y_os, psi_os = os_pos[frame_idx, 0], os_pos[frame_idx, 1], os_psi[frame_idx]
        x_ts, y_ts, psi_ts = ts_pos[frame_idx, 0], ts_pos[frame_idx, 1], ts_psi[frame_idx]

        p_os, p_dom = draw_tito_neri(ax, x_os, y_os, psi_os, color="#2b6cb0", draw_domain=True)
        p_ts, _ = draw_tito_neri(ax, x_ts, y_ts, psi_ts, color="#c53030", draw_domain=False)
        active_patches.extend([p_os, p_dom, p_ts])

        # DCPA / TCPA Computation
        v_os = np.array([os_vx[frame_idx], os_vy[frame_idx]])
        v_ts = np.array([ts_vx[frame_idx], ts_vy[frame_idx]])
        dcpa, tcpa = compute_dcpa_tcpa(os_pos[frame_idx], v_os, ts_pos[frame_idx], v_ts)

        hud_text = (
            f"Sim Time: {t[frame_idx]:4.1f} s\n"
            f"DCPA:     {dcpa:4.2f} m\n"
            f"TCPA:     {tcpa:4.1f} s"
        )
        hud_box.set_text(hud_text)

        return os_track, ts_track, os_route_line, ts_route_line, hud_box

    anim = animation.FuncAnimation(fig, update, frames=frame_indices, interval=1000 / fps, blit=False)

    print(f"Rendering animation to '{output_file}' ({len(frame_indices)} frames @ {fps} FPS)...")
    if fmt == "gif":
        anim.save(output_file, writer="pillow", fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, extra_args=["-vcodec", "libx264"])
        anim.save(output_file, writer=writer)

    plt.close(fig)
    print(f"[SUCCESS] Saved animation to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render simulation .npz log to MP4/GIF")
    parser.add_argument("log_path", type=str, help="Path to .npz log file")
    parser.add_argument("--out", type=str, default=None, help="Output file path")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--speedup", type=float, default=1.0, help="Playback speed multiplier (e.g. 2.0, 3.0, 4.0)")
    parser.add_argument("--format", type=str, choices=["mp4", "gif"], default="mp4", help="Export format")
    args = parser.parse_args()

    animate_scenario(args.log_path, args.out, fps=args.fps, speedup=args.speedup, fmt=args.format)