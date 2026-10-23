#!/usr/bin/env python3
"""
animate_run.py
Generates an MP4 or GIF animation from a saved simulation .npz log file,
matching the exact visual layout and color palette of own_ship_node's
plot_encounter_metrics():
  - Tito Neri hull geometry for OS and TS (with accurate heading orientation)
  - OS Ship Domain Ellipse
  - Sand canal banks (#ebdcb9) with shoreline edge detection
  - Dynamic active OS route & perceived TS intent route
  - Monospace HUD text box (Sim Time, DCPA, TCPA)
  - Configurable playback speedup (1x, 2x, 3x, 4x, etc.)
  - Saves all animations directly to results_plots/
"""

import argparse
from pathlib import Path
import sys

# Connect to the external core Python library
parent_repo = Path('/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs')
if str(parent_repo) not in sys.path:
    sys.path.insert(0, str(parent_repo))

OUTPUT_DIR = parent_repo / "results_plots"

import matplotlib.animation as animation
from matplotlib.patches import Ellipse, Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon as ShapelyMultiPolygon

from gnc_core.config.vessel_params import VesselParams
from gnc_core.imazu_cases.scenario_loader import load_scenario


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

    polygon = MplPolygon(
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

    # --- Heading Angle Recovery for Target Ship ---
    if "ts_psi" in data and not np.all(data["ts_psi"] == 0):
        ts_psi = data["ts_psi"]
    else:
        # Reconstruct TS heading psi in NED (x=North, y=East) using trajectory gradient
        if len(ts_pos) > 1:
            dx_north = np.gradient(ts_pos[:, 0])
            dy_east = np.gradient(ts_pos[:, 1])
            ts_psi = np.arctan2(dy_east, dx_north)

            # Hold previous heading when stopped
            speed = np.hypot(dx_north, dy_east)
            for i in range(1, len(ts_psi)):
                if speed[i] < 1e-3:
                    ts_psi[i] = ts_psi[i - 1]
        else:
            ts_psi = np.zeros_like(t)

    # Dynamic Route Histories
    os_wps_hist = data["os_wps_hist"] if "os_wps_hist" in data else None
    ts_route_hist = data["ts_route_hist"] if "ts_route_hist" in data else None

    scenario_name = str(data["scenario"]).lower() if "scenario" in data else "case01"
    sim_mode = str(data["mode"]) if "mode" in data else "RA"
    sim_latency = float(data["latency"]) if "latency" in data else 0.0
    sim_interval = float(data["interval"]) if "interval" in data else 5.0

    # Load Scenario Config
    try:
        config = load_scenario(scenario_name)
    except Exception as e:
        print(f"Warning: Could not load scenario config '{scenario_name}': {e}")
        config = {}

    nom_wps = data["nominal_wps"] if "nominal_wps" in data else config.get("os_mission_wps", np.array([]))
    evasive_wps = data["evasive_wps"] if "evasive_wps" in data else np.array([])

    fig, ax = plt.subplots(figsize=(6, 9), dpi=100)
    fig.canvas.manager.set_window_title("2D Spatial Trajectory Animation")

    # ----------------------------------------------------------------------
    # 1. Title String
    # ----------------------------------------------------------------------
    title_str = f"Run Trajectory: {scenario_name.upper()} ({sim_mode})"
    if sim_mode == "IS":
        title_str += f" | $\\tau$={sim_latency:.1f}s, $\\Delta T$={sim_interval:.1f}s"
    ax.set_title(title_str, fontsize=11, fontweight="bold", pad=10)

    ax.set_xlabel("East (y) [m]", fontsize=10)
    ax.set_ylabel("North (x) [m]", fontsize=10)
    ax.grid(True, linestyle="-", alpha=0.5)

    # ----------------------------------------------------------------------
    # 2. Draw Canal Banks
    # ----------------------------------------------------------------------
    bank_edge_color = '#8c564b'
    bank_face_color = '#ebdcb9'
    bank_labeled = False

    polys = []
    canal_polygons = data["canal_polygons"] if "canal_polygons" in data else config.get("canal_polygons", None)
    if canal_polygons is not None:
        raw_poly = canal_polygons
        if hasattr(raw_poly, 'item'):
            try:
                raw_poly = raw_poly.item()
            except Exception:
                pass

        if isinstance(raw_poly, ShapelyPolygon):
            polys = [raw_poly]
        elif isinstance(raw_poly, ShapelyMultiPolygon) or hasattr(raw_poly, 'geoms'):
            polys = list(raw_poly.geoms)

    if polys:
        for poly in polys:
            x_pts, y_pts = poly.exterior.xy
            coords = np.column_stack([y_pts, x_pts])
            patch = MplPolygon(
                coords,
                closed=True,
                facecolor=bank_face_color,
                edgecolor=bank_edge_color,
                linewidth=1.4,
                alpha=0.75,
                zorder=1,
                label="Canal Bank" if not bank_labeled else None
            )
            ax.add_patch(patch)
            bank_labeled = True

    elif "canal_bounds" in data or "canal_bounds" in config:
        cb = data["canal_bounds"].item() if "canal_bounds" in data else config.get("canal_bounds", {})
        y_min = cb.get('y_min', -4.0)
        y_max = cb.get('y_max', 4.0)
        x_min, x_max = -200.0, 200.0

        port_coords = np.array([[-200.0, x_min], [y_min, x_min], [y_min, x_max], [-200.0, x_max]])
        ax.add_patch(MplPolygon(port_coords, closed=True, facecolor=bank_face_color,
                                 edgecolor=bank_edge_color, linewidth=1.4, alpha=0.75, zorder=1, label="Canal Bank"))
        stbd_coords = np.array([[y_max, x_min], [200.0, x_min], [200.0, x_max], [y_max, x_max]])
        ax.add_patch(MplPolygon(stbd_coords, closed=True, facecolor=bank_face_color,
                                 edgecolor=bank_edge_color, linewidth=1.4, alpha=0.75, zorder=1))

    # ----------------------------------------------------------------------
    # 3. Dynamic Axes Boundaries
    # ----------------------------------------------------------------------
    pts_north = [os_pos[:, 0], ts_pos[:, 0]]
    pts_east = [os_pos[:, 1], ts_pos[:, 1]]
    if len(nom_wps) > 0:
        pts_north.append(nom_wps[:, 0])
        pts_east.append(nom_wps[:, 1])

    all_north = np.concatenate(pts_north)
    all_east = np.concatenate(pts_east)

    n_min, n_max = float(np.min(all_north)), float(np.max(all_north))
    n_pad = max(2.5, 0.05 * (n_max - n_min))
    y_bottom = n_min - n_pad
    y_top = n_max + n_pad
    ax.set_ylim(y_bottom, y_top)

    port_inners, stbd_inners = [], []
    if polys:
        for poly in polys:
            x_pts, y_pts = poly.exterior.xy
            n_pts = len(x_pts)
            for i in range(n_pts - 1):
                x1, x2 = x_pts[i], x_pts[i + 1]
                y1, y2 = y_pts[i], y_pts[i + 1]
                if max(x1, x2) >= y_bottom and min(x1, x2) <= y_top:
                    y_mid = 0.5 * (y1 + y2)
                    if y_mid < 0:
                        port_inners.append(max(y1, y2))
                    else:
                        stbd_inners.append(min(y1, y2))

    cb = data["canal_bounds"].item() if "canal_bounds" in data else config.get("canal_bounds", None)
    if cb is not None:
        if 'y_min' in cb:
            port_inners.append(float(cb['y_min']))
        if 'y_max' in cb:
            stbd_inners.append(float(cb['y_max']))

    port_inner = max(port_inners) if port_inners else min(-4.0, float(np.min(all_east)) - 1.5)
    stbd_inner = min(stbd_inners) if stbd_inners else max(4.0, float(np.max(all_east)) + 1.5)

    glimpse_width = 1.5
    x_left = min(float(np.min(all_east)) - 1.0, port_inner - glimpse_width)
    x_right = max(float(np.max(all_east)) + 1.0, stbd_inner + glimpse_width)
    ax.set_xlim(x_left, x_right)

    # ----------------------------------------------------------------------
    # 4. Planned Routes & Dynamic Overlays
    # ----------------------------------------------------------------------
    if len(nom_wps) > 0:
        ax.plot(nom_wps[:, 1], nom_wps[:, 0], 'k--', alpha=0.45, linewidth=1.2, label='Original Mission')

    (os_track,) = ax.plot([], [], color="#1f77b4", linewidth=2.2, label="OS Track", zorder=3)
    (ts_track,) = ax.plot([], [], color="#d62728", linewidth=2.2, label="TS Track", zorder=3)

    (os_route_line,) = ax.plot([], [], color="tab:cyan", linestyle="--", linewidth=1.2, marker="o", markersize=3.5, label="Active OS Route", zorder=3)
    (ts_route_line,) = ax.plot([], [], color="#d62728", linestyle="--", linewidth=1.2, marker="s", markersize=3.5, label="Perceived TS Intent", zorder=3)

    # Start Position Markers
    ax.scatter(os_pos[0, 1], os_pos[0, 0], color='#1f77b4', s=45, marker='o', zorder=4)
    ax.scatter(ts_pos[0, 1], ts_pos[0, 0], color='#d62728', s=45, marker='o', zorder=4)

    # Monospace HUD Text Box
    hud_box = ax.text(
        0.03, 0.97, "", transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
        verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.85),
        zorder=10
    )

    ax.legend(loc="lower left", fontsize=8.0, framealpha=0.92)

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
        elif len(evasive_wps) > 0:
            os_route_line.set_data(evasive_wps[:, 1], evasive_wps[:, 0])

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

        # Draw Hulls & OS Domain with Correct Heading Rotation
        x_os, y_os, psi_os = os_pos[frame_idx, 0], os_pos[frame_idx, 1], os_psi[frame_idx]
        x_ts, y_ts, psi_ts_curr = ts_pos[frame_idx, 0], ts_pos[frame_idx, 1], ts_psi[frame_idx]

        p_os, p_dom = draw_tito_neri(ax, x_os, y_os, psi_os, color="#1f77b4", draw_domain=True)
        p_ts, _ = draw_tito_neri(ax, x_ts, y_ts, psi_ts_curr, color="#d62728", draw_domain=False)
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

    print(f"Rendering animation to '{output_file}' ({len(frame_indices)} frames @ {fps} FPS, {speedup}x speedup)...")
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