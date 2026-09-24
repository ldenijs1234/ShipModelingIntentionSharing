#!/usr/bin/env python3
import sys
import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Ellipse
from shapely.geometry import Polygon, MultiPolygon

# Ensure gnc_core and imazu_cases can be imported
parent_repo = Path('/mnt/c/Users/lars/Documents/Ship dynamics - Thesis Lars de Nijs')
if str(parent_repo) not in sys.path:
    sys.path.insert(0, str(parent_repo))

from gnc_core.config.vessel_params import VesselParams
try:
    from gnc_core.imazu_cases.scenario_loader import load_scenario
except ImportError:
    load_scenario = None


def draw_tito_neri(ax, x, y, psi, color, label=None, draw_domain=False):
    """Draws Tito Neri hull and domain matching live_plotter_node."""
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
    # East (y) on x-axis, North (x) on y-axis
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


def draw_canal_banks_live_style(ax, data, scenario_name: str):
    """Renders canal banks with brown borders and light tan fills matching the live view."""
    bank_edge_color = '#8c564b'
    bank_face_color = '#ebdcb9'  # Warm canal bank fill matching the live screenshot
    label_set = False

    polys = []

    # 1. From saved data in .npz
    if 'canal_polygons' in data:
        raw_poly = data['canal_polygons']
        if hasattr(raw_poly, 'item'):
            try:
                raw_poly = raw_poly.item()
            except Exception:
                pass

        if raw_poly is not None and not (isinstance(raw_poly, np.ndarray) and raw_poly.size == 0):
            if isinstance(raw_poly, Polygon):
                polys = [raw_poly]
            elif isinstance(raw_poly, MultiPolygon) or hasattr(raw_poly, 'geoms'):
                polys = list(raw_poly.geoms)

    # 2. From scenario configuration fallback
    if not polys and load_scenario is not None:
        try:
            cfg = load_scenario(scenario_name.lower())
            poly_full = cfg.get("canal_polygons", None)
            if poly_full is not None:
                polys = [poly_full] if isinstance(poly_full, Polygon) else list(poly_full.geoms)
        except Exception:
            pass

    # Render polygons as filled patches
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
                label="Canal Bank" if not label_set else None
            )
            ax.add_patch(patch)
            label_set = True
        return

    # 3. Fallback: Draw rectangular bounds if explicit polygons are absent
    bounds = data.get('canal_bounds')
    if bounds is not None:
        if hasattr(bounds, 'item'):
            bounds = bounds.item()
        if isinstance(bounds, dict):
            y_min = bounds.get('y_min', -4.0)
            y_max = bounds.get('y_max', 4.0)
            x_min = bounds.get('x_min', -15.0)
            x_max = bounds.get('x_max', 60.0)

            # Port bank patch [y_outer to y_min]
            port_coords = np.array([
                [-15.0, x_min], [y_min, x_min], [y_min, x_max], [-15.0, x_max]
            ])
            ax.add_patch(MplPolygon(port_coords, closed=True, facecolor=bank_face_color,
                                    edgecolor=bank_edge_color, linewidth=1.8, alpha=0.65, zorder=1, label="Canal Bank"))
            
            # Starboard bank patch [y_max to y_outer]
            stbd_coords = np.array([
                [y_max, x_min], [15.0, x_min], [15.0, x_max], [y_max, x_max]
            ])
            ax.add_patch(MplPolygon(stbd_coords, closed=True, facecolor=bank_face_color,
                                    edgecolor=bank_edge_color, linewidth=1.8, alpha=0.65, zorder=1))


def plot_single_run_figure(npz_path: str, fig_idx: int):
    if not os.path.exists(npz_path):
        print(f"Warning: File not found -> {npz_path}")
        return None

    data = np.load(npz_path, allow_pickle=True)

    t = data['t']
    os_pos = data['os_pos']          # [North (x), East (y)]
    ts_pos = data['ts_pos']          # [North (x), East (y)]
    os_psi = data['os_psi'] if 'os_psi' in data else np.zeros(len(t))
    nom_wps = data['nominal_wps'] if 'nominal_wps' in data else np.array([])
    evasive_wps = data['evasive_wps'] if 'evasive_wps' in data else np.array([])
    w_ts_delayed = data['w_ts_delayed'] if 'w_ts_delayed' in data else np.array([])

    scenario = str(data.get('scenario', 'case04'))
    mode = str(data.get('mode', ''))
    latency = float(data.get('latency', 0.0))
    interval = float(data.get('interval', 0.0))

    # Calculate separation distance history and CPA
    dists = np.hypot(os_pos[:, 0] - ts_pos[:, 0], os_pos[:, 1] - ts_pos[:, 1])
    min_dist_idx = int(np.argmin(dists))
    min_dist = float(dists[min_dist_idx])
    t_min = float(t[min_dist_idx])

    fig, ax = plt.subplots(figsize=(8.0, 9.5))
    run_tag = os.path.splitext(os.path.basename(npz_path))[0]
    fig.canvas.manager.set_window_title(f"Run {fig_idx}: {run_tag}")

    # 1. Canal Banks (Rendered with filled patches)
    draw_canal_banks_live_style(ax, data, scenario)

    # 2. Planned routes (East = y, North = x)
    if len(nom_wps) > 0:
        ax.plot(nom_wps[:, 1], nom_wps[:, 0], 'k--', alpha=0.4, label='Original Mission')
        ax.scatter(nom_wps[:, 1], nom_wps[:, 0], c='black', s=20, alpha=0.4)

    if len(w_ts_delayed) > 0:
        ax.plot(w_ts_delayed[:, 1], w_ts_delayed[:, 0], color='#d62728', linestyle='--', linewidth=1.4, alpha=0.7, label='TS Route')
        ax.scatter(w_ts_delayed[:, 1], w_ts_delayed[:, 0], color='#d62728', marker='s', s=16, alpha=0.7)

    if len(evasive_wps) > 0:
        ax.plot(evasive_wps[:, 1], evasive_wps[:, 0], color='#1f77b4', linestyle='--', marker='o',
                markersize=4, label='OS Active Route')

    # 3. Sailed Trajectories
    ax.plot(os_pos[:, 1], os_pos[:, 0], color='#0047ab', linewidth=2.2, label='OS Track')
    ax.plot(ts_pos[:, 1], ts_pos[:, 0], color='#b22222', linewidth=2.2, label='TS Track')

    # 4. Tito Neri Vessel Silhouettes at Terminal Positions
    final_os_psi = float(os_psi[-1]) if len(os_psi) > 0 else 0.0
    final_ts_psi = np.arctan2(ts_pos[-1, 0] - ts_pos[-2, 0], ts_pos[-1, 1] - ts_pos[-2, 1]) if len(ts_pos) > 1 else 0.0
    draw_tito_neri(ax, os_pos[-1, 0], os_pos[-1, 1], final_os_psi, color='#2b6cb0', label='Own Ship', draw_domain=True)
    draw_tito_neri(ax, ts_pos[-1, 0], ts_pos[-1, 1], final_ts_psi, color='#c53030', label='Target Ship')

    # Start markers
    ax.scatter(os_pos[0, 1], os_pos[0, 0], color='#0047ab', s=50, marker='o', label='OS Start')
    ax.scatter(ts_pos[0, 1], ts_pos[0, 0], color='#b22222', s=50, marker='o', label='TS Start')

    # CPA marker
    ax.scatter(os_pos[min_dist_idx, 1], os_pos[min_dist_idx, 0], color='darkorange', s=90, marker='*', zorder=6)
    ax.scatter(ts_pos[min_dist_idx, 1], ts_pos[min_dist_idx, 0], color='darkorange', s=90, marker='*', zorder=6)
    ax.plot([os_pos[min_dist_idx, 1], ts_pos[min_dist_idx, 1]],
            [os_pos[min_dist_idx, 0], ts_pos[min_dist_idx, 0]],
            'k:', linewidth=1.5, label=f'CPA: {min_dist:.2f} m @ {t_min:.1f} s')

    # Title & HUD
    ax.set_title(f"Autonomous Collision Avoidance ({scenario.upper()})", fontsize=11, fontweight='bold', pad=10)

    hud_text = (
        f"Sim Time: {t[-1]:4.1f} s\n"
        f"Min CPA:  {min_dist:4.2f} m\n"
        f"CPA Time: {t_min:4.1f} s"
    )
    ax.text(0.03, 0.97, hud_text, transform=ax.transAxes, fontsize=8.5, fontfamily="monospace",
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", alpha=0.85))

    ax.set_xlabel('East (y) [m]', fontsize=10, fontweight='medium')
    ax.set_ylabel('North (x) [m]', fontsize=10, fontweight='medium')
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.set_aspect('equal', adjustable='box')
    ax.legend(loc='lower left', fontsize=8.0, framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()

    # Save to ros2_ws/results_plots
    output_dir = os.path.join(str(parent_repo), "ros2_ws", "results_plots")
    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, f"{run_tag}.png")

    plt.savefig(output_png, dpi=300)
    print(f"\033[92m[Saved Figure {fig_idx}] {output_png}\033[0m")
    return fig


def main():
    if len(sys.argv) > 1:
        raw_inputs = sys.argv[1:]
        files = []
        for item in raw_inputs:
            matched = glob.glob(item)
            if matched:
                files.extend(matched)
            else:
                files.append(item)
        files = sorted(list(set(files)))
    else:
        files = sorted(glob.glob("simulation_logs/*.npz"))
        if not files:
            print("No .npz files found in simulation_logs/.")
            return

    print(f"Creating live-styled figures for {len(files)} run(s)...")

    for idx, path in enumerate(files, start=1):
        plot_single_run_figure(path, idx)

    plt.show()


if __name__ == '__main__':
    main()