#!/usr/bin/env python3
"""
plot_case_definitions.py
Generates clean, publication-ready 2D route definition plots in the North-East Down (NED) frame
using the exact dynamic polygon and shoreline edge-scanning logic from the own_ship node.

Saves figures to:
results_plots
"""

import glob
import os
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np

try:
    from shapely.geometry import MultiPolygon, Polygon
except ImportError:
    Polygon, MultiPolygon = None, None

# Primary target directory
OUTPUT_DIR = Path(r"results_plots")
REPO_ROOT = Path(__file__).resolve().parents[0]


def plot_single_case_definition(sc_name: str, run_data: dict, out_dir: Path):
    """Plots the case definition using the dynamic boundary and bank-scanning logic from own_ship."""
    fig, ax = plt.subplots(figsize=(6, 9), dpi=300)
    fig.canvas.manager.set_window_title(f"Case Definition: {sc_name.upper()}")

    bank_edge_color = '#8c564b'
    bank_face_color = '#ebdcb9'
    bank_labeled = False

    # -------------------------------------------------------------------------
    # 1. Dynamic Canal Banks Extraction (Matching own_ship node)
    # -------------------------------------------------------------------------
    polys = []
    canal_polygons = run_data.get("canal_polygons", None)
    canal_bounds = run_data.get("canal_bounds", None)

    if canal_polygons is not None:
        raw_poly = canal_polygons
        if hasattr(raw_poly, "item"):
            try:
                raw_poly = raw_poly.item()
            except Exception:
                pass

        if Polygon is not None and isinstance(raw_poly, Polygon):
            polys = [raw_poly]
        elif (MultiPolygon is not None and isinstance(raw_poly, MultiPolygon)) or hasattr(raw_poly, "geoms"):
            polys = list(raw_poly.geoms)
        elif hasattr(raw_poly, "exterior"):
            polys = [raw_poly]
        elif isinstance(raw_poly, (list, tuple)):
            polys = list(raw_poly)

    if polys:
        for poly in polys:
            if hasattr(poly, "exterior"):
                x_pts, y_pts = poly.exterior.xy  # x = North, y = East
            else:
                poly_arr = np.asarray(poly)
                x_pts, y_pts = poly_arr[:, 0], poly_arr[:, 1]

            coords = np.column_stack([y_pts, x_pts])  # Plot: [East, North]
            patch = MplPolygon(
                coords,
                closed=True,
                facecolor=bank_face_color,
                edgecolor=bank_edge_color,
                linewidth=1.4,
                alpha=0.75,
                zorder=1,
                label="Canal Bank" if not bank_labeled else None,
            )
            ax.add_patch(patch)
            bank_labeled = True

    elif canal_bounds is not None:
        raw_bounds = canal_bounds
        if hasattr(raw_bounds, "item"):
            try:
                raw_bounds = raw_bounds.item()
            except Exception:
                pass

        if isinstance(raw_bounds, dict):
            y_min = raw_bounds.get("y_min", -4.0)
            y_max = raw_bounds.get("y_max", 4.0)
        else:
            y_min, y_max = -4.0, 4.0

        x_min = -200.0
        x_max = 200.0

        port_coords = np.array([[-200.0, x_min], [y_min, x_min], [y_min, x_max], [-200.0, x_max]])
        ax.add_patch(
            MplPolygon(
                port_coords,
                closed=True,
                facecolor=bank_face_color,
                edgecolor=bank_edge_color,
                linewidth=1.4,
                alpha=0.75,
                zorder=1,
                label="Canal Bank",
            )
        )
        stbd_coords = np.array([[y_max, x_min], [200.0, x_min], [200.0, x_max], [y_max, x_max]])
        ax.add_patch(
            MplPolygon(
                stbd_coords,
                closed=True,
                facecolor=bank_face_color,
                edgecolor=bank_edge_color,
                linewidth=1.4,
                alpha=0.75,
                zorder=1,
            )
        )

    # -------------------------------------------------------------------------
    # 2. Extract Waypoint and Route Coordinates (NED: col 0 = North, col 1 = East)
    # -------------------------------------------------------------------------
    nom_wps = np.asarray(run_data["nominal_wps"], dtype=float)
    has_ts = "ts_pos" in run_data and len(run_data["ts_pos"]) > 0
    ts_pos = np.asarray(run_data["ts_pos"], dtype=float) if has_ts else np.empty((0, 2))

    pts_north = []
    pts_east = []

    if len(nom_wps) > 0:
        pts_north.append(nom_wps[:, 0])
        pts_east.append(nom_wps[:, 1])

    if len(ts_pos) > 0:
        pts_north.append(ts_pos[:, 0])
        pts_east.append(ts_pos[:, 1])

    all_north = np.concatenate(pts_north)
    all_east = np.concatenate(pts_east)

    # -------------------------------------------------------------------------
    # 3. Dynamic Vertical Limits
    # -------------------------------------------------------------------------
    n_min, n_max = float(np.min(all_north)), float(np.max(all_north))
    n_pad = max(2.5, 0.05 * (n_max - n_min))
    y_bottom = n_min - n_pad
    y_top = n_max + n_pad
    ax.set_ylim(y_bottom, y_top)

    # -------------------------------------------------------------------------
    # 4. Dynamic Horizontal Limits (Guaranteed Shoreline & Waypoint Visibility)
    # -------------------------------------------------------------------------
    port_inners = []
    stbd_inners = []

    # Detect edge segments passing through the active vertical window
    if polys:
        for poly in polys:
            if hasattr(poly, "exterior"):
                x_pts, y_pts = poly.exterior.xy
            else:
                poly_arr = np.asarray(poly)
                x_pts, y_pts = poly_arr[:, 0], poly_arr[:, 1]

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

    if canal_bounds is not None:
        raw_bounds = canal_bounds
        if hasattr(raw_bounds, "item"):
            try:
                raw_bounds = raw_bounds.item()
            except Exception:
                pass
        if isinstance(raw_bounds, dict):
            if "y_min" in raw_bounds:
                port_inners.append(float(raw_bounds["y_min"]))
            if "y_max" in raw_bounds:
                stbd_inners.append(float(raw_bounds["y_max"]))

    port_inner = max(port_inners) if port_inners else min(-4.0, float(np.min(all_east)) - 1.5)
    stbd_inner = min(stbd_inners) if stbd_inners else max(4.0, float(np.max(all_east)) + 1.5)

    glimpse_width = 1.5
    x_left = min(float(np.min(all_east)) - 1.0, port_inner - glimpse_width)
    x_right = max(float(np.max(all_east)) + 1.0, stbd_inner + glimpse_width)
    ax.set_xlim(x_left, x_right)

    # -------------------------------------------------------------------------
    # 5. Plot Routes & Waypoints
    # -------------------------------------------------------------------------
    # Own Ship Nominal Route
    if len(nom_wps) > 0:
        ax.plot(
            nom_wps[:, 1],
            nom_wps[:, 0],
            color="#1f77b4",
            linestyle="-",
            linewidth=2.2,
            label="OS Nominal Route",
            zorder=3,
        )

        # Waypoint Markers and Labels
        for idx, (n_wp, e_wp) in enumerate(zip(nom_wps[:, 0], nom_wps[:, 1])):
            ax.scatter(e_wp, n_wp, color="#1f77b4", s=40, zorder=5)
            ax.text(
                e_wp + 0.30,
                n_wp,
                f"$W_{{{idx+1}}}$",
                fontsize=8.5,
                fontweight="bold",
                color="#1e3a8a",
                va="center",
                zorder=6,
            )

        # OS Start & End
        ax.scatter(nom_wps[0, 1], nom_wps[0, 0], color="#1f77b4", s=50, marker="o", label="OS Start", zorder=5)
        ax.plot(
            nom_wps[-1, 1],
            nom_wps[-1, 0],
            "o",
            color="#1f77b4",
            mfc="none",
            markersize=8,
            markeredgewidth=1.8,
            label="OS End",
            zorder=5,
        )

    # Target Ship Route
    if len(ts_pos) > 0:
        stride = max(1, len(ts_pos) // 80)
        ax.plot(
            ts_pos[::stride, 1],
            ts_pos[::stride, 0],
            color="#d62728",
            linestyle="-",
            linewidth=2.0,
            label="TS Route",
            zorder=3,
        )
        ax.scatter(ts_pos[0, 1], ts_pos[0, 0], color="#d62728", s=50, marker="o", label="TS Start", zorder=5)
        ax.plot(
            ts_pos[-1, 1],
            ts_pos[-1, 0],
            "o",
            color="#d62728",
            mfc="none",
            markersize=8,
            markeredgewidth=1.8,
            label="TS End",
            zorder=5,
        )

    # -------------------------------------------------------------------------
    # 6. Formatting matching Figure 2
    # -------------------------------------------------------------------------
    ax.set_title(f"Case Definition: {sc_name.upper()}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("East (y) [m]")
    ax.set_ylabel("North (x) [m]")
    ax.grid(True, linestyle="-", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8.0, framealpha=0.92)

    fig.tight_layout()
    out_file = out_dir / f"{sc_name.lower()}_case_definition.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved dynamic case definition to: {out_file}")


def generate_all_case_definition_plots(log_dir=None):
    if log_dir is None:
        possible_dirs = [
            Path("simulation_logs"),
            Path("../simulation_logs"),
            REPO_ROOT / "simulation_logs",
            Path.cwd() / "simulation_logs",
        ]
        log_dir = next((d for d in possible_dirs if d.exists()), Path("simulation_logs"))
    else:
        log_dir = Path(log_dir)

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target_dir = OUTPUT_DIR
    except Exception:
        target_dir = Path("results_plots")
        target_dir.mkdir(parents=True, exist_ok=True)

    npz_files = glob.glob(os.path.join(log_dir, "*.npz"))
    if not npz_files:
        print(f"[ERROR] No log files found in '{log_dir.resolve()}'.")
        return

    scenario_files = {}
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            sc = str(data["scenario"]).lower()
            mode = str(data["mode"]).upper() if "mode" in data else ""
            if sc not in scenario_files or mode == "RA":
                scenario_files[sc] = f
        except Exception:
            continue

    print(f"Generating dynamic case definition plots for: {sorted(list(scenario_files.keys()))}")
    for sc_name in sorted(scenario_files.keys()):
        filepath = scenario_files[sc_name]
        data = np.load(filepath, allow_pickle=True)
        plot_single_case_definition(sc_name, data, target_dir)


if __name__ == "__main__":
    generate_all_case_definition_plots()