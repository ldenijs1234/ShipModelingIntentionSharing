#!/usr/bin/env python3
"""
plot_case_definitions.py
Directly imports ground-truth scenario configs from gnc_core.imazu_cases.scenario_loader.
Plots nominal waypoint routes (OS & TS) and any dynamic mid-run intent updates
(e.g., Case 07 revised plan as a dotted line) in the NED frame.

Saves figures directly to results_plots/.
"""

import sys
from pathlib import Path
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt
import numpy as np

# Ensure workspace root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gnc_core.imazu_cases.scenario_loader import load_scenario

try:
    from shapely.geometry import MultiPolygon, Polygon
except ImportError:
    Polygon, MultiPolygon = None, None

OUTPUT_DIR = Path("results_plots")


def plot_case(case_name: str, out_dir: Path):
    cfg = load_scenario(case_name)

    fig, ax = plt.subplots(figsize=(6, 9), dpi=300)

    bank_edge_color = '#8c564b'
    bank_face_color = '#ebdcb9'
    bank_labeled = False

    # -------------------------------------------------------------------------
    # 1. Dynamic Canal Banks
    # -------------------------------------------------------------------------
    polys = []
    canal_poly = cfg.get("canal_polygons", None)
    canal_bounds = cfg.get("canal_bounds", None)

    if canal_poly is not None:
        if Polygon is not None and isinstance(canal_poly, Polygon):
            polys = [canal_poly]
        elif (MultiPolygon is not None and isinstance(canal_poly, MultiPolygon)) or hasattr(canal_poly, "geoms"):
            polys = list(canal_poly.geoms)
        elif hasattr(canal_poly, "exterior"):
            polys = [canal_poly]
        elif isinstance(canal_poly, (list, tuple)):
            polys = list(canal_poly)

    if polys:
        for poly in polys:
            x_pts, y_pts = poly.exterior.xy  # x = North, y = East
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
        y_min = canal_bounds.get("y_min", -3.0)
        y_max = canal_bounds.get("y_max", 3.0)
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
    # 2. Extract Waypoint Arrays (NED: col 0 = North, col 1 = East)
    # -------------------------------------------------------------------------
    os_wps = np.asarray(cfg["os_mission_wps"], dtype=float)
    ts_wps = np.asarray(cfg["ts_mission_wps"], dtype=float)

    # Check for mid-run dynamic update event (Case 07)
    dynamic_event = cfg.get("dynamic_event", None)
    has_revised_ts = (
        dynamic_event is not None
        and isinstance(dynamic_event, dict)
        and "revised_ts_wps" in dynamic_event
        and dynamic_event["revised_ts_wps"] is not None
    )
    rev_ts_wps = (
        np.asarray(dynamic_event["revised_ts_wps"], dtype=float)
        if has_revised_ts
        else np.empty((0, 2))
    )

    pts_north = [os_wps[:, 0], ts_wps[:, 0]]
    pts_east = [os_wps[:, 1], ts_wps[:, 1]]
    if has_revised_ts and len(rev_ts_wps) > 0:
        pts_north.append(rev_ts_wps[:, 0])
        pts_east.append(rev_ts_wps[:, 1])

    all_north = np.concatenate(pts_north)
    all_east = np.concatenate(pts_east)

    # -------------------------------------------------------------------------
    # 3. Dynamic Vertical Limits
    # -------------------------------------------------------------------------
    n_min, n_max = float(np.min(all_north)), float(np.max(all_north))
    n_pad = max(2.5, 0.08 * (n_max - n_min))
    y_bottom = n_min - n_pad
    y_top = n_max + n_pad
    ax.set_ylim(y_bottom, y_top)

    # -------------------------------------------------------------------------
    # 4. Dynamic Horizontal Limits (Shoreline Edge Scanning)
    # -------------------------------------------------------------------------
    port_inners = []
    stbd_inners = []

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

    if canal_bounds is not None:
        if "y_min" in canal_bounds:
            port_inners.append(float(canal_bounds["y_min"]))
        if "y_max" in canal_bounds:
            stbd_inners.append(float(canal_bounds["y_max"]))

    half_w = float(cfg.get("canal_width", 6.0)) / 2.0
    port_inner = max(port_inners) if port_inners else min(-half_w, float(np.min(all_east)) - 1.5)
    stbd_inner = min(stbd_inners) if stbd_inners else max(half_w, float(np.max(all_east)) + 1.5)

    glimpse_width = 1.5
    x_left = min(float(np.min(all_east)) - 1.2, port_inner - glimpse_width)
    x_right = max(float(np.max(all_east)) + 1.2, stbd_inner + glimpse_width)
    ax.set_xlim(x_left, x_right)

    # -------------------------------------------------------------------------
    # 5. Plot Own Ship Waypoint Route
    # -------------------------------------------------------------------------
    ax.plot(
        os_wps[:, 1],
        os_wps[:, 0],
        color="#1f77b4",
        linestyle="-",
        linewidth=2.2,
        label="OS Nominal Route",
        zorder=3,
    )

    for idx, (n_val, e_val) in enumerate(zip(os_wps[:, 0], os_wps[:, 1])):
        ax.scatter(e_val, n_val, color="#1f77b4", s=35, zorder=5)
        ax.text(
            e_val + 0.28,
            n_val,
            f"$W_{{{idx+1}}}$",
            fontsize=8.5,
            fontweight="bold",
            color="#1e3a8a",
            va="center",
            ha="left",
            zorder=6,
        )

    ax.scatter(os_wps[0, 1], os_wps[0, 0], color="#1f77b4", s=50, marker="o", label="OS Start", zorder=5)
    ax.plot(
        os_wps[-1, 1],
        os_wps[-1, 0],
        "o",
        color="#1f77b4",
        mfc="none",
        markersize=8,
        markeredgewidth=1.8,
        label="OS End",
        zorder=5,
    )

    # -------------------------------------------------------------------------
    # 6. Plot Target Ship Initial Route
    # -------------------------------------------------------------------------
    ax.plot(
        ts_wps[:, 1],
        ts_wps[:, 0],
        color="#d62728",
        linestyle="-",
        linewidth=2.2,
        label="TS Initial Intent" if has_revised_ts else "TS Route",
        zorder=3,
    )

    for idx, (n_val, e_val) in enumerate(zip(ts_wps[:, 0], ts_wps[:, 1])):
        ax.scatter(e_val, n_val, color="#d62728", s=35, zorder=5)
        ax.text(
            e_val - 0.28,
            n_val,
            f"$W_{{{idx+1}}}$",
            fontsize=8.5,
            fontweight="bold",
            color="#991b1b",
            va="center",
            ha="right",
            zorder=6,
        )

    ax.scatter(ts_wps[0, 1], ts_wps[0, 0], color="#d62728", s=50, marker="o", label="TS Start", zorder=5)
    ax.plot(
        ts_wps[-1, 1],
        ts_wps[-1, 0],
        "o",
        color="#d62728",
        mfc="none",
        markersize=8,
        markeredgewidth=1.8,
        label="TS End (Initial)" if has_revised_ts else "TS End",
        zorder=5,
    )

    # -------------------------------------------------------------------------
    # 7. Plot Target Ship Revised Plan (Dotted Line)
    # -------------------------------------------------------------------------
    if has_revised_ts and len(rev_ts_wps) > 0:
        ax.plot(
            rev_ts_wps[:, 1],
            rev_ts_wps[:, 0],
            color="#d62728",
            linestyle=":",
            linewidth=2.5,
            label="TS Updated Plan",
            zorder=4,
        )

        for idx, (n_val, e_val) in enumerate(zip(rev_ts_wps[:, 0], rev_ts_wps[:, 1])):
            is_duplicate = np.any(np.all(np.isclose(ts_wps, [n_val, e_val], atol=1e-2), axis=1))
            ax.scatter(e_val, n_val, color="#d62728", s=40, marker="d", zorder=5)
            if not is_duplicate:
                # Place label to avoid overlapping with OS nominal line
                ax.text(
                    e_val - 0.28,
                    n_val,
                    f"$W'_{{{idx+2}}}$",
                    fontsize=8.5,
                    fontweight="bold",
                    color="#991b1b",
                    va="center",
                    ha="right",
                    zorder=6,
                )

        ax.plot(
            rev_ts_wps[-1, 1],
            rev_ts_wps[-1, 0],
            "o",
            color="#d62728",
            mfc="none",
            markersize=8,
            markeredgewidth=1.8,
            linestyle=":",
            zorder=5,
        )

    # -------------------------------------------------------------------------
    # 8. Titles & Legend
    # -------------------------------------------------------------------------
    ax.set_title(f"Case Definition: {case_name.upper()}", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("East (y) [m]")
    ax.set_ylabel("North (x) [m]")
    ax.grid(True, linestyle="-", alpha=0.5)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    fig.tight_layout()
    out_file = out_dir / f"{case_name.lower()}_case_definition.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Opgeslagen in: {out_file}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Opslaan in: {OUTPUT_DIR.resolve()}")

    cases = [f"case{i:02d}" for i in range(1, 8)]
    for c in cases:
        try:
            plot_case(c, OUTPUT_DIR)
        except Exception as e:
            print(f"[ERROR] Fout bij genereren van {c}: {e}")


if __name__ == "__main__":
    main()