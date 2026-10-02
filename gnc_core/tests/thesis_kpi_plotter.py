#!/usr/bin/env python3
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnc_core.config.vessel_params import VesselParams
from gnc_core.tests.kpi_evaluator import ScenarioKPIEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]
# Check local workspace folder first, then fall back to root container path
_workspace_csv_dir = Path.cwd() / "catkin_ws" / "src" / "informed_sbmpc" / "src" / "sim_results"
if not _workspace_csv_dir.exists():
    _workspace_csv_dir = REPO_ROOT.parent / "catkin_ws" / "src" / "informed_sbmpc" / "src" / "sim_results"

AKDAG_CSV_DIR = _workspace_csv_dir if _workspace_csv_dir.exists() else Path("/root/catkin_ws/src/informed_sbmpc/src/sim_results")


def compute_safety_risk_factors(
    min_dist: float,
    rudder_history: float | np.ndarray = 0.0,
    time_history: np.ndarray | None = None,
    d_safe: float = 1.0,
    d_max: float = 2.8,
    eps_tol: float = 0.05,
    max_rudder_rate_deg_s: float = 90.0,
    w_dist: float = 0.6,
    w_steer: float = 0.4,
    **kwargs  # absorbs rudder_or_rate if passed elsewhere
) -> float:
    # Handle alias if caller passed rudder_or_rate via kwargs
    if "rudder_or_rate" in kwargs:
        rudder_history = kwargs["rudder_or_rate"]

    effective_d_safe = d_safe - eps_tol  # 0.95 m

    # 1. Proximity Risk Factor (R_dist)
    if min_dist < effective_d_safe:
        r_dist = 1.15  # Capped inside the red hatched BREACH zone
    elif min_dist >= d_max:
        r_dist = 0.0
    else:
        # Exponential cubic decay
        decay_val = ((d_max - min_dist) / (d_max - effective_d_safe)) ** 3.0
        r_dist = float(np.clip(decay_val, 0.0, 0.98))

    # 2. Dynamic Maneuver Risk Factor (R_steer)
    if isinstance(rudder_history, (int, float, np.floating)):
        max_rate = float(rudder_history)
    elif time_history is not None and len(rudder_history) > 1 and len(time_history) > 1:
        dt = np.diff(time_history)
        valid_dt = np.where(dt > 1e-4, dt, 1e-4)
        max_val = np.max(np.abs(rudder_history))
        if max_val < 1.5:  # rad to deg
            d_rudder = np.rad2deg(np.diff(rudder_history))
        else:
            d_rudder = np.diff(rudder_history)
        rates = np.abs(d_rudder / valid_dt)
        max_rate = float(np.max(rates)) if len(rates) > 0 else 0.0
    else:
        max_rate = 0.0

    r_steer = float(np.clip(max_rate / max_rudder_rate_deg_s, 0.0, 1.0))

    if min_dist < effective_d_safe:
        return 1.15  # True breach: displayed cleanly at 1.15 inside hatch

    combined_risk = w_dist * r_dist + w_steer * r_steer
    return float(np.clip(combined_risk, 0.0, 1.15))


def plot_safety_assessment_chart(summary_df: pd.DataFrame, title_suffix: str = "", filename: str = "safety_assessment.png"):
    """Recreates your exact original Safety Assessment plot with 4-tier risk bands."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # 1. Background Risk Bands matching your original plot
    ax.axhspan(0.00, 0.35, color='#F0FFF0', alpha=0.6)  # LOW
    ax.axhspan(0.35, 0.70, color='#FFFFE0', alpha=0.6)  # MODERATE
    ax.axhspan(0.70, 1.00, color='#FFF0F5', alpha=0.6)  # HIGH
    ax.axhspan(1.00, 1.20, color='#B22222', alpha=0.25, hatch='//')  # BREACH

    # Risk Labels
    x_pos = len(summary_df) - 0.5
    ax.text(x_pos, 0.17, "LOW RISK", color='#2E8B57', fontweight='bold', fontsize=9)
    ax.text(x_pos, 0.52, "MODERATE RISK", color='#B8860B', fontweight='bold', fontsize=9)
    ax.text(x_pos, 0.85, "HIGH RISK", color='#A52A2A', fontweight='bold', fontsize=9)
    ax.text(x_pos, 1.08, "BREACH", color='#800000', fontweight='bold', fontsize=9)

    # 2. Bar plotting
    x = np.arange(len(summary_df))
    width = 0.35
    scenarios = summary_df['scenario'].tolist()

    ax.bar(x - width/2, summary_df['R_safety_RA'], width, label='RA Baseline', color='#2b5c8f')
    ax.bar(x + width/2, summary_df['R_safety_IS'], width, label='IS (Mean across intervals)', color='#41ab5d')

    # Breach threshold line (plain text to avoid matplotlib parser errors)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Breach Threshold (R_safety >= 1.0)')

    ax.set_ylim(-0.05, 1.20)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontweight='bold')
    ax.set_ylabel(r"Safety Risk Factor $R_{\mathrm{safety}}$", fontweight='bold')
    ax.set_title(f"Safety Assessment: Proximity & Rudder Rate Risk {title_suffix}", fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    os.makedirs(REPO_ROOT / "results_plots", exist_ok=True)
    out_path = REPO_ROOT / "results_plots" / filename
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved safety assessment plot to: {out_path}")


# ==============================================================================
# 1. EVALUATION FOR YOUR MODEL (.npz logs, averaged across intervals)
# ==============================================================================
def run_dynamic_batch(log_dir=None, d_safe=VesselParams.DCPA_safe, r_max=2.8):
    if log_dir is None:
        log_dir = REPO_ROOT / "simulation_logs"
    else:
        log_dir = Path(log_dir)

    npz_files = glob.glob(os.path.join(log_dir, "*.npz"))
    if not npz_files:
        print(f"No log files found in '{log_dir}'.")
        return

    runs = []
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        scenario = str(data["scenario"]).lower() if "scenario" in data else "unknown"
        mode = str(data["mode"]) if "mode" in data else "RA"
        latency = float(data["latency"]) if "latency" in data else 0.0
        interval = float(data["interval"]) if "interval" in data else 3.0

        runs.append({
            "filepath": f,
            "scenario": scenario,
            "mode": mode,
            "latency": latency,
            "interval": interval,
            "data": data,
        })

    scenarios = sorted(list(set(r["scenario"] for r in runs)))
    all_results = []

    for sc in scenarios:
        sc_runs = [r for r in runs if r["scenario"] == sc]
        ra_runs = [r for r in sc_runs if r["mode"] == "RA"]
        is_runs = [r for r in sc_runs if r["mode"] == "IS"]

        if not ra_runs:
            continue

        ra_entry = ra_runs[0]
        nominal_wps = ra_entry["data"]["nominal_wps"]
        evaluator = ScenarioKPIEvaluator(d_safe=d_safe, nominal_waypoints=nominal_wps, w_cte=0.6, w_ctrl=0.4, w_speed=0.0)

        # Baseline RA
        kpi_ra = evaluator.evaluate_single_run(
            ra_entry["data"]["t"],
            ra_entry["data"]["os_pos"],
            ra_entry["data"]["os_psi"],
            ra_entry["data"]["os_r"],
            ra_entry["data"]["ts_pos"],
        )
        r_safety_ra = compute_safety_risk_factors(
            min_dist=kpi_ra["r_min"],
            rudder_history=ra_entry["data"]["os_r"],
            time_history=ra_entry["data"]["t"],
            d_safe=d_safe,
            d_max=2.8,
            eps_tol=0.05,
            max_rudder_rate_deg_s=90.0,
            w_dist=0.6,
            w_steer=0.4
        )

        # All IS runs for this scenario
        for is_entry in is_runs:
            kpi_is = evaluator.evaluate_single_run(
                is_entry["data"]["t"],
                is_entry["data"]["os_pos"],
                is_entry["data"]["os_psi"],
                is_entry["data"]["os_r"],
                is_entry["data"]["ts_pos"],
            )
            comp = evaluator.evaluate_comparison(kpi_ra, kpi_is)

            r_safety_is = compute_safety_risk_factors(
                min_dist=kpi_is["r_min"],
                rudder_history=is_entry["data"]["os_r"],
                time_history=is_entry["data"]["t"],
                d_safe=d_safe,
                d_max=2.8,
                eps_tol=0.05,
                max_rudder_rate_deg_s=90.0,
                w_dist=0.6,
                w_steer=0.4
            )

            all_results.append({
                "scenario": sc.upper(),
                "interval": is_entry["interval"],
                "latency": is_entry["latency"],
                "R_safety_RA": r_safety_ra,
                "R_safety_IS": r_safety_is,
                "delta_j_pct": comp.get("delta_j_pct", 0.0),
            })

    if not all_results:
        return

    df = pd.DataFrame(all_results)
    
    # Average across all sharing intervals and latencies
    thesis_summary = df.groupby("scenario").agg({
        "R_safety_RA": "first",
        "R_safety_IS": "mean",
        "delta_j_pct": "mean"
    }).reset_index()

    plot_safety_assessment_chart(
        thesis_summary, 
        title_suffix="(Own Thesis Model)", 
        filename="thesis_model_safety_assessment.png"
    )


# ==============================================================================
# 2. EVALUATION FOR AKDAĞ'S BENCHMARK (CSVs from informed_sbmpc)
# ==============================================================================
def run_akdag_batch(csv_dir=AKDAG_CSV_DIR, d_safe=300.0, r_max_deg_s=3.0):
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        print(f"Directory not found: '{csv_dir}'")
        return

    # Auto-detect all cases present in sim_results (e.g., case01 through case08)
    ra_files = sorted(glob.glob(str(csv_dir / "*_RA_metrics.csv")))
    if not ra_files:
        print(f"No Akdağ CSV files found in '{csv_dir}'.")
        return

    cases = [Path(f).stem.replace("_RA_metrics", "") for f in ra_files]
    rows = []

    for c in cases:
        ra_csv = csv_dir / f"{c}_RA_metrics.csv"
        is_csv = csv_dir / f"{c}_IS_metrics.csv"

        if not (ra_csv.exists() and is_csv.exists()):
            continue

        df_ra = pd.read_csv(ra_csv)
        df_is = pd.read_csv(is_csv)

        # Remove dummy uninitialized rows
        df_ra = df_ra.loc[~((df_ra['ship_2_x'] == 0.0) & (df_ra['ship_2_y'] == 0.0))]
        df_is = df_is.loc[~((df_is['ship_2_x'] == 0.0) & (df_is['ship_2_y'] == 0.0))]

        # RA calculations
        dists_ra = np.sqrt((df_ra['ship_1_x'] - df_ra['ship_2_x'])**2 + (df_ra['ship_1_y'] - df_ra['ship_2_y'])**2)
        r_min_ra = float(np.min(dists_ra)) if len(dists_ra) > 0 else np.nan
        dt_ra = np.diff(df_ra['time'].to_numpy())
        dt_ra[dt_ra == 0] = 1.0
        r_yaw_ra = np.abs(np.rad2deg(np.diff(np.unwrap(df_ra['ship_1_psi'].to_numpy())) / dt_ra))
        max_r_ra = float(np.max(r_yaw_ra)) if len(r_yaw_ra) > 0 else 0.0
        r_safety_ra = compute_safety_risk_factors(
            min_dist=r_min_ra,
            rudder_or_rate=max_r_ra,
            d_safe=d_safe,
            d_max=3.0 * d_safe,
            eps_tol=0.02 * d_safe,
            max_rudder_rate_deg_s=r_max_deg_s,
            w_dist=0.6,
            w_steer=0.4
        )

        # IS calculations
        dists_is = np.sqrt((df_is['ship_1_x'] - df_is['ship_2_x'])**2 + (df_is['ship_1_y'] - df_is['ship_2_y'])**2)
        r_min_is = float(np.min(dists_is)) if len(dists_is) > 0 else np.nan
        dt_is = np.diff(df_is['time'].to_numpy())
        dt_is[dt_is == 0] = 1.0
        r_yaw_is = np.abs(np.rad2deg(np.diff(np.unwrap(df_is['ship_1_psi'].to_numpy())) / dt_is))
        max_r_is = float(np.max(r_yaw_is)) if len(r_yaw_is) > 0 else 0.0
        r_safety_is = compute_safety_risk_factors(
            min_dist=r_min_is,
            rudder_or_rate=max_r_is,
            d_safe=d_safe,
            d_max=3.0 * d_safe,
            eps_tol=0.02 * d_safe,
            max_rudder_rate_deg_s=r_max_deg_s,
            w_dist=0.6,
            w_steer=0.4
        )

        rows.append({
            "scenario": c.upper(),
            "R_safety_RA": r_safety_ra,
            "R_safety_IS": r_safety_is,
        })

    if not rows:
        print(f"No paired Akdağ CSV files found in '{csv_dir}'.")
        return

    akdag_summary = pd.DataFrame(rows)
    plot_safety_assessment_chart(
        akdag_summary,
        title_suffix="(Akdağ IFAC Benchmark)",
        filename="akdag_benchmark_safety_assessment.png"
    )


if __name__ == "__main__":
    # Evaluates your model .npz archives (averaging across interval and latency)
    run_dynamic_batch()
    # Evaluates Akdağ's CSV benchmarks in the same format
    run_akdag_batch()