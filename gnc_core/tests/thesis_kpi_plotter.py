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

# Workspace / CSV path resolution
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
    **kwargs
) -> float:
    if "rudder_or_rate" in kwargs:
        rudder_history = kwargs["rudder_or_rate"]

    effective_d_safe = d_safe - eps_tol

    # 1. Proximity Risk Factor (R_dist)
    if min_dist < effective_d_safe:
        r_dist = 1.15
    elif min_dist >= d_max:
        r_dist = 0.0
    else:
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
        return 1.15

    combined_risk = w_dist * r_dist + w_steer * r_steer
    return float(np.clip(combined_risk, 0.0, 1.15))


def plot_safety_assessment_chart(summary_df: pd.DataFrame, title_suffix: str = "", filename: str = "safety_assessment.png"):
    """Generates the 4-tier safety assessment bar chart with clean layout."""
    if summary_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)

    # 1. Four-tier Background Risk Bands
    ax.axhspan(0.00, 0.35, color='#F0FFF0', alpha=0.6)  # LOW
    ax.axhspan(0.35, 0.70, color='#FFFFE0', alpha=0.6)  # MODERATE
    ax.axhspan(0.70, 1.00, color='#FFF0F5', alpha=0.6)  # HIGH
    ax.axhspan(1.00, 1.20, color='#B22222', alpha=0.25, hatch='//')  # BREACH

    # Risk Labels aligned on the right edge
    x_pos = len(summary_df) - 0.4
    ax.text(x_pos, 0.17, "LOW RISK", color='#2E8B57', fontweight='bold', fontsize=9)
    ax.text(x_pos, 0.52, "MODERATE RISK", color='#B8860B', fontweight='bold', fontsize=9)
    ax.text(x_pos, 0.85, "HIGH RISK", color='#A52A2A', fontweight='bold', fontsize=9)
    ax.text(x_pos, 1.08, "BREACH", color='#800000', fontweight='bold', fontsize=9)

    # 2. Side-by-Side Bars
    x = np.arange(len(summary_df))
    width = 0.35
    scenarios = summary_df['scenario'].tolist()

    ax.bar(x - width/2, summary_df['R_safety_RA'], width, label='RA Baseline', color='#2b5c8f')
    ax.bar(x + width/2, summary_df['R_safety_IS'], width, label='IS (Mean across intervals)', color='#41ab5d')

    # Breach threshold line
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='Breach Threshold (R_safety >= 1.0)')

    ax.set_ylim(-0.05, 1.20)
    ax.set_xlim(-0.6, len(scenarios) - 0.2)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontweight='bold', fontsize=10)
    ax.set_ylabel(r"Safety Risk Factor $R_{\mathrm{safety}}$", fontweight='bold')
    ax.set_title(f"Safety Assessment: Proximity & Rudder Rate Risk {title_suffix}", fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout()
    out_dir = REPO_ROOT / "results_plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / filename
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved safety assessment plot to: {out_path}")


def extract_model_data(log_dir=None, d_safe=VesselParams.DCPA_safe, r_max=2.8):
    if log_dir is None:
        log_dir = REPO_ROOT / "simulation_logs"
    else:
        log_dir = Path(log_dir)

    npz_files = glob.glob(os.path.join(log_dir, "**", "*.npz"), recursive=True)
    if not npz_files:
        return {}

    runs = []
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            stem = Path(f).stem.lower()
            sc = str(data["scenario"]).lower() if "scenario" in data else stem.split("_")[0]
            mode = str(data["mode"]).upper() if "mode" in data else ("RA" if "_ra" in stem else "IS")
            runs.append({"scenario": sc, "mode": mode, "data": data})
        except Exception:
            continue

    scenarios = sorted(list(set(r["scenario"] for r in runs if "case" in r["scenario"])))
    model_data = {}

    for sc in scenarios:
        sc_runs = [r for r in runs if r["scenario"] == sc]
        ra_runs = [r for r in sc_runs if r["mode"] == "RA"]
        is_runs = [r for r in sc_runs if r["mode"] == "IS"]
        model_data[sc] = {}

        # Process RA
        if ra_runs:
            d = ra_runs[0]["data"]
            ranges = np.linalg.norm(d["ts_pos"] - d["os_pos"], axis=1)
            r_min = float(np.min(ranges))
            os_r = d["os_r"]
            if np.max(np.abs(os_r)) < 1.5:
                peak_rate = float(np.rad2deg(np.max(np.abs(os_r))))
            else:
                peak_rate = float(np.max(np.abs(os_r)))

            risk = compute_safety_risk_factors(
                min_dist=r_min, rudder_history=os_r, time_history=d["t"],
                d_safe=d_safe, d_max=2.8, eps_tol=0.05, max_rudder_rate_deg_s=90.0,
                w_dist=0.6, w_steer=0.4
            )
            model_data[sc]["RA"] = {
                "r_min": r_min, "peak_rate": peak_rate, "risk": risk
            }

        # Process IS (Average across intervals/latencies)
        if is_runs:
            r_mins, peak_rates, risks = [], [], []
            for r in is_runs:
                d = r["data"]
                ranges = np.linalg.norm(d["ts_pos"] - d["os_pos"], axis=1)
                r_min = float(np.min(ranges))
                os_r = d["os_r"]
                if np.max(np.abs(os_r)) < 1.5:
                    peak_rate = float(np.rad2deg(np.max(np.abs(os_r))))
                else:
                    peak_rate = float(np.max(np.abs(os_r)))

                risk = compute_safety_risk_factors(
                    min_dist=r_min, rudder_history=os_r, time_history=d["t"],
                    d_safe=d_safe, d_max=2.8, eps_tol=0.05, max_rudder_rate_deg_s=90.0,
                    w_dist=0.6, w_steer=0.4
                )
                r_mins.append(r_min)
                peak_rates.append(peak_rate)
                risks.append(risk)

            model_data[sc]["IS"] = {
                "r_min": float(np.mean(r_mins)),
                "peak_rate": float(np.mean(peak_rates)),
                "risk": float(np.mean(risks))
            }

    return model_data


def extract_akdag_data(csv_dir=AKDAG_CSV_DIR, d_safe=300.0, r_max_deg_s=3.0):
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        return {}

    ra_files = sorted(glob.glob(str(csv_dir / "*_RA_metrics.csv")))
    cases = [Path(f).stem.replace("_RA_metrics", "").lower() for f in ra_files]
    akdag_data = {}

    for c in cases:
        ra_csv = csv_dir / f"{c}_RA_metrics.csv"
        is_csv = csv_dir / f"{c}_IS_metrics.csv"
        if not (ra_csv.exists() and is_csv.exists()):
            continue

        df_ra = pd.read_csv(ra_csv)
        df_is = pd.read_csv(is_csv)

        df_ra = df_ra.loc[~((df_ra['ship_2_x'] == 0.0) & (df_ra['ship_2_y'] == 0.0))].copy()
        df_is = df_is.loc[~((df_is['ship_2_x'] == 0.0) & (df_is['ship_2_y'] == 0.0))].copy()

        akdag_data[c] = {}

        # RA
        dists_ra = np.sqrt((df_ra['ship_1_x'] - df_ra['ship_2_x'])**2 + (df_ra['ship_1_y'] - df_ra['ship_2_y'])**2)
        r_min_ra = float(np.min(dists_ra)) if len(dists_ra) > 0 else np.nan
        dt_ra = np.diff(df_ra['time'].to_numpy())
        dt_ra[dt_ra == 0] = 1.0
        r_yaw_ra = np.rad2deg(np.diff(np.unwrap(df_ra['ship_1_psi'].to_numpy())) / dt_ra)
        peak_rate_ra = float(np.max(np.abs(r_yaw_ra))) if len(r_yaw_ra) > 0 else 0.0
        risk_ra = compute_safety_risk_factors(
            min_dist=r_min_ra, rudder_or_rate=peak_rate_ra, d_safe=d_safe,
            d_max=3.0 * d_safe, eps_tol=0.02 * d_safe, max_rudder_rate_deg_s=r_max_deg_s,
            w_dist=0.6, w_steer=0.4
        )
        akdag_data[c]["RA"] = {"r_min": r_min_ra, "peak_rate": peak_rate_ra, "risk": risk_ra}

        # IS
        dists_is = np.sqrt((df_is['ship_1_x'] - df_is['ship_2_x'])**2 + (df_is['ship_1_y'] - df_is['ship_2_y'])**2)
        r_min_is = float(np.min(dists_is)) if len(dists_is) > 0 else np.nan
        dt_is = np.diff(df_is['time'].to_numpy())
        dt_is[dt_is == 0] = 1.0
        r_yaw_is = np.rad2deg(np.diff(np.unwrap(df_is['ship_1_psi'].to_numpy())) / dt_is)
        peak_rate_is = float(np.max(np.abs(r_yaw_is))) if len(r_yaw_is) > 0 else 0.0
        risk_is = compute_safety_risk_factors(
            min_dist=r_min_is, rudder_or_rate=peak_rate_is, d_safe=d_safe,
            d_max=3.0 * d_safe, eps_tol=0.02 * d_safe, max_rudder_rate_deg_s=r_max_deg_s,
            w_dist=0.6, w_steer=0.4
        )
        akdag_data[c]["IS"] = {"r_min": r_min_is, "peak_rate": peak_rate_is, "risk": risk_is}

    return akdag_data


def run_dynamic_batch(model_data):
    """Generates the safety assessment plot for Own Thesis Model."""
    rows = []
    for sc, modes in model_data.items():
        if "RA" in modes and "IS" in modes:
            rows.append({
                "scenario": sc.upper(),
                "R_safety_RA": modes["RA"]["risk"],
                "R_safety_IS": modes["IS"]["risk"],
            })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(by="scenario")
    plot_safety_assessment_chart(
        df,
        title_suffix="(Own Thesis Model)",
        filename="thesis_model_safety_assessment.png"
    )


def run_akdag_batch(akdag_data):
    """Generates the safety assessment plot for Akdağ IFAC Benchmark."""
    rows = []
    for sc, modes in akdag_data.items():
        if "RA" in modes and "IS" in modes:
            rows.append({
                "scenario": sc.upper(),
                "R_safety_RA": modes["RA"]["risk"],
                "R_safety_IS": modes["IS"]["risk"],
            })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values(by="scenario")
    plot_safety_assessment_chart(
        df,
        title_suffix="(Akdağ IFAC Benchmark)",
        filename="akdag_benchmark_safety_assessment.png"
    )


def generate_comparison_table(
    model_data,
    akdag_data,
    d_safe_thesis=1.0,
    d_safe_akdag=300.0,
    r_max_thesis_deg=90.0,
    r_max_akdag_deg=3.0
):
    all_scenarios = sorted(list(set(list(model_data.keys()) + list(akdag_data.keys()))))
    rows = []

    for sc in all_scenarios:
        for mode in ["RA", "IS"]:
            m_info = model_data.get(sc, {}).get(mode, {})
            a_info = akdag_data.get(sc, {}).get(mode, {})

            # 1. Non-dimensional Clearance Ratio (r_min / d_safe)
            if "r_min" in m_info and not np.isnan(m_info["r_min"]):
                m_ratio = m_info["r_min"] / d_safe_thesis
                m_clr_str = f"{m_ratio:.2f} ({m_info['r_min']:.2f}m)"
            else:
                m_clr_str = "-"

            if "r_min" in a_info and not np.isnan(a_info["r_min"]):
                a_ratio = a_info["r_min"] / d_safe_akdag
                a_clr_str = f"{a_ratio:.2f} ({a_info['r_min']:.1f}m)"
            else:
                a_clr_str = "-"

            # 2. Peak Steering Utilization (% of actuator limit + raw deg/s)
            if "peak_rate" in m_info and not np.isnan(m_info["peak_rate"]):
                m_util = (m_info["peak_rate"] / r_max_thesis_deg) * 100.0
                m_rate_str = f"{m_util:.1f}% ({m_info['peak_rate']:.1f}°/s)"
            else:
                m_rate_str = "-"

            if "peak_rate" in a_info and not np.isnan(a_info["peak_rate"]):
                a_util = (a_info["peak_rate"] / r_max_akdag_deg) * 100.0
                a_rate_str = f"{a_util:.1f}% ({a_info['peak_rate']:.2f}°/s)"
            else:
                a_rate_str = "-"

            # 3. Combined Risk Safety Factor
            m_risk = f"{m_info['risk']:.3f}" if ("risk" in m_info and not np.isnan(m_info["risk"])) else "-"
            a_risk = f"{a_info['risk']:.3f}" if ("risk" in a_info and not np.isnan(a_info["risk"])) else "-"

            rows.append({
                "Scenario": sc.upper(),
                "Mode": mode,
                "r_min/d_safe (Thesis)": m_clr_str,
                "r_min/d_safe (Akdağ)": a_clr_str,
                "Peak Turn % (Thesis)": m_rate_str,
                "Peak Turn % (Akdağ)": a_rate_str,
                "Risk SF (Thesis)": m_risk,
                "Risk SF (Akdağ)": a_risk,
            })

    df_comp = pd.DataFrame(rows)

    print("\n" + "=" * 115)
    print("                     QUALITATIVE & QUANTITATIVE BENCHMARK: THESIS MODEL vs. AKDAĞ BENCHMARK")
    print("=" * 115)
    print(df_comp.to_string(index=False))
    print("=" * 115 + "\n")

    os.makedirs(REPO_ROOT / "results_plots", exist_ok=True)
    out_csv = REPO_ROOT / "results_plots" / "thesis_vs_akdag_detailed_comparison.csv"
    df_comp.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Exported detailed comparison table to: {out_csv}")


if __name__ == "__main__":
    d_safe_thesis = 1.0
    d_safe_akdag = 300.0
    r_max_thesis = 90.0
    r_max_akdag = 3.0

    # 1. Parse both datasets
    model_data = extract_model_data(d_safe=d_safe_thesis)
    akdag_data = extract_akdag_data(d_safe=d_safe_akdag, r_max_deg_s=r_max_akdag)

    # 2. Generate side-by-side Safety Factor bar charts
    run_dynamic_batch(model_data)
    run_akdag_batch(akdag_data)

    # 3. Print and export the detailed comparison table
    generate_comparison_table(
        model_data,
        akdag_data,
        d_safe_thesis=d_safe_thesis,
        d_safe_akdag=d_safe_akdag,
        r_max_thesis_deg=r_max_thesis,
        r_max_akdag_deg=r_max_akdag
    )