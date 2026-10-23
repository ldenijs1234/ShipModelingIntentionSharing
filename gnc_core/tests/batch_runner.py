#!/usr/bin/env python3
"""
batch_runner.py
Executes batch evaluation over simulation logs using the two-tier KPI framework:
  1. Computes standardized evaluation window T = [t_0, t_w] per scenario.
  2. Evaluates Tier 1 Safety and Tier 2 Efficiency KPIs.
  3. Formulates percentage improvement curves I(tau) and identifies tipping latencies.
  4. Generates two-panel publication plots.
"""

import glob
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

from gnc_core.config.vessel_params import VesselParams
from gnc_core.tests.kpi_evaluator import ScenarioKPIEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_cutoff_time(run_data: dict, nominal_wps: np.ndarray, d_slow: float = 2.0) -> float:
    """
    Finds the time Own Ship enters the terminal deceleration zone (d_slow)
    around the final mission waypoint.
    """
    t = run_data["t"]
    os_pos = run_data["os_pos"]
    w_final = nominal_wps[-1, :2]
    dists = np.linalg.norm(os_pos - w_final, axis=1)
    slow_idx = np.where(dists <= d_slow)[0]
    if len(slow_idx) > 0:
        return float(t[slow_idx[0]])
    return float(t[-1])


def plot_combined_assessment_curve(df_scenario: pd.DataFrame, scenario_name: str):
    """
    Plots the standardized two-panel assessment:
      - Top: Tier 1 Standalone Safety Index S in [0, 1] with risk zones.
      - Bottom: Tier 2 Total Percentage Improvement I [%] relative to RA baseline.
    """
    fig, (ax_safe, ax_eff) = plt.subplots(2, 1, figsize=(10.5, 9.5), dpi=100, sharex=True)

    df = df_scenario.copy()
    df["interval_round"] = df["interval"].round(1)

    unique_intervals = np.sort(df["interval_round"].unique())
    num_intervals = len(unique_intervals)
    colors = cm.turbo(np.linspace(0.08, 0.92, num_intervals))
    color_map = {dt: colors[i] for i, dt in enumerate(unique_intervals)}

    max_tau = float(np.nanmax(df["latency"].to_numpy()))

    # =========================================================================
    # SUBFIGURE 1: Safety Assessment (Top) - Absolute Scale S in [0, 1]
    # =========================================================================
    y_safe_breach_ceiling = 1.08

    # Qualitative zones
    ax_safe.axhspan(0.0, 0.35, color="#f0fdf4", alpha=0.55, zorder=1)
    ax_safe.axhspan(0.35, 0.70, color="#fef9c3", alpha=0.55, zorder=1)
    ax_safe.axhspan(0.70, 1.00, color="#fee2e2", alpha=0.55, zorder=1)
    ax_safe.axhspan(
        1.00, 1.18, facecolor="#7f1d1d", edgecolor="#991b1b",
        alpha=0.35, hatch="//", linewidth=0.8, zorder=1
    )

    ax_safe.text(0.1, 0.15, "LOW RISK (S < 0.35)", fontsize=8.0, fontweight="bold", color="#166534", alpha=0.85)
    ax_safe.text(0.1, 0.50, "MODERATE RISK (0.35 ≤ S < 0.70)", fontsize=8.0, fontweight="bold", color="#854d0e", alpha=0.85)
    ax_safe.text(0.1, 0.85, "HIGH RISK (S ≥ 0.70)", fontsize=8.0, fontweight="bold", color="#991b1b", alpha=0.85)
    ax_safe.text(0.1, 1.09, "HARD BREACH (d_min < 0.98 m)", fontsize=8.5, fontweight="bold", color="#7f1d1d", alpha=0.95)

    ax_safe.axhline(1.0, color="#991b1b", linestyle=":", linewidth=1.2, alpha=0.85, zorder=2)

    # RA Baseline line
    ra_s = df["ra_s"].iloc[0]
    ra_breached = bool(df["ra_breached"].iloc[0])
    if ra_breached:
        ax_safe.axhline(
            y_safe_breach_ceiling, color="#1f2937", linestyle="--", linewidth=1.4,
            label=r"RA Baseline ($\mathrm{BREACH}$)", zorder=3
        )
    else:
        ax_safe.axhline(
            ra_s, color="#1f2937", linestyle="--", linewidth=1.3,
            label=rf"RA Baseline ($S = {ra_s:.2f}$)", zorder=3
        )

    # Plot IS points
    for dt_val, group in df.groupby("interval_round"):
        sorted_group = group.sort_values(by="latency")
        tau = sorted_group["latency"].to_numpy()
        s_val = sorted_group["is_s"].to_numpy()
        breached = sorted_group["is_breached"].to_numpy()
        color = color_map[dt_val]

        safe_idx = np.where(~breached)[0]
        if len(safe_idx) > 0:
            segments = np.split(safe_idx, np.where(np.diff(safe_idx) > 1)[0] + 1)
            for i, seg in enumerate(segments):
                if len(seg) > 0:
                    ax_safe.plot(
                        tau[seg], s_val[seg], marker="s", color=color,
                        linewidth=1.6, markersize=4.5,
                        label=rf"$\Delta T_{{\mathrm{{IS}}}} = {dt_val:.1f}\,$s" if i == 0 else None,
                        zorder=4,
                    )

        if np.any(breached):
            ax_safe.scatter(
                tau[breached], np.full(np.sum(breached), y_safe_breach_ceiling),
                color=color, marker="X", s=75, edgecolor="black", linewidth=0.7, zorder=5
            )

    ax_safe.set_title(
        f"Tier 1: Standalone Safety Assessment ({scenario_name.upper()})",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_safe.set_ylabel(r"Safety Index $S = \max(R_{\mathrm{dist}}, R_{\mathrm{steer}})$", fontsize=9.5)
    ax_safe.set_ylim(-0.05, 1.18)
    ax_safe.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_safe.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax_safe.legend(loc="upper left", ncol=2, fontsize=7.5, framealpha=0.92)

    # =========================================================================
    # SUBFIGURE 2: Safety & Efficiency Assessment (Bottom) - Percentage Scale I
    # =========================================================================
    y_min, y_max = -115.0, 105.0
    y_breach_floor = -105.0

    # Qualitative zones
    ax_eff.axhspan(5.0, y_max, color="#f0fdf4", alpha=0.55, zorder=1)
    ax_eff.axhspan(-5.0, 5.0, color="#f8fafc", alpha=0.75, zorder=1)
    ax_eff.axhspan(y_min, -5.0, color="#fee2e2", alpha=0.55, zorder=1)

    ax_eff.text(0.1, 15.0, "IMPROVEMENT (I > +5%)", fontsize=8.0, fontweight="bold", color="#166534", alpha=0.85)
    ax_eff.text(0.1, 0.0, "PARITY / SAME (|I| ≤ 5%)", fontsize=7.5, fontweight="bold", color="#475569", alpha=0.85, va="center")
    ax_eff.text(0.1, -15.0, "LOSS OF IMPROVEMENT (I < -5%)", fontsize=8.0, fontweight="bold", color="#991b1b", alpha=0.85, va="top")

    ax_eff.axhline(0.0, color="#1f2937", linestyle="--", linewidth=1.3, label=r"RA Baseline ($I = 0\%$)", zorder=2)
    ax_eff.axhline(5.0, color="#166534", linestyle=":", linewidth=0.9, alpha=0.6, zorder=2)
    ax_eff.axhline(-5.0, color="#991b1b", linestyle=":", linewidth=0.9, alpha=0.6, zorder=2)

    for dt_val, group in df.groupby("interval_round"):
        sorted_group = group.sort_values(by="latency")
        tau = sorted_group["latency"].to_numpy()
        i_total = sorted_group["i_total"].to_numpy()
        breached = sorted_group["is_breached"].to_numpy()
        color = color_map[dt_val]

        safe_indices = np.where(~breached)[0]
        if len(safe_indices) > 0:
            segments = np.split(safe_indices, np.where(np.diff(safe_indices) > 1)[0] + 1)
            for i, seg in enumerate(segments):
                if len(seg) > 0:
                    ax_eff.plot(
                        tau[seg], i_total[seg], marker="o", color=color,
                        linewidth=1.7, markersize=4.5,
                        label=rf"$\Delta T_{{\mathrm{{IS}}}} = {dt_val:.1f}\,$s" if i == 0 else None,
                        zorder=3,
                    )

        if np.any(breached):
            ax_eff.scatter(
                tau[breached], np.full(np.sum(breached), y_breach_floor),
                color=color, marker="X", s=70, edgecolor="black", linewidth=0.6, zorder=4
            )
            for t_b in tau[breached]:
                ax_eff.vlines(
                    t_b, y_min, y_breach_floor, color=color,
                    linestyle=":", linewidth=0.9, alpha=0.65, zorder=2
                )

        # Dynamic Tipping Line: First latency entering Loss of Improvement (I < -5% or breach)
        loss_indices = np.where(breached | (i_total < -5.0))[0]
        if len(loss_indices) > 0:
            tip_tau = float(tau[loss_indices[0]])
            ax_eff.axvline(tip_tau, color=color, linestyle="-.", linewidth=1.1, alpha=0.65, zorder=2)
            
    ax_eff.set_title(
        f"Tier 2: Total Performance Assessment ({scenario_name.upper()}): Latency Tipping Curve",
        fontsize=11, fontweight="bold", pad=8
    )
    ax_eff.set_xlabel(r"Communication Latency $\tau$ [s]", fontsize=10)
    ax_eff.set_ylabel(r"Total Improvement $I = 0.5\,I_S + 0.5\,I_E$ [\%]", fontsize=9.5)
    ax_eff.set_xlim(-0.3, max_tau + 0.3)
    ax_eff.set_ylim(y_min, y_max)
    ax_eff.set_xticks(np.arange(0, int(max_tau) + 1, 1))
    ax_eff.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax_eff.legend(loc="upper left", ncol=2, fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    os.makedirs("results_plots", exist_ok=True)
    out_path = f"results_plots/{scenario_name}_combined_assessment.png"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[SUCCESS] Saved combined assessment plot to: {out_path}")


def run_dynamic_batch(log_dir=None, d_safe=VesselParams.DCPA_safe):
    if log_dir is None:
        log_dir = REPO_ROOT / "simulation_logs"
    else:
        log_dir = Path(log_dir)

    npz_files = glob.glob(os.path.join(log_dir, "*.npz"))
    if not npz_files:
        print(f"No log files found in '{log_dir}'.")
        return

    # 1. Collect lightweight metadata without holding large arrays in memory
    runs = []
    for f in npz_files:
        with np.load(f, allow_pickle=True) as data:
            scenario = str(data["scenario"]) if "scenario" in data else "unknown"
            mode = str(data["mode"]) if "mode" in data else "RA"
            latency = float(data["latency"]) if "latency" in data else 0.0
            interval = float(data["interval"]) if "interval" in data else 3.0
            t_advance = float(data["t_advance"]) if "t_advance" in data else np.nan

            runs.append({
                "filepath": f,
                "scenario": scenario,
                "mode": mode,
                "latency": latency,
                "interval": interval,
                "t_advance": t_advance,
            })

    scenarios = sorted(list(set(r["scenario"] for r in runs)))

    for sc in scenarios:
        sc_runs = [r for r in runs if r["scenario"] == sc]
        ra_runs = [r for r in sc_runs if r["mode"] == "RA"]
        is_runs = [r for r in sc_runs if r["mode"] == "IS"]

        if not ra_runs:
            print(f"[{sc}] Skipping: No baseline RA run found.")
            continue

        ra_entry = ra_runs[0]

        # Load baseline RA file on-demand
        with np.load(ra_entry["filepath"], allow_pickle=True) as ra_data:
            nominal_wps = ra_data["nominal_wps"]

            # Load scenario group files on-demand for window cutoff calculation
            all_group_runs = ra_runs + is_runs
            group_datas = [np.load(r["filepath"], allow_pickle=True) for r in all_group_runs]
            t_cutoffs = [find_cutoff_time(d, nominal_wps, d_slow=2.0) for d in group_datas]
            t_w = float(min(t_cutoffs))

            # Close temporary open handles
            for d in group_datas:
                d.close()

            evaluator = ScenarioKPIEvaluator(
                d_safe=d_safe,
                eps_d=0.02,
                d_far_factor=3.0,
                r_cap=1.50,
                nominal_waypoints=nominal_wps,
                w_v=0.30,
                w_c=0.70,
                omega=0.60,
                delta_deadband=5.0,
            )

            # Evaluate baseline RA
            kpi_ra = evaluator.evaluate_single_run(
                ra_data["t"],
                ra_data["os_pos"],
                ra_data["os_psi"],
                ra_data["os_r"],
                ra_data["ts_pos"],
                os_u=ra_data["os_u"] if "os_u" in ra_data else None,
                t_w=t_w,
            )

        sc_results = []
        for is_entry in is_runs:
            with np.load(is_entry["filepath"], allow_pickle=True) as is_data:
                kpi_is = evaluator.evaluate_single_run(
                    is_data["t"],
                    is_data["os_pos"],
                    is_data["os_psi"],
                    is_data["os_r"],
                    is_data["ts_pos"],
                    os_u=is_data["os_u"] if "os_u" in is_data else None,
                    num_waypoints=len(is_data["evasive_wps"]) if "evasive_wps" in is_data else 0,
                    configured_interval=is_entry["interval"],
                    t_w=t_w,
                )
                comp = evaluator.evaluate_comparison(kpi_ra, kpi_is)

                sc_results.append({
                    "scenario": sc,
                    "interval": is_entry["interval"],
                    "latency": is_entry["latency"],
                    "t_advance": is_entry["t_advance"],
                    "t_w": t_w,
                    # RA Baseline
                    "ra_breached": kpi_ra["breached"],
                    "ra_dmin": kpi_ra["d_min"],
                    "ra_s": kpi_ra["s_index"],
                    "ra_class": kpi_ra["risk_class"],
                    "ra_e": kpi_ra["e_index"],
                    # IS Run
                    "is_dmin": kpi_is["d_min"],
                    "is_breached": kpi_is["breached"],
                    "is_r_dist": kpi_is["r_dist"],
                    "is_r_pk_deg": kpi_is["r_pk_deg_s"],
                    "is_r_steer": kpi_is["r_steer"],
                    "is_s": kpi_is["s_index"],
                    "is_class": kpi_is["risk_class"],
                    "is_e_spd": kpi_is["e_speed"],
                    "is_e_rte": kpi_is["e_route"],
                    "is_e": kpi_is["e_index"],
                    # Comparative Metrics
                    "i_s": comp["i_s"],
                    "i_e": comp["i_e"],
                    "i_total": comp["i_total"],
                    "outcome": comp["outcome"],
                })

        if not sc_results:
            continue

        df_sc = pd.DataFrame(sc_results)
        df_sc["latency"] = df_sc["latency"].astype(float)
        df_sc["interval"] = df_sc["interval"].astype(float)
        ra_tcpa_init = ra_entry["t_advance"]
        tcpa_header_str = f"{ra_tcpa_init:.1f}s" if not np.isnan(ra_tcpa_init) else "N/A"
        df_sc = df_sc.sort_values(by=["interval", "latency"], ascending=[True, True]).reset_index(drop=True)

        # Print tables
        print(f"\n==================== TIER 1: SAFETY EVALUATION: {sc.upper()} ====================")
        print(f"Common Window: t in [0.0, {t_w:.1f}s] | d_safe: {d_safe:.2f}m")
        print(
            f"Baseline RA: d_min={kpi_ra['d_min']:.3f}m | "
            f"R_dist={kpi_ra['r_dist']:.3f} | "
            f"R_steer={kpi_ra['r_steer']:.3f} ({kpi_ra['r_pk_deg_s']:.1f}°/s) | "
            f"S={kpi_ra['s_index']:.3f} ({kpi_ra['risk_class']})"
        )
        print("-" * 105)
        print(
            df_sc[[
                "scenario", "interval", "latency", "t_advance", "is_dmin",
                "is_r_dist", "is_r_pk_deg", "is_r_steer", "is_s", "is_class"
            ]].to_string(index=False, justify="right", formatters={
                "interval": "{:.1f}".format,
                "latency": "{:.1f}".format,
                "t_advance": lambda x: f"{x:.1f}s" if not np.isnan(x) else "N/A",
                "is_dmin": "{:.3f}".format,
                "is_r_dist": "{:.3f}".format,
                "is_r_pk_deg": "{:.1f}°/s".format,
                "is_r_steer": "{:.3f}".format,
                "is_s": "{:.3f}".format,
                "is_class": "{}".format,
            })
        )

        print(f"\n============= TIER 2: TOTAL PERFORMANCE EVALUATION: {sc.upper()} =============")
        print(f"Baseline RA: E={kpi_ra['e_index']:.3f} (Speed={kpi_ra['e_speed']:.3f}, Route={kpi_ra['e_route']:.3f})")
        print("-" * 105)
        print(
            df_sc[[
                "scenario", "interval", "latency", "is_e_spd",
                "is_e_rte", "is_e", "i_s", "i_e", "i_total", "outcome"
            ]].to_string(index=False, justify="right", formatters={
                "interval": "{:.1f}".format,
                "latency": "{:.1f}".format,
                "is_e_spd": "{:.3f}".format,
                "is_e_rte": "{:.3f}".format,
                "is_e": "{:.3f}".format,
                "i_s": "{:+.1f}%".format,
                "i_e": "{:+.1f}%".format,
                "i_total": "{:+.1f}%".format,
                "outcome": "{}".format,
            })
        )

        print(f"\n--- Tipping Latencies for {sc.upper()} (Initial TCPA / t_advance = {tcpa_header_str}) ---")
        for dt_val, group in df_sc.groupby("interval"):
            sorted_g = group.sort_values(by="latency")
            taus = sorted_g["latency"].to_numpy()
            outcomes = sorted_g["outcome"].to_numpy()

            imp_mask = [all(outcomes[: j + 1] == "Improvement") for j in range(len(outcomes))]
            tau_imp = taus[max(np.where(imp_mask)[0])] if any(imp_mask) else "None"

            loss_idx = np.where(outcomes == "Loss of Improvement")[0]
            tau_loss = f"{taus[loss_idx[0]]:.1f}s" if len(loss_idx) > 0 else "None"

            print(f"  ΔT_IS = {dt_val:.1f}s -> tau*_imp = {tau_imp if isinstance(tau_imp, str) else f'{tau_imp:.1f}s'}, tau*_loss = {tau_loss}")

        # Render figure
        if len(df_sc) > 1:
            plot_combined_assessment_curve(df_sc, sc)

        # Force figure cleanup and garbage collection per scenario
        plt.close('all')
        gc.collect()


if __name__ == "__main__":
    run_dynamic_batch(log_dir="simulation_logs", d_safe=VesselParams.DCPA_safe)