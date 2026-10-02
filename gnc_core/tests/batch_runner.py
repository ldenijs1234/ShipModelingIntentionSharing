import glob
import os
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnc_core.config.vessel_params import VesselParams
from gnc_core.tests.kpi_evaluator import ScenarioKPIEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]


def compute_safety_risk_factors(
    min_dist: float,
    rudder_history: np.ndarray,
    time_history: np.ndarray,
    d_safe: float = 1.0,
    d_max: float = 2.8,
    eps_tol: float = 0.05,
    max_rudder_rate_deg_s: float = 90.0,
    w_dist: float = 0.5,
    w_steer: float = 0.5,
):
    """
    Computes normalized risk factors in [0, 1] for proximity and steering change.
    """
    effective_d_safe = d_safe - eps_tol  # 0.95m tolerance for tracking variance

    # 1. Proximity Risk Factor (R_dist)
    if min_dist < effective_d_safe:
        r_dist = float("inf")
    elif min_dist >= d_max:
        r_dist = 0.0
    else:
        # Exponential cubic decay: r_dist -> 1.0 near d_safe, 0.0 at d_max
        r_dist = float(((d_max - min_dist) / (d_max - effective_d_safe)) ** 3.0)

    # 2. Dynamic Maneuver Risk Factor (R_steer)
    if len(rudder_history) > 1 and len(time_history) > 1:
        dt = np.diff(time_history)
        d_rudder = np.diff(rudder_history)
        valid_dt = np.where(dt > 1e-4, dt, 1e-4)
        rudder_rates = np.abs(d_rudder / valid_dt)  # deg/s
        peak_rate = float(np.max(rudder_rates))
    else:
        peak_rate = 0.0

    r_steer = float(min(1.0, (peak_rate / max_rudder_rate_deg_s) ** 2.0))

    # 3. Composite Safety Risk (R_safety)
    if r_dist == float("inf"):
        r_safety = float("inf")
    else:
        r_safety = float(w_dist * r_dist + w_steer * r_steer)

    return r_dist, peak_rate, r_steer, r_safety


def plot_combined_assessment_curve(df_scenario: pd.DataFrame, scenario_name: str):
    fig, (ax_safe, ax_eff) = plt.subplots(2, 1, figsize=(10.5, 9.5), dpi=100, sharex=True)

    df = df_scenario.copy()
    df["interval_round"] = df["interval"].round(1)

    unique_intervals = np.sort(df["interval_round"].unique())
    num_intervals = len(unique_intervals)
    colors = cm.turbo(np.linspace(0.08, 0.92, num_intervals))
    color_map = {dt: colors[i] for i, dt in enumerate(unique_intervals)}

    max_tau = float(np.nanmax(df["latency"].to_numpy()))

    # =========================================================================
    # SUBFIGURE 1: Safety Assessment (Top) - Absolute Risk Scale [0, 1]
    # =========================================================================
    y_safe_breach_ceiling = 1.08  # Elevation above 1.0 for breached markers/lines

    # Qualitative zones below 1.0
    ax_safe.axhspan(0.0, 0.35, color="#f0fdf4", alpha=0.45, zorder=1)
    ax_safe.axhspan(0.35, 0.70, color="#fef9c3", alpha=0.45, zorder=1)
    ax_safe.axhspan(0.70, 1.00, color="#fee2e2", alpha=0.45, zorder=1)

    # 1. Dark-red diagonal striped hatch region for the Breach Zone (> 1.0)
    ax_safe.axhspan(
        1.00,
        1.18,
        facecolor="#7f1d1d",
        edgecolor="#991b1b",
        alpha=0.35,
        hatch="//",
        linewidth=0.8,
        zorder=1,
    )

    # Text annotations for the regions
    ax_safe.text(0.1, 0.15, "LOW RISK", fontsize=8.0, fontweight="bold", color="#166534", alpha=0.75)
    ax_safe.text(0.1, 0.50, "MODERATE RISK", fontsize=8.0, fontweight="bold", color="#854d0e", alpha=0.75)
    ax_safe.text(0.1, 0.85, "HIGH RISK", fontsize=8.0, fontweight="bold", color="#991b1b", alpha=0.75)
    ax_safe.text(
        0.1,
        1.09,
        "BREACH",
        fontsize=8.5,
        fontweight="bold",
        color="#7f1d1d",
        alpha=0.90,
    )

    # Hard boundary line at R = 1.0
    ax_safe.axhline(1.0, color="#991b1b", linestyle=":", linewidth=1.2, alpha=0.85, zorder=2)

    # 2. Check and Plot RA Baseline (Handles finite and inf values)
    ra_r_safety = df["ra_r_safety"].iloc[0] if "ra_r_safety" in df.columns else float("inf")

    if not np.isfinite(ra_r_safety):
        # Baseline is breached: display inside the striped zone at y = 1.08
        ax_safe.axhline(
            y_safe_breach_ceiling,
            color="#1f2937",
            linestyle="--",
            linewidth=1.4,
            label=r"RA Baseline ($\infty$ - BREACH)",
            zorder=3,
        )
    else:
        # Standard finite baseline
        ax_safe.axhline(
            ra_r_safety,
            color="#1f2937",
            linestyle="--",
            linewidth=1.3,
            label=rf"RA Baseline ($R_{{\mathrm{{safety}}}} = {ra_r_safety:.2f}$)",
            zorder=3,
        )

    # 3. Plot IS Data Points and Breaches
    for dt_val, group in df.groupby("interval_round"):
        sorted_group = group.sort_values(by="latency")
        tau = sorted_group["latency"].to_numpy()
        r_safety = sorted_group["r_safety"].to_numpy()
        breached = (sorted_group["r_min"] < 0.95).to_numpy() | ~np.isfinite(r_safety)
        color = color_map[dt_val]

        safe_idx = np.where(~breached)[0]
        if len(safe_idx) > 0:
            segments = np.split(safe_idx, np.where(np.diff(safe_idx) > 1)[0] + 1)
            for i, seg in enumerate(segments):
                if len(seg) > 0:
                    ax_safe.plot(
                        tau[seg],
                        r_safety[seg],
                        marker="s",
                        color=color,
                        linewidth=1.6,
                        markersize=4.5,
                        label=rf"$\Delta T_{{\mathrm{{IS}}}} = {dt_val:.1f}\,$s" if i == 0 else None,
                        zorder=4,
                    )

        if np.any(breached):
            ax_safe.scatter(
                tau[breached],
                np.full(np.sum(breached), y_safe_breach_ceiling),
                color=color,
                marker="X",
                s=75,
                edgecolor="black",
                linewidth=0.7,
                zorder=5,
            )
            for t_b in tau[breached]:
                ax_safe.vlines(
                    t_b,
                    0.0,
                    y_safe_breach_ceiling,
                    color=color,
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.65,
                    zorder=2,
                )

    ax_safe.set_title(
        f"Safety Assessment ({scenario_name.upper()}): Proximity & Rudder Rate Risk",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_safe.set_ylabel(r"Safety Risk Factor $R_{\mathrm{safety}}$", fontsize=9.5)
    ax_safe.set_ylim(-0.05, 1.18)
    ax_safe.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_safe.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax_safe.legend(loc="upper left", ncol=2, fontsize=7.5, framealpha=0.92)

    # =========================================================================
    # SUBFIGURE 2: Safety & Efficiency Assessment (Bottom)
    # =========================================================================
    y_min, y_max = -1.1, 1.1
    y_breach_ceiling = 0.55

    ax_eff.axhspan(0.0, y_max, color="#fee2e2", alpha=0.5, zorder=1)
    ax_eff.axhspan(y_min, 0.0, color="#f0fdf4", alpha=0.5, zorder=1)

    ax_eff.text(0.1, 0.04, "IS INFERIOR (Worse than RA)", fontsize=8.0, fontweight="bold", color="#991b1b", alpha=0.75)
    ax_eff.text(0.1, -0.04, "IS SUPERIOR (Better than RA)", fontsize=8.0, fontweight="bold", color="#166534", alpha=0.75, va="top")

    ax_eff.axhline(
        0.0,
        color="#1f2937",
        linestyle="--",
        linewidth=1.3,
        label=r"RA Baseline ($\Delta J = 0$)",
        zorder=2,
    )

    for dt_val, group in df.groupby("interval_round"):
        sorted_group = group.sort_values(by="latency")
        tau = sorted_group["latency"].to_numpy()
        delta_j = sorted_group["delta_j"].to_numpy()
        breached = (sorted_group["r_min"] < 0.95).to_numpy() | ~np.isfinite(delta_j)
        color = color_map[dt_val]

        safe_indices = np.where(~breached)[0]
        if len(safe_indices) > 0:
            segments = np.split(safe_indices, np.where(np.diff(safe_indices) > 1)[0] + 1)
            for i, seg in enumerate(segments):
                if len(seg) > 0:
                    ax_eff.plot(
                        tau[seg],
                        delta_j[seg],
                        marker="o",
                        color=color,
                        linewidth=1.7,
                        markersize=4.5,
                        label=rf"$\Delta T_{{\mathrm{{IS}}}} = {dt_val:.1f}\,$s" if i == 0 else None,
                        zorder=3,
                    )

        if np.any(breached):
            ax_eff.scatter(
                tau[breached],
                np.full(np.sum(breached), y_breach_ceiling),
                color=color,
                marker="X",
                s=70,
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
            )
            for t_b in tau[breached]:
                ax_eff.vlines(
                    t_b,
                    0.0,
                    y_breach_ceiling,
                    color=color,
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.65,
                    zorder=2,
                )

        # Dynamic Tipping Line per Interval
        breach_indices = np.where(breached)[0]
        if len(breach_indices) > 0:
            tip_tau = float(tau[breach_indices[0]])
            ax_eff.axvline(tip_tau, color=color, linestyle="-.", linewidth=1.0, alpha=0.6, zorder=2)
        else:
            valid_mask = np.isfinite(delta_j)
            v_tau = tau[valid_mask]
            v_dj = delta_j[valid_mask]
            sign_changes = np.where(np.diff(np.sign(v_dj)) > 0)[0]
            if len(sign_changes) > 0:
                idx = sign_changes[0]
                t0, t1 = v_tau[idx], v_tau[idx + 1]
                y0, y1 = v_dj[idx], v_dj[idx + 1]
                if abs(y1 - y0) > 1e-6:
                    tip_tau = float(t0 + (-y0) * (t1 - t0) / (y1 - y0))
                    ax_eff.axvline(tip_tau, color=color, linestyle="-.", linewidth=1.0, alpha=0.6, zorder=2)

    ax_eff.set_title(
        f"Safety & Efficiency Assessment ({scenario_name.upper()}): Latency Tipping Curve",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax_eff.set_xlabel(r"Communication Latency $\tau$ [s]", fontsize=10)
    ax_eff.set_ylabel(r"Performance Differential $\Delta J = J_{\mathrm{IS}} - 1.0$", fontsize=9.5)
    ax_eff.set_xlim(-0.3, max_tau + 0.3)
    ax_eff.set_ylim(y_min, y_max)
    ax_eff.set_xticks(np.arange(0, int(max_tau) + 1, 1))
    ax_eff.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax_eff.legend(loc="upper left", ncol=2, fontsize=7.5, framealpha=0.92)

    plt.tight_layout()
    os.makedirs("results_plots", exist_ok=True)
    out_path = f"results_plots/{scenario_name}_combined_assessment.png"
    plt.savefig(out_path, dpi=300)
    plt.show()


def run_dynamic_batch(log_dir=None, d_safe=VesselParams.DCPA_safe):
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
            "data": data,
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
        nominal_wps = ra_entry["data"]["nominal_wps"]
        evaluator = ScenarioKPIEvaluator(
            d_safe=d_safe,
            nominal_waypoints=nominal_wps,
            w_cte=1/3,
            w_ctrl=1/3,
            w_speed=1/3,
        )

        kpi_ra = evaluator.evaluate_single_run(
            ra_entry["data"]["t"],
            ra_entry["data"]["os_pos"],
            ra_entry["data"]["os_psi"],
            ra_entry["data"]["os_r"],
            ra_entry["data"]["ts_pos"],
            os_u=ra_entry["data"]["os_u"] if "os_u" in ra_entry["data"] else None,
        )

        # Baseline RA Safety Risk Factors
        ra_rudder = np.degrees(ra_entry["data"]["os_psi"])
        ra_r_dist, ra_peak_rate, ra_r_steer, ra_r_safety = compute_safety_risk_factors(
            kpi_ra["r_min"],
            ra_rudder,
            ra_entry["data"]["t"],
            d_safe=d_safe,
            max_rudder_rate_deg_s=90.0,
        )

        sc_results = []
        for is_entry in is_runs:
            kpi_is = evaluator.evaluate_single_run(
                is_entry["data"]["t"],
                is_entry["data"]["os_pos"],
                is_entry["data"]["os_psi"],
                is_entry["data"]["os_r"],
                is_entry["data"]["ts_pos"],
                os_u=is_entry["data"]["os_u"] if "os_u" in is_entry["data"] else None,
            )
            comp = evaluator.evaluate_comparison(kpi_ra, kpi_is)

            # Safety Risk Factors for IS run
            is_rudder = np.degrees(is_entry["data"]["os_psi"])
            r_dist, peak_rate, r_steer, r_safety = compute_safety_risk_factors(
                kpi_is["r_min"],
                is_rudder,
                is_entry["data"]["t"],
                d_safe=d_safe,
                max_rudder_rate_deg_s=90.0,
            )

            sc_results.append({
                "scenario": sc,
                "interval": is_entry["interval"],
                "latency": is_entry["latency"],
                "t_advance": is_entry["t_advance"],
                "r_min": kpi_is["r_min"],
                "r_dist": r_dist,
                "peak_rudder_rate": peak_rate,
                "r_steer": r_steer,
                "r_safety": r_safety,
                "ra_r_safety": ra_r_safety,
                "status": comp["status"],
                "norm_ctrl": comp.get("norm_ctrl", float("inf")),
                "norm_cte": comp.get("norm_cte", float("inf")),
                "norm_spd": comp.get("norm_speed", float("inf")),
                "j_total_is": comp.get("j_total_is", float("inf")),
                "delta_j": comp.get("delta_j", float("inf")),
                "delta_j_pct": comp.get("delta_j_pct", 0.0),
            })

        if not sc_results:
            print(f"[{sc}] No IS runs found to pair with RA baseline.")
            continue

        df_sc = pd.DataFrame(sc_results)
        df_sc["latency"] = df_sc["latency"].astype(float)
        df_sc["interval"] = df_sc["interval"].astype(float)
        df_sc["t_advance"] = df_sc["t_advance"].astype(float)
        df_sc = df_sc.sort_values(by=["interval", "latency"], ascending=[True, True]).reset_index(drop=True)

        # ---------------------------------------------------------------------
        # Table 1: Safety Evaluation
        # ---------------------------------------------------------------------
        print(f"\n================ SAFETY EVALUATION: {sc.upper()} ================")
        print(f"Baseline RA Reference: r_min={kpi_ra['r_min']:.3f}m | R_dist={ra_r_dist:.3f} | Peak_Rate={ra_peak_rate:.1f}°/s | R_steer={ra_r_steer:.3f} | R_safety={ra_r_safety:.3f}")
        print(
            df_sc[[
                "scenario",
                "interval",
                "latency",
                "r_min",
                "r_dist",
                "peak_rudder_rate",
                "r_steer",
                "r_safety",
                "status",
            ]].to_string(index=False, justify="right", formatters={
                "interval": "{:.1f}".format,
                "latency": "{:.1f}".format,
                "r_min": "{:.3f}".format,
                "r_dist": lambda x: "BREACH" if not np.isfinite(x) else f"{x:.3f}",
                "peak_rudder_rate": "{:.1f}".format,
                "r_steer": "{:.3f}".format,
                "r_safety": lambda x: "BREACH" if not np.isfinite(x) else f"{x:.3f}",
            })
        )

        # ---------------------------------------------------------------------
        # Table 2: Safety & Efficiency Evaluation
        # ---------------------------------------------------------------------
        print(f"\n=========== SAFETY AND EFFICIENCY EVALUATION: {sc.upper()} ===========")
        print(
            df_sc[[
                "scenario",
                "interval",
                "latency",
                "t_advance",
                "r_min",
                "norm_ctrl",
                "norm_cte",
                "norm_spd",
                "delta_j",
                "delta_j_pct",
            ]].to_string(index=False, justify="right", formatters={
                "interval": "{:.1f}".format,
                "latency": "{:.1f}".format,
                "t_advance": "{:.2f}".format,
                "r_min": "{:.3f}".format,
                "norm_ctrl": "{:.3f}".format,
                "norm_cte": "{:.3f}".format,
                "norm_spd": "{:.3f}".format,
                "delta_j": "{:.4f}".format,
                "delta_j_pct": "{:+.2f}%".format,
            })
        )

        # Plot combined two-panel figure
        if len(df_sc) > 1:
            plot_combined_assessment_curve(df_sc, sc)


if __name__ == "__main__":
    run_dynamic_batch(log_dir="simulation_logs", d_safe=VesselParams.DCPA_safe)