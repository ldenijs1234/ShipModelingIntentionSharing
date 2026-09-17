import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from gnc_core.config.vessel_params import VesselParams
from gnc_core.tests.kpi_evaluator import ScenarioKPIEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]

def plot_summary_latency_curve(df_scenario: pd.DataFrame, scenario_name: str):
    """Plots Delta J vs.

    Latency dynamically, detecting safety boundaries and zero-crossings
    strictly from the data without hardcoded values.
    """
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=100)

    # Use rounded interval comparison to prevent float precision drops
    df = df_scenario.copy()
    df["interval_round"] = df["interval"].round(1)
    df_dt5 = df[df["interval_round"] == 5.0].sort_values(by="latency")

    if df_dt5.empty:
        df_dt5 = df.sort_values(by="latency")

    tau = df_dt5["latency"].to_numpy()
    delta_j = df_dt5["delta_j"].to_numpy()

    # Identify safety breaches: r_min < d_safe (1.0 m) or non-finite cost
    breached = (df_dt5["r_min"] < 1.0).to_numpy() | ~np.isfinite(delta_j)

    y_min, y_max = -0.55, 0.65
    y_breach_ceiling = 0.55

    # 1. Background Qualitative Regions
    ax.axhspan(0.0, y_max, color="#fee2e2", alpha=0.5, zorder=1)
    ax.axhspan(y_min, 0.0, color="#f0fdf4", alpha=0.5, zorder=1)

    ax.text(
        0.2,
        0.04,
        "IS INFERIOR (Worse than RA)",
        fontsize=8,
        fontweight="bold",
        color="#991b1b",
        alpha=0.8,
    )
    ax.text(
        0.2,
        -0.04,
        "IS SUPERIOR (Better than RA)",
        fontsize=8,
        fontweight="bold",
        color="#166534",
        alpha=0.8,
        va="top",
    )

    # 2. RA Parity Baseline
    ax.axhline(
        0.0,
        color="#374151",
        linestyle="--",
        linewidth=1.2,
        label=r"RA Baseline ($\Delta J = 0$)",
        zorder=2,
    )

    # 3. Safe Continuous Segments (omits broken lines over breach gaps)
    safe_indices = np.where(~breached)[0]
    if len(safe_indices) > 0:
        segments = np.split(
            safe_indices, np.where(np.diff(safe_indices) > 1)[0] + 1
        )
        for i, seg in enumerate(segments):
            if len(seg) > 0:
                ax.plot(
                    tau[seg],
                    delta_j[seg],
                    marker="o",
                    color="#0284c7",
                    linewidth=1.8,
                    markersize=5,
                    label=(
                        r"$\Delta T_{\mathrm{IS}} = 5.0\,\mathrm{s}$"
                        if i == 0
                        else None
                    ),
                    zorder=3,
                )

    # 4. Safety Breaches (r_min < d_safe) Placed at Ceiling with Vertical Markers
    if np.any(breached):
        ax.scatter(
            tau[breached],
            np.full(np.sum(breached), y_breach_ceiling),
            color="#dc2626",
            marker="X",
            s=80,
            edgecolor="#7f1d1d",
            linewidth=0.8,
            label=r"Safety Breach ($r_{\mathrm{min}} < d_{\mathrm{safe}}$)",
            zorder=4,
        )
        for t_b in tau[breached]:
            ax.vlines(
                t_b,
                0.0,
                y_breach_ceiling,
                color="#dc2626",
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
                zorder=2,
            )

    # 5. Purely Dynamic Tipping Point Detection
    tipping_tau = None
    tipping_label = None

    breach_indices = np.where(breached)[0]
    if len(breach_indices) > 0:
        # Criterion A: First safety violation encountered
        first_breach_idx = breach_indices[0]
        tipping_tau = float(tau[first_breach_idx])
        tipping_label = (
            rf"Safety Boundary ($\tau = {tipping_tau:.1f}\,\mathrm{{s}}$)"
        )
    else:
        # Criterion B: Zero-crossing interpolation (from negative to positive Delta J)
        valid_mask = np.isfinite(delta_j)
        valid_tau = tau[valid_mask]
        valid_dj = delta_j[valid_mask]
        sign_changes = np.where(np.diff(np.sign(valid_dj)) > 0)[0]

        if len(sign_changes) > 0:
            idx = sign_changes[0]
            t0, t1 = valid_tau[idx], valid_tau[idx + 1]
            y0, y1 = valid_dj[idx], valid_dj[idx + 1]
            if abs(y1 - y0) > 1e-6:
                tipping_tau = float(t0 + (-y0) * (t1 - t0) / (y1 - y0))
                tipping_label = rf"Efficiency Tipping ($\tau \approx {tipping_tau:.2f}\,\mathrm{{s}}$)"

    if tipping_tau is not None:
        ax.axvline(
            tipping_tau,
            color="#b91c1c",
            linestyle="-.",
            linewidth=1.2,
            alpha=0.85,
            label=tipping_label,
            zorder=2,
        )

    # Formatting and Layout
    ax.set_title(
        f"Latency Sensitivity & Tipping Point Analysis ({scenario_name.upper()})",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel(r"Communication Latency $\tau$ [s]", fontsize=9.5)
    ax.set_ylabel(
        r"Performance Differential $\Delta J = J_{\mathrm{IS}} - 1.0$",
        fontsize=9.5,
    )
    ax.set_xlim(-0.3, float(np.nanmax(tau)) + 0.5)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.arange(0, int(np.nanmax(tau)) + 1, 1))

    ax.grid(True, linestyle=":", alpha=0.55, zorder=0)
    ax.legend(
        loc="lower left", fontsize=8, framealpha=0.9, edgecolor="#cbd5e1"
    )

    plt.tight_layout()

    os.makedirs("results_plots", exist_ok=True)
    out_path = f"results_plots/{scenario_name}_latency_clean.png"
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

  # 1. Load all archives and index by metadata
  runs = []
  for f in npz_files:
    data = np.load(f, allow_pickle=True)
    scenario = str(data["scenario"]) if "scenario" in data else "unknown"
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

  # 2. Group and evaluate per scenario
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
        w_cte=0.5,
        w_ctrl=0.25,
        w_speed=0.25,
    )

    kpi_ra = evaluator.evaluate_single_run(
        ra_entry["data"]["t"],
        ra_entry["data"]["os_pos"],
        ra_entry["data"]["os_psi"],
        ra_entry["data"]["os_r"],
        ra_entry["data"]["ts_pos"],
        os_u=ra_entry["data"]["os_u"] if "os_u" in ra_entry["data"] else None,
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

      sc_results.append({
          "scenario": sc,
          "interval": is_entry["interval"],
          "latency": is_entry["latency"],
          "r_min": kpi_is["r_min"],
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

    # Build and sort DataFrame strictly for the current scenario
    df_sc = pd.DataFrame(sc_results)
    df_sc["latency"] = df_sc["latency"].astype(float)
    df_sc["interval"] = df_sc["interval"].astype(float)
    df_sc = df_sc.sort_values(
        by=["interval", "latency"], ascending=[True, True]
    ).reset_index(drop=True)

    print(f"\n================ BATCH SUMMARY: {sc.upper()} ================")
    print(
        df_sc[[
            "scenario",
            "interval",
            "latency",
            "r_min",
            "norm_ctrl",
            "norm_cte",
            "norm_spd",
            "delta_j",
            "delta_j_pct",
        ]].to_string(index=False)
    )

    # Generate tipping curve per scenario if multiple runs exist
    if len(df_sc) > 1:
      plot_summary_latency_curve(df_sc, sc)


if __name__ == "__main__":
  run_dynamic_batch(log_dir="simulation_logs", d_safe=VesselParams.DCPA_safe)