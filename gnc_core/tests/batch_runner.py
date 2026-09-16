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

  Latency for each update interval Delta T_IS, showing the tipping point
  crossing at Delta J = 0.
  """
  plt.figure(figsize=(8, 5))

  # Group by update interval to draw one curve per Delta T_IS
  for dt, group in df_scenario.groupby("interval"):
    sorted_group = group.sort_values(by="latency")
    plt.plot(
        sorted_group["latency"],
        sorted_group["delta_j"],
        marker="o",
        linewidth=2.0,
        label=r"$\Delta T_{\mathrm{IS}} = $" + f"{dt:.1f} s",
    )

  # Baseline reference (Delta J = 0 is RA equivalent)
  plt.axhline(
      0.0,
      color="black",
      linestyle="--",
      linewidth=1.2,
      label=r"RA Baseline ($\Delta J = 0$)",
  )
  plt.title(
      f"Latency Tipping Point Analysis ({scenario_name.upper()})",
      fontweight="bold",
  )
  plt.xlabel(r"Communication Latency $\tau$ [s]")
  plt.ylabel(r"Performance Differential $\Delta J = J_{\mathrm{IS}} - 1.0$")
  plt.grid(True, linestyle=":", alpha=0.6)
  plt.legend()
  plt.tight_layout()

  os.makedirs("results_plots", exist_ok=True)
  plt.savefig(
      f"results_plots/{scenario_name}_latency_tipping_curve.png", dpi=300
  )
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
    )

    sc_results = []
    for is_entry in is_runs:
      kpi_is = evaluator.evaluate_single_run(
        is_entry["data"]["t"],
        is_entry["data"]["os_pos"],
        is_entry["data"]["os_psi"],
        is_entry["data"]["os_r"],
        is_entry["data"]["ts_pos"],
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