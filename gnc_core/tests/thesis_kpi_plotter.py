#!/usr/bin/env python3
import glob
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
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


def extract_model_data(log_dir=None, d_safe=VesselParams.DCPA_safe, L_os=VesselParams.L):
    """
    Extracts trajectory features from Own Thesis Model (.npz simulation logs).
    Computes turn direction, peak course alteration, dimensionless yaw rate,
    normalized lateral deviation, and minimum separation distance.
    """
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
        model_data[sc] = {}

        for mode in ["RA", "IS"]:
            matching = [r for r in sc_runs if r["mode"] == mode]
            if not matching:
                continue

            # In IS mode, average over multi-interval batch logs if present
            r_mins, peak_dchis, peak_r_primes, peak_ey_Ls = [], [], [], []
            initial_turns = []

            for run in matching:
                d = run["data"]
                os_pos = d["os_pos"]
                ts_pos = d["ts_pos"]
                t = d["t"]

                # 1. Minimum Obstacle Distance
                ranges = np.linalg.norm(ts_pos - os_pos, axis=1)
                r_min = float(np.min(ranges))
                r_mins.append(r_min)

                # 2. Heading Extraction & Initial Turn Direction
                if "os_psi" in d:
                    psi = np.unwrap(d["os_psi"])
                elif "os_states" in d and d["os_states"].shape[1] >= 3:
                    psi = np.unwrap(d["os_states"][:, 2])
                else:
                    diffs = np.diff(os_pos, axis=0)
                    psi = np.unwrap(np.arctan2(diffs[:, 1], diffs[:, 0]))
                    psi = np.pad(psi, (0, 1), mode='edge')

                d_chi = np.rad2deg(psi - psi[0])
                peak_dchi = float(np.max(np.abs(d_chi)))
                peak_dchis.append(peak_dchi)

                # Initial Turn Side: Sign of first significant deviation (> 1.5 deg or > 0.05m y)
                turn_idx = np.where(np.abs(d_chi) > 1.5)[0]
                if len(turn_idx) > 0:
                    turn_side = "S" if d_chi[turn_idx[0]] > 0 else "P"
                else:
                    y_dev = os_pos[:, 1] - os_pos[0, 1]
                    lat_idx = np.where(np.abs(y_dev) > 0.05)[0]
                    turn_side = ("S" if y_dev[lat_idx[0]] > 0 else "P") if len(lat_idx) > 0 else "None"
                initial_turns.append(turn_side)

                # 3. Dimensionless Yaw Rate r' = (r * L) / u
                os_r = d["os_r"]
                r_rad_s = os_r if np.max(np.abs(os_r)) < 1.5 else np.deg2rad(os_r)
                dt = np.diff(t)
                valid_dt = np.where(dt > 1e-4, dt, 1e-4)
                velocities = np.linalg.norm(np.diff(os_pos, axis=0), axis=1) / valid_dt
                u_mean = float(np.mean(velocities[velocities > 0.05])) if np.any(velocities > 0.05) else 0.45
                r_prime = np.abs(r_rad_s) * L_os / max(u_mean, 0.05)
                peak_r_primes.append(float(np.max(r_prime)))

                # 4. Dimensionless Lateral Deviation ey / L
                ey = np.abs(os_pos[:, 1] - os_pos[0, 1])
                peak_ey_Ls.append(float(np.max(ey) / L_os))

            # Store aggregated case metrics
            model_data[sc][mode] = {
                "initial_turn": max(set(initial_turns), key=initial_turns.count) if initial_turns else "None",
                "r_min": float(np.mean(r_mins)),
                "d_safe": float(d_safe),
                "r_min_ratio": float(np.mean(r_mins)) / float(d_safe),
                "peak_dchi": float(np.mean(peak_dchis)),
                "peak_r_prime": float(np.mean(peak_r_primes)),
                "peak_ey_L": float(np.mean(peak_ey_Ls)),
            }

    return model_data


def extract_akdag_data(csv_dir=AKDAG_CSV_DIR, d_safe=300.0, L_akdag=110.0, u_nominal=5.0):
    """
    Extracts trajectory features from Akdağ (2022) IFAC benchmark CSV logs.
    """
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

        akdag_data[c] = {}

        for mode, fpath in [("RA", ra_csv), ("IS", is_csv)]:
            df = pd.read_csv(fpath)
            # Remove zero padding / stationary rows
            df = df.loc[~((df['ship_1_x'] == 0.0) & (df['ship_1_y'] == 0.0))].copy()
            if df.empty:
                continue

            # 1. Minimum Separation Distance
            dists = np.sqrt((df['ship_1_x'] - df['ship_2_x'])**2 + (df['ship_1_y'] - df['ship_2_y'])**2)
            r_min = float(np.min(dists)) if len(dists) > 0 else np.nan

            # 2. Heading Deviation & Initial Turn Side
            psi = np.unwrap(df['ship_1_psi'].to_numpy())
            d_chi = np.rad2deg(psi - psi[0])
            peak_dchi = float(np.max(np.abs(d_chi)))

            turn_idx = np.where(np.abs(d_chi) > 1.0)[0]
            if len(turn_idx) > 0:
                turn_side = "S" if d_chi[turn_idx[0]] > 0 else "P"
            else:
                y_dev = df['ship_1_y'].to_numpy() - df['ship_1_y'].iloc[0]
                lat_idx = np.where(np.abs(y_dev) > 2.0)[0]
                turn_side = ("S" if y_dev[lat_idx[0]] > 0 else "P") if len(lat_idx) > 0 else "None"

            # 3. Dimensionless Yaw Rate r' = (r * L) / u
            t = df['time'].to_numpy()
            dt = np.diff(t)
            dt[dt == 0] = 1.0
            r_rad_s = np.diff(psi) / dt
            r_prime = np.abs(r_rad_s) * L_akdag / u_nominal
            peak_r_prime = float(np.max(r_prime)) if len(r_prime) > 0 else 0.0

            # 4. Dimensionless Lateral Deviation ey / L
            ey = np.abs(df['ship_1_y'].to_numpy() - df['ship_1_y'].iloc[0])
            peak_ey_L = float(np.max(ey) / L_akdag)

            akdag_data[c][mode] = {
                "initial_turn": turn_side,
                "r_min": r_min,
                "d_safe": float(d_safe),
                "r_min_ratio": r_min / float(d_safe),
                "peak_dchi": peak_dchi,
                "peak_r_prime": peak_r_prime,
                "peak_ey_L": peak_ey_L,
            }

    return akdag_data


def generate_protocol_tables(model_data, akdag_data):
    """
    Constructs, displays, and exports Table A (Qualitative Signature)
    and Table B (Scale-Free Relative Effects).
    """
    out_dir = REPO_ROOT / "results_plots"
    os.makedirs(out_dir, exist_ok=True)

    all_cases = sorted(list(set(list(model_data.keys()) + list(akdag_data.keys()))))
    cases = [c for c in all_cases if "case" in c]

    # =========================================================================
    # TABLE A: QUALITATIVE SIGNATURE VALIDATION MATRIX
    # =========================================================================
    rows_a = []
    latex_rows_a = []

    for c in cases:
        m = model_data.get(c, {})
        a = akdag_data.get(c, {})

        m_ra, m_is = m.get("RA", {}), m.get("IS", {})
        a_ra, a_is = a.get("RA", {}), a.get("IS", {})

        # 1. Turn Sides
        a_turn_ra = a_ra.get("initial_turn", "-")
        m_turn_ra = m_ra.get("initial_turn", "-")
        a_turn_is = a_is.get("initial_turn", "-")
        m_turn_is = m_is.get("initial_turn", "-")

        # 2. Magnitude Reduction: |chi|_IS < |chi|_RA
        a_mag = ("PASS" if a_is.get("peak_dchi", 0.0) < a_ra.get("peak_dchi", 1.0) else "FAIL") if (a_is and a_ra) else "-"
        m_mag = ("PASS" if m_is.get("peak_dchi", 0.0) < m_ra.get("peak_dchi", 1.0) else "FAIL") if (m_is and m_ra) else "-"

        # 3. Safety Clearance: CPA_IS >= d_safe
        a_cpa = ("PASS" if a_is.get("r_min_ratio", 0.0) >= 0.98 else "FAIL") if a_is else "-"
        m_cpa = ("PASS" if m_is.get("r_min_ratio", 0.0) >= 0.98 else "FAIL") if m_is else "-"

        # Qualitative Agreement Scoring
        match_turn = (a_turn_ra == m_turn_ra) and (a_turn_is == m_turn_is) and (a_turn_ra != "-")
        match_mag = (a_mag == m_mag) and (a_mag != "-")
        match_cpa = (a_cpa == m_cpa) and (a_cpa != "-")
        overall = "AGREE" if (match_turn and match_mag and match_cpa) else ("PARTIAL" if (match_turn or match_mag) else "DIVERGE")

        rows_a.append({
            "Case": c.upper(),
            "Turn RA (Akd)": a_turn_ra,
            "Turn RA (Sur)": m_turn_ra,
            "Turn IS (Akd)": a_turn_is,
            "Turn IS (Sur)": m_turn_is,
            "|χ|_IS < |χ|_RA (Akd)": a_mag,
            "|χ|_IS < |χ|_RA (Sur)": m_mag,
            "CPA >= d_safe (Akd)": a_cpa,
            "CPA >= d_safe (Sur)": m_cpa,
            "Verdict": overall
        })

        def to_latex_mark(val):
            if val == "PASS": return r"\checkmark"
            if val == "FAIL": return r"$\times$"
            return val

        latex_rows_a.append(
            f"{c.upper()} & {a_turn_ra} & {m_turn_ra} & {a_turn_is} & {m_turn_is} & "
            f"{to_latex_mark(a_mag)} & {to_latex_mark(m_mag)} & "
            f"{to_latex_mark(a_cpa)} & {to_latex_mark(m_cpa)} & {overall} \\\\"
        )

    df_a = pd.DataFrame(rows_a)

    print("\n" + "=" * 115)
    print("                    PROTOCOL A: QUALITATIVE SIGNATURE VALIDATION MATRIX")
    print("=" * 115)
    print(df_a.to_string(index=False))
    print("=" * 115 + "\n")

    df_a.to_csv(out_dir / "protocol_a_signature_validation.csv", index=False)

    # Export Table A LaTeX
    tex_a = r"""\begin{table}[htbp]
\centering
\small
\caption{Protocol A: Qualitative Signature Validation Matrix comparing the discrete surrogate SB-MPC against the continuous benchmark by Akda\u{g} (2022).}
\label{tab:protocol_a_signature}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l c c c c c c c c c @{}}
\toprule
\multirow{2}{*}{\textbf{Case}} & \multicolumn{2}{c}{\textbf{Turn Side (RA)}} & \multicolumn{2}{c}{\textbf{Turn Side (IS)}} & \multicolumn{2}{c}{\textbf{$|\chi|_{\text{IS}} < |\chi|_{\text{RA}}$}} & \multicolumn{2}{c}{\textbf{$\text{CPA}_{\text{IS}} \ge d_{\text{safe}}$}} & \multirow{2}{*}{\textbf{Verdict}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
& \textbf{Akda\u{g}} & \textbf{Surrogate} & \textbf{Akda\u{g}} & \textbf{Surrogate} & \textbf{Akda\u{g}} & \textbf{Surrogate} & \textbf{Akda\u{g}} & \textbf{Surrogate} & \\
\midrule
""" + "\n".join(latex_rows_a) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
    \footnotesize
    \item \textit{Note:} Turn directions are designated Port (P) or Starboard (S). Checks (\checkmark) denote satisfied boundary conditions.
\end{tablenotes}
\end{table}
"""
    with open(out_dir / "protocol_a_signature.tex", "w") as f:
        f.write(tex_a)

    # =========================================================================
    # TABLE B: SCALE-FREE RELATIVE EFFECTS (IS / RA RATIOS)
    # =========================================================================
    rows_b = []
    latex_rows_b = []

    for c in cases:
        m = model_data.get(c, {})
        a = akdag_data.get(c, {})

        m_ra, m_is = m.get("RA", {}), m.get("IS", {})
        a_ra, a_is = a.get("RA", {}), a.get("IS", {})

        def safe_ratio(num_dict, den_dict, key):
            if key in num_dict and key in den_dict and den_dict[key] > 1e-4:
                return num_dict[key] / den_dict[key]
            return np.nan

        # 1. Peak Heading Ratio
        a_r_dchi = safe_ratio(a_is, a_ra, "peak_dchi")
        m_r_dchi = safe_ratio(m_is, m_ra, "peak_dchi")

        # 2. Peak Dimensionless Yaw Rate Ratio
        a_r_rprime = safe_ratio(a_is, a_ra, "peak_r_prime")
        m_r_rprime = safe_ratio(m_is, m_ra, "peak_r_prime")

        # 3. Peak Lateral Displacement / L Ratio
        a_r_ey = safe_ratio(a_is, a_ra, "peak_ey_L")
        m_r_ey = safe_ratio(m_is, m_ra, "peak_ey_L")

        # 4. Normalized Safety Margin Ratio (r_min / d_safe)
        a_r_margin = safe_ratio(a_is, a_ra, "r_min_ratio")
        m_r_margin = safe_ratio(m_is, m_ra, "r_min_ratio")

        def fmt(v):
            return f"{v:.2f}" if not np.isnan(v) else "-"

        rows_b.append({
            "Case": c.upper(),
            "|Δχ| Ratio (Akd)": fmt(a_r_dchi),
            "|Δχ| Ratio (Sur)": fmt(m_r_dchi),
            "r' Ratio (Akd)": fmt(a_r_rprime),
            "r' Ratio (Sur)": fmt(m_r_rprime),
            "ey/L Ratio (Akd)": fmt(a_r_ey),
            "ey/L Ratio (Sur)": fmt(m_r_ey),
            "Margin Ratio (Akd)": fmt(a_r_margin),
            "Margin Ratio (Sur)": fmt(m_r_margin),
        })

        latex_rows_b.append(
            f"{c.upper()} & {fmt(a_r_dchi)} & {fmt(m_r_dchi)} & {fmt(a_r_rprime)} & {fmt(m_r_rprime)} & "
            f"{fmt(a_r_ey)} & {fmt(m_r_ey)} & {fmt(a_r_margin)} & {fmt(m_r_margin)} \\\\"
        )

    df_b = pd.DataFrame(rows_b)

    print("\n" + "=" * 115)
    print("                    PROTOCOL B: SCALE-FREE RELATIVE EFFECTS (IS / RA RATIOS)")
    print("=" * 115)
    print(df_b.to_string(index=False))
    print("=" * 115 + "\n")

    df_b.to_csv(out_dir / "protocol_b_relative_effects.csv", index=False)

    # Export Table B LaTeX
    tex_b = r"""\begin{table}[htbp]
\centering
\small
\caption{Protocol B: Scale-Free Relative Effect Ratios ($\text{IS}/\text{RA}$) evaluating internal model sensitivity and structural agreement between models.}
\label{tab:protocol_b_relative_ratios}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l c c c c c c c c @{}}
\toprule
\multirow{2}{*}{\textbf{Case}} & \multicolumn{2}{c}{\textbf{Peak $|\Delta\chi|$ Ratio}} & \multicolumn{2}{c}{\textbf{Peak $|r'|$ Ratio}} & \multicolumn{2}{c}{\textbf{Peak $|e_y/L|$ Ratio}} & \multicolumn{2}{c}{\textbf{Margin ($R_{\min}/d_{\text{safe}}$) Ratio}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
& \textbf{Akda\u{g}} & \textbf{Surrogate} & \textbf{Akda\u{g}} & \textbf{Surrogate} & \textbf{Akda\u{g}} & \textbf{Surrogate} & \textbf{Akda\u{g}} & \textbf{Surrogate} \\
\midrule
""" + "\n".join(latex_rows_b) + r"""
\bottomrule
\end{tabular*}
\begin{tablenotes}
    \footnotesize
    \item \textit{Note:} Relative effects are evaluated within each model as $\text{Metric}_{\text{IS}} / \text{Metric}_{\text{RA}}$. Ratios $< 1.0$ indicate performance reductions (smoother maneuvers or smaller fairway excursions) enabled by Intent-Sharing.
\end{tablenotes}
\end{table}
"""
    with open(out_dir / "protocol_b_relative_effects.tex", "w") as f:
        f.write(tex_b)

    print(f"[SUCCESS] Exported Protocol A & B tables to: {out_dir}")


if __name__ == "__main__":
    d_safe_thesis = VesselParams.DCPA_safe
    d_safe_akdag = 300.0
    L_thesis = VesselParams.L
    L_akdag = 110.0

    # 1. Parse both datasets
    model_data = extract_model_data(d_safe=d_safe_thesis, L_os=L_thesis)
    akdag_data = extract_akdag_data(d_safe=d_safe_akdag, L_akdag=L_akdag, u_nominal=5.0)

    # 2. Protocol Validation Tables A & B
    generate_protocol_tables(model_data, akdag_data)

