"""
kpi_evaluator.py
Implements the standardized two-tier Key Performance Indicator (KPI) framework:
  - Tier 1: Standalone Safety Assessment (R_dist, R_steer, S, Risk Class)
  - Tier 2: Efficiency and Total Assessment (E_speed, E_route, E, Percentage Improvement I)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


def compute_asm_telecom_kpis(num_waypoints: int, configured_interval: float) -> dict:
    """
    Computes exact bit length, payload bytes, ITU slot occupancy,
    and regulatory compliance according to the ASM schema and ITU-R M.1371-6.
    """
    n_wp = int(np.clip(num_waypoints, 0, 7))

    bits_standard_frame = 6 + 2 + 30 + 2  # 40 bits (Msg 8 header)
    bits_app_header = 16 + 3 + 2 + 17 + 3  # 41 bits (IAI through Num_WP)

    if n_wp == 0:
        total_bits = bits_standard_frame + bits_app_header
    else:
        bits_wp1 = 27 + 26 + 9 + 10  # 72 bits
        bits_subsequent = (n_wp - 1) * (12 + 12 + 9 + 10)  # 53 bits each
        total_bits = bits_standard_frame + bits_app_header + bits_wp1 + bits_subsequent

    payload_bytes = int(np.ceil((total_bits - bits_standard_frame) / 8.0))

    if payload_bytes <= 12:
        slots = 1
    elif payload_bytes <= 40:
        slots = 2
    elif payload_bytes <= 68:
        slots = 3
    else:
        slots = int(np.ceil(payload_bytes / 28.0))

    min_compliant_interval = (slots * 3.0) / np.sqrt(30)
    configured_interval_full = configured_interval * np.sqrt(30)
    actual_slots_per_min = (
        (slots * 60.0) / configured_interval_full if configured_interval_full > 0.0 else 0.0
    )
    is_vdl_compliant = (actual_slots_per_min <= 20.0) and (slots <= 3)

    return {
        "asm_num_wp": n_wp,
        "asm_total_bits": total_bits,
        "asm_payload_bytes": payload_bytes,
        "asm_slots": slots,
        "asm_min_interval_s": min_compliant_interval,
        "asm_slots_per_min": actual_slots_per_min,
        "asm_compliant": is_vdl_compliant,
    }


def resolve_effective_interval(
    input_interval: float, num_waypoints: int
) -> tuple[float, float]:
    telecom = compute_asm_telecom_kpis(
        num_waypoints=num_waypoints, configured_interval=input_interval
    )
    min_compliant = telecom["asm_min_interval_s"]
    effective_interval = max(float(input_interval), float(min_compliant))
    return effective_interval, min_compliant


class ScenarioKPIEvaluator:
    """
    Standardized Two-Tier KPI Evaluator for Inland Waterway Collision Avoidance.
    """

    def __init__(
        self,
        d_safe: float = 1.0,
        eps_d: float = 0.02,
        d_far_factor: float = 3.0,
        r_cap: float = 1.50,  # rad/s (~86.0 deg/s maximum achievable steady-state yaw rate)
        nominal_waypoints: Optional[np.ndarray] = None,
        u_nominal: float = 0.45,
        w_v: float = 0.30,
        w_c: float = 0.70,
        omega: float = 0.50,
        delta_deadband: float = 5.0,  # % percentage point deadband
        s_floor: float = 0.10,
        e_floor: float = 0.05,
    ):
        self.d_safe = float(d_safe)
        self.eps_d = float(eps_d)
        self.d_far = float(d_far_factor * d_safe)
        self.r_cap = float(r_cap)
        self.nominal_wps = nominal_waypoints
        self.u_nominal = float(u_nominal)

        # Baseline efficiency weights (sum to 1.0)
        tot_w = w_v + w_c
        self.w_v = w_v / tot_w
        self.w_c = w_c / tot_w

        self.omega = float(omega)
        self.delta_deadband = float(delta_deadband)
        self.s_floor = float(s_floor)
        self.e_floor = float(e_floor)

    def calculate_multi_segment_cte(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes true perpendicular cross-track error across a multi-waypoint polyline
        using clamped orthogonal segment projection (Equation 3.29).
        """
        if self.nominal_wps is None or len(self.nominal_wps) < 2:
            return np.zeros(len(x))

        pts = np.column_stack([x, y])
        n_pts = len(pts)
        ctes = np.zeros(n_pts)

        wps = self.nominal_wps[:, :2]
        seg_starts = wps[:-1]
        seg_ends = wps[1:]
        seg_vecs = seg_ends - seg_starts
        seg_lens_sq = np.sum(seg_vecs**2, axis=1)

        for i in range(n_pts):
            p = pts[i]
            v = p - seg_starts
            t = np.sum(v * seg_vecs, axis=1) / np.maximum(seg_lens_sq, 1e-6)
            t_clamped = np.clip(t, 0.0, 1.0)
            projs = seg_starts + t_clamped[:, np.newaxis] * seg_vecs
            dists = np.linalg.norm(p - projs, axis=1)
            ctes[i] = np.min(dists)

        return ctes

    def evaluate_single_run(
        self,
        t: np.ndarray,
        os_pos: np.ndarray,
        os_psi: np.ndarray,
        os_r: np.ndarray,
        ts_pos: np.ndarray,
        os_u: Optional[np.ndarray] = None,
        num_waypoints: int = 0,
        configured_interval: float = 0.0,
        t_w: Optional[float] = None,
    ) -> dict:
        """
        Evaluates Tier 1 Safety and Tier 2 Efficiency KPIs for a single run over [t_0, t_w].
        """
        # 1. Sanitize timestamps
        t_clean, unique_idx = np.unique(t, return_index=True)
        t = t_clean
        os_pos = os_pos[unique_idx]
        os_psi = os_psi[unique_idx]
        os_r = os_r[unique_idx]
        ts_pos = ts_pos[unique_idx]

        if os_u is not None:
            os_u = os_u[unique_idx]
        else:
            if len(t) > 1:
                vx = np.gradient(os_pos[:, 0], t)
                vy = np.gradient(os_pos[:, 1], t)
                os_u = np.hypot(vx, vy)
            else:
                os_u = np.full_like(t, self.u_nominal)

        # 2. Slice to the common evaluation window [t_0, t_w]
        if t_w is not None:
            mask = t <= (t_w + 1e-4)
            if np.any(mask):
                t = t[mask]
                os_pos = os_pos[mask]
                os_psi = os_psi[mask]
                os_r = os_r[mask]
                ts_pos = ts_pos[mask]
                os_u = os_u[mask]

        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        duration = max(t[-1] - t[0], 1.0)

        # -------------------------------------------------------------
        # TIER 1: SAFETY ASSESSMENT
        # -------------------------------------------------------------
        ranges = np.linalg.norm(ts_pos - os_pos, axis=1)
        d_min = float(np.min(ranges)) if len(ranges) > 0 else float("inf")

        # Breach condition with numerical tolerance
        breached = bool(d_min < (self.d_safe - self.eps_d))
        b_flag = 1 if breached else 0

        # Linear Distance Risk Factor R_dist in [0, 1]
        r_dist = float(np.clip((self.d_far - d_min) / (self.d_far - self.d_safe), 0.0, 1.0))

        # Steering Risk Factor R_steer in [0, 1]
        os_r_rad = os_r if np.nanmax(np.abs(os_r)) < 3.0 else np.deg2rad(os_r)
        r_pk = float(np.nanmax(np.abs(os_r_rad))) if len(os_r_rad) > 0 else 0.0
        r_steer = float(np.clip(r_pk / max(self.r_cap, 1e-3), 0.0, 1.0))

        # Composite Safety Index S = max(R_dist, R_steer)
        s_index = float(max(r_dist, r_steer))

        if breached:
            risk_class = "Breach"
        elif s_index < 0.35:
            risk_class = "Low Risk"
        elif s_index < 0.70:
            risk_class = "Moderate Risk"
        else:
            risk_class = "High Risk"

        # -------------------------------------------------------------
        # TIER 2: EFFICIENCY ASSESSMENT
        # -------------------------------------------------------------
        # Speed Degradation Index E_speed
        ranges = np.linalg.norm(ts_pos - os_pos, axis=1)

        # Identify active encounter bounds where vessels are within d_far
        encounter_mask = ranges <= self.d_far

        if np.any(encounter_mask) and len(t) > 1:
            t_enc = t[encounter_mask]
            os_u_enc = os_u[encounter_mask]
            duration_enc = max(t_enc[-1] - t_enc[0], 1.0)

            speed_ratio_enc = np.clip(os_u_enc / max(self.u_nominal, 0.05), 0.0, 1.0)
            e_speed = float(trapz_fn((1.0 - speed_ratio_enc) ** 2, t_enc)) / duration_enc
        elif len(t) > 1:
            speed_ratio = np.clip(os_u / max(self.u_nominal, 0.05), 0.0, 1.0)
            e_speed = float(trapz_fn((1.0 - speed_ratio) ** 2, t)) / duration
        else:
            e_speed = 0.0

        e_speed = float(np.clip(e_speed, 0.0, 1.0))

        # Route Deviation Index E_route
        if len(t) > 1:
            ctes = self.calculate_multi_segment_cte(os_pos[:, 0], os_pos[:, 1])
            mean_cte = float(trapz_fn(ctes, t)) / duration
            e_route = float(np.clip(mean_cte / max(self.d_far, 1e-3), 0.0, 1.0))
        else:
            e_route = 0.0

        # Efficiency Index E across baseline weights and sensitivity pairs
        e_index = float(self.w_v * e_speed + self.w_c * e_route)
        e_50_50 = float(0.50 * e_speed + 0.50 * e_route)
        e_40_60 = float(0.40 * e_speed + 0.60 * e_route)

        telecom = compute_asm_telecom_kpis(
            num_waypoints=num_waypoints, configured_interval=configured_interval
        )

        metrics = {
            "d_min": d_min,
            "breached": breached,
            "b_flag": b_flag,
            "r_dist": r_dist,
            "r_pk_rad_s": r_pk,
            "r_pk_deg_s": float(np.rad2deg(r_pk)),
            "r_steer": r_steer,
            "s_index": s_index,
            "risk_class": risk_class,
            "e_speed": e_speed,
            "e_route": e_route,
            "e_index": e_index,
            "e_50_50": e_50_50,
            "e_40_60": e_40_60,
        }
        metrics.update(telecom)
        return metrics

    def evaluate_comparison(self, raw_ra: dict, raw_is: dict) -> dict:
        """
        Computes percentage improvements (I_S, I_E, I_total) and classifies the run
        according to the updated Table 4.5 rules.
        """
        b_ra = raw_ra["b_flag"]
        b_is = raw_is["b_flag"]

        s_ra, s_is = raw_ra["s_index"], raw_is["s_index"]
        e_ra, e_is = raw_ra["e_index"], raw_is["e_index"]

        # Percentage Improvements bounded in [-100%, +100%]
        i_s = 100.0 * float(np.clip((s_ra - s_is) / max(s_ra, self.s_floor), -1.0, 1.0))
        i_e = 100.0 * float(np.clip((e_ra - e_is) / max(e_ra, self.e_floor), -1.0, 1.0))

        delta = self.delta_deadband

        # -------------------------------------------------------------
        # SAFETY-GATED TOTAL EVALUATION & OUTCOME CLASSIFICATION
        # -------------------------------------------------------------
        if b_ra == 1 and b_is == 0:
            # Baseline RA breached safety domain, but Intent Sharing avoided the breach.
            # Safety strictly dominates track precision -> Mask route deviation penalty.
            i_e_gated = 0.0
            i_total = float(self.omega * i_s + (1.0 - self.omega) * i_e_gated)
            
            # Ensure i_total reflects clear improvement above the deadband
            if i_total <= delta:
                i_total = max(i_s, delta + 1.0)
            outcome = "Improvement"

        elif b_is == 1:
            # Intent Sharing breached -> Critical safety failure
            i_total = -100.0
            outcome = "Loss of Improvement"

        else:
            # Standard weighted combination when Baseline RA was already safe (b_ra == 0, b_is == 0)
            i_total = float(self.omega * i_s + (1.0 - self.omega) * i_e)
            if i_total > delta:
                outcome = "Improvement"
            elif i_total < -delta:
                outcome = "Loss of Improvement"
            else:
                outcome = "Same / Inconclusive"

        res = {
            "b_ra": b_ra,
            "b_is": b_is,
            "s_ra": s_ra,
            "s_is": s_is,
            "e_ra": e_ra,
            "e_is": e_is,
            "i_s": i_s,
            "i_e": i_e,
            "i_total": i_total,
            "outcome": outcome,
        }
        for k in raw_is:
            if k.startswith("asm_"):
                res[k] = raw_is[k]
        return res