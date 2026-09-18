import numpy as np


def compute_asm_telecom_kpis(num_waypoints: int, configured_interval: float) -> dict:
    """
    Computes exact bit length, payload bytes, ITU slot occupancy,
    and regulatory compliance according to the ASM schema (Figure 2.1)
    and ITU-R M.1371-6 Table 56.
    """
    # Schema supports up to 7 route waypoints (max 8 entries total)
    n_wp = int(np.clip(num_waypoints, 0, 7))

    # 1. Header bit definitions
    bits_standard_frame = 6 + 2 + 30 + 2       # 40 bits (Msg 8 header)
    bits_app_header = 16 + 3 + 2 + 17 + 3      # 41 bits (IAI through Num_WP)

    # 2. Waypoint bit definitions
    if n_wp == 0:
        total_bits = bits_standard_frame + bits_app_header
    else:
        bits_wp1 = 27 + 26 + 9 + 10            # 72 bits (Absolute WGS-84 anchor)
        bits_subsequent = (n_wp - 1) * (12 + 12 + 9 + 10)  # 53 bits each (Delta)
        total_bits = bits_standard_frame + bits_app_header + bits_wp1 + bits_subsequent

    # Binary data payload evaluated against ITU Table 56 excludes the 40-bit frame
    payload_bytes = int(np.ceil((total_bits - bits_standard_frame) / 8.0))

    # 3. Required ITU TDMA slots (ITU-R M.1371-6 Table 56)
    if payload_bytes <= 12:
        slots = 1
    elif payload_bytes <= 40:
        slots = 2
    elif payload_bytes <= 68:
        slots = 3
    else:
        slots = int(np.ceil(payload_bytes / 28.0))

    # 4. Regulatory checks (20 slots/min RATDMA mobile station limit)
    min_compliant_interval = slots * 3.0
    actual_slots_per_min = (slots * 60.0) / configured_interval if configured_interval > 0.0 else 0.0
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
    """Clamps the input broadcast interval to the ITU-R M.1371-6 RATDMA minimum threshold.

    Returns:
        effective_interval: max(input_interval, min_compliant_interval)
        min_compliant_interval: ITU-R M.1371-6 limit (slots * 3.0s)
    """
    telecom = compute_asm_telecom_kpis(
        num_waypoints=num_waypoints, configured_interval=input_interval
    )
    min_compliant = telecom["asm_min_interval_s"]
    effective_interval = max(float(input_interval), float(min_compliant))

    return effective_interval, min_compliant


class ScenarioKPIEvaluator:

    def __init__(
        self,
        d_safe: float,
        nominal_waypoints: np.ndarray,
        w_cte: float = 0.6,
        w_ctrl: float = 0.2,
        w_speed: float = 0.2,
        u_nominal: float = 0.45,
    ):
        """
        Evaluates safety, control effort, mission deviation, and speed degradation.
        """
        self.d_safe = d_safe
        self.nominal_wps = nominal_waypoints

        # Normalize weights so they sum to 1.0
        total_w = w_cte + w_ctrl + w_speed
        self.w_cte = w_cte / total_w
        self.w_ctrl = w_ctrl / total_w
        self.w_speed = w_speed / total_w

        self.u_nominal = u_nominal

    def calculate_cte(
        self, x: np.ndarray, y: np.ndarray, wp_a: np.ndarray, wp_b: np.ndarray
    ) -> np.ndarray:
        """Computes signed cross-track error using orthogonal leg projection."""
        dx = wp_b[0] - wp_a[0]
        dy = wp_b[1] - wp_a[1]
        alpha_k = np.arctan2(dy, dx)
        return -(x - wp_a[0]) * np.sin(alpha_k) + (y - wp_a[1]) * np.cos(alpha_k)

    def calculate_multi_segment_cte(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Computes true perpendicular CTE across a multi-waypoint polyline."""
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
        os_u: np.ndarray = None,
        num_waypoints: int = 0,
        configured_interval: float = 0.0,
    ) -> dict:
        """Evaluates raw metrics and telecom footprint for a single simulation run."""
        # 0. Sanitize duplicate timestamps to prevent gradient divide-by-zero
        t_clean, unique_indices = np.unique(t, return_index=True)
        os_pos = os_pos[unique_indices]
        os_psi = os_psi[unique_indices]
        os_r = os_r[unique_indices]
        ts_pos = ts_pos[unique_indices]
        t = t_clean

        # Infer surge speed u(t) from coordinate displacement if not directly supplied
        if os_u is None:
            if len(t) > 1:
                vx = np.gradient(os_pos[:, 0], t)
                vy = np.gradient(os_pos[:, 1], t)
                os_u = np.hypot(vx, vy)
            else:
                os_u = np.full_like(t, self.u_nominal)
        else:
            os_u = os_u[unique_indices]

        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        duration = max(t[-1] - t[0], 1.0)

        # 1. Safety Constraint: Range R(t)
        ranges = np.linalg.norm(ts_pos - os_pos, axis=1)
        r_min = float(np.min(ranges))
        breached = r_min < self.d_safe
        j_safe = np.inf if breached else 0.0

        # 2. Control Effort: Integral of squared yaw acceleration r_dot^2
        if len(t) > 1:
            r_dot = np.gradient(os_r, t)
            j_ctrl = float(trapz_fn(r_dot**2, t)) / duration
        else:
            j_ctrl = 0.0

        # 3. Mission Tracking: Multi-segment Cross-Track Error
        e_cte = self.calculate_multi_segment_cte(os_pos[:, 0], os_pos[:, 1])
        j_cte = float(trapz_fn(np.abs(e_cte), t)) / duration if len(t) > 1 else 0.0

        # 4. Speed Degradation Penalty: Normalized deviation from u_nominal
        if len(t) > 1:
            speed_loss = np.clip(1.0 - (os_u / max(self.u_nominal, 1e-3)), 0.0, 1.0)
            j_speed = float(trapz_fn(speed_loss**2, t)) / duration
        else:
            j_speed = 0.0

        # 5. Telecom Footprint & VDL Loading
        telecom_metrics = compute_asm_telecom_kpis(
            num_waypoints=num_waypoints,
            configured_interval=configured_interval,
        )

        metrics = {
            "r_min": r_min,
            "breached": breached,
            "j_safe": j_safe,
            "j_ctrl": j_ctrl,
            "j_cte": j_cte,
            "j_speed": j_speed,
        }
        metrics.update(telecom_metrics)
        return metrics

    def evaluate_comparison(self, raw_ra: dict, raw_is: dict) -> dict:
        """Computes relative baseline normalization and composite delta J."""
        if raw_is["breached"]:
            res = {
                "status": "IS_BREACH",
                "j_total_is": np.inf,
                "delta_j": np.inf,
                "delta_j_pct": -np.inf,
            }
            # Carry over IS telecom KPIs into comparison output
            for k in raw_is:
                if k.startswith("asm_"):
                    res[k] = raw_is[k]
            return res

        if raw_ra["breached"] and not raw_is["breached"]:
            res = {
                "status": "SAFETY_ENHANCEMENT",
                "j_total_is": 0.0,
                "delta_j": -np.inf,
                "delta_j_pct": 100.0,
            }
            for k in raw_is:
                if k.startswith("asm_"):
                    res[k] = raw_is[k]
            return res

        # Normalize metrics against RA baseline
        norm_ctrl = raw_is["j_ctrl"] / max(raw_ra["j_ctrl"], 1e-6)
        norm_cte = raw_is["j_cte"] / max(raw_ra["j_cte"], 1e-6)
        norm_speed = raw_is["j_speed"] / max(raw_ra["j_speed"], 1e-6)
        
        j_total_is = (
            self.w_ctrl * norm_ctrl
            + self.w_cte * norm_cte
            + self.w_speed * norm_speed
        )
        delta_j = j_total_is - 1.0
        delta_j_pct = (1.0 - j_total_is) * 100.0

        res = {
            "status": "SAFE_COMPARISON",
            "norm_ctrl": norm_ctrl,
            "norm_cte": norm_cte,
            "norm_speed": norm_speed,
            "j_total_is": j_total_is,
            "delta_j": delta_j,
            "delta_j_pct": delta_j_pct,
        }
        for k in raw_is:
            if k.startswith("asm_"):
                res[k] = raw_is[k]
        return res