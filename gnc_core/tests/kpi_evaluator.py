import numpy as np


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
            # Projection factor t for all segments
            v = p - seg_starts
            t = np.sum(v * seg_vecs, axis=1) / np.maximum(seg_lens_sq, 1e-6)
            t_clamped = np.clip(t, 0.0, 1.0)
            
            # Closest point on each segment
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
    ) -> dict:
        """Evaluates raw metrics for a single simulation run."""
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
        # Exclude terminal deceleration when arriving at the final waypoint
        if len(t) > 1:
            speed_loss = np.clip(1.0 - (os_u / max(self.u_nominal, 1e-3)), 0.0, 1.0)
            j_speed = float(trapz_fn(speed_loss**2, t)) / duration
        else:
            j_speed = 0.0

        return {
            "r_min": r_min,
            "breached": breached,
            "j_safe": j_safe,
            "j_ctrl": j_ctrl,
            "j_cte": j_cte,
            "j_speed": j_speed,
        }

    def evaluate_comparison(self, raw_ra: dict, raw_is: dict) -> dict:
        """Computes relative baseline normalization and composite delta J."""
        if raw_is["breached"]:
            return {
                "status": "IS_BREACH",
                "j_total_is": np.inf,
                "delta_j": np.inf,
                "delta_j_pct": -np.inf,
            }

        if raw_ra["breached"] and not raw_is["breached"]:
            return {
                "status": "SAFETY_ENHANCEMENT",
                "j_total_is": 0.0,
                "delta_j": -np.inf,
                "delta_j_pct": 100.0,
            }

        # Normalize metrics against RA baseline
        norm_ctrl = raw_is["j_ctrl"] / max(raw_ra["j_ctrl"], 1e-6)
        norm_cte = raw_is["j_cte"] / max(raw_ra["j_cte"], 1e-6)

        # Baseline normalization for speed
        if raw_ra["j_speed"] > 1e-5:
            norm_speed = raw_is["j_speed"] / raw_ra["j_speed"]
        else:
            # If RA experienced zero speed reduction, assess IS penalty directly
            norm_speed = 1.0 + raw_is["j_speed"]

        j_total_is = (
            self.w_ctrl * norm_ctrl
            + self.w_cte * norm_cte
            + self.w_speed * norm_speed
        )
        delta_j = j_total_is - 1.0
        delta_j_pct = (1.0 - j_total_is) * 100.0

        return {
            "status": "SAFE_COMPARISON",
            "norm_ctrl": norm_ctrl,
            "norm_cte": norm_cte,
            "norm_speed": norm_speed,
            "j_total_is": j_total_is,
            "delta_j": delta_j,
            "delta_j_pct": delta_j_pct,
        }