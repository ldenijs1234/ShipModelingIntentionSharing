from typing import Tuple
import numpy as np
from gnc_core.config.vessel_params import VesselParams


class LOSGuidance:

    @staticmethod
    def compute_heading_reference(
        x_os: np.ndarray, w_active: np.ndarray, current_wp_idx: int
    ) -> Tuple[float, float, int]:
        num_wps = len(w_active)
        if num_wps < 2:
            return float(x_os[2]), 0.0, current_wp_idx

        # Enforce valid segment index (at least 1 for w_active[idx - 1] to exist)
        idx = max(1, min(current_wp_idx, num_wps - 1))
        wp_prev = w_active[idx - 1]
        wp_curr = w_active[idx]

        # -------------------------------------------------------------
        # Waypoint Switching Logic (Distance + Along-Track Overflight)
        # -------------------------------------------------------------
        leg_vec = wp_curr[:2] - wp_prev[:2]
        leg_len = float(np.hypot(leg_vec[0], leg_vec[1]))
        u_leg = leg_vec / max(leg_len, 1e-4)

        # Vector from target waypoint to vessel position
        v_to_os = x_os[:2] - wp_curr[:2]
        dist_to_wp = float(np.hypot(v_to_os[0], v_to_os[1]))

        # Condition 1: Vessel is within circular acceptance zone
        reached_dist = dist_to_wp <= VesselParams.D_m

        # Condition 2: Vessel crossed the orthogonal boundary of the waypoint
        passed_wp = (np.dot(v_to_os, u_leg) > 0.0) and (dist_to_wp < (VesselParams.D_m * 3.5))

        # Advance to downstream waypoint if not already at the terminal waypoint
        if (reached_dist or passed_wp) and (idx < num_wps - 1):
            idx += 1
            wp_prev = w_active[idx - 1]
            wp_curr = w_active[idx]

        # Track orientation angle psi_trk
        dx = wp_curr[0] - wp_prev[0]
        dy = wp_curr[1] - wp_prev[1]
        psi_trk = np.arctan2(dy, dx)

        # Vector from wp_prev to own ship position
        p_x = x_os[0] - wp_prev[0]
        p_y = x_os[1] - wp_prev[1]

        # Perpendicular cross-track error e_cte (positive if OS is starboard of track)
        e_cte = -np.sin(psi_trk) * p_x + np.cos(psi_trk) * p_y

        # LOS steering law
        psi_los = psi_trk - np.arctan(e_cte / VesselParams.mu_los)
        psi_los = (psi_los + np.pi) % (2.0 * np.pi) - np.pi

        return float(psi_los), float(e_cte), idx