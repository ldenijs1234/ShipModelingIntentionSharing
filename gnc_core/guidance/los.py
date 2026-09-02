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

        idx = min(current_wp_idx, num_wps - 1)
        wp_prev = w_active[idx - 1]
        wp_curr = w_active[idx]

        # Waypoint switching via circle of acceptance D_m
        dist_to_wp = np.sqrt((wp_curr[0] - x_os[0])**2 + (wp_curr[1] - x_os[1])**2)
        if dist_to_wp < VesselParams.D_m and idx < (num_wps - 1):
            idx += 1
            wp_prev, wp_curr = w_active[idx - 1], w_active[idx]

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