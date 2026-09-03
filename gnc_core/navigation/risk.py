from typing import Tuple
import numpy as np


class RiskCalculator:

    @staticmethod
    def calculate_relative_bearing(x_os: np.ndarray, x_ts: np.ndarray) -> float:
        """Computes relative bearing beta [0, 360) deg (Eq. 2.1 & 2.2)."""
        phi = np.arctan2(x_ts[1] - x_os[1], x_ts[0] - x_os[0])
        return float(np.degrees(phi - x_os[2]) % 360.0)

    @staticmethod
    def classify_colreg_scenario(beta_deg: float) -> str:
        """Classifies encounter based on relative bearing boundaries (Table 2.1)."""
        if beta_deg <= 22.5 or beta_deg >= 337.5:
            return "Head-On"
        elif 112.5 < beta_deg < 247.5:
            return "Overtaking"
        elif 22.5 < beta_deg <= 112.5:
            return "Crossing_A"
        return "Crossing_B"

    @staticmethod
    def calculate_cpa(x_os: np.ndarray, x_ts: np.ndarray) -> Tuple[float, float]:
        """Calculates DCPA and TCPA geometrically (Eq. 3.14 - 3.22)."""
        X, Y, psi, u, _, _ = x_os
        X_ob, Y_ob, psi_ts, u_ts, _, _ = x_ts

        Vx, Vy = u * np.cos(psi), u * np.sin(psi)
        Vx_ob, Vy_ob = u_ts * np.cos(psi_ts), u_ts * np.sin(psi_ts)

        dx, dy = X_ob - X, Y_ob - Y
        R = np.sqrt(dx**2 + dy**2)

        dVx, dVy = Vx_ob - Vx, Vy_ob - Vy
        V_rel = np.sqrt(dVx**2 + dVy**2)
        if V_rel < 1e-5:
            return float(R), 0.0

        psi_vrel = np.arctan2(dVy, dVx)
        rho = np.arctan2(-dy, -dx)
        alpha = rho - psi_vrel

        dcpa = np.abs(R * np.sin(alpha))
        tcpa = (R * np.cos(alpha)) / V_rel
        return float(dcpa), float(tcpa)

    @staticmethod
    def calculate_trajectory_cpa(
        os_trajectory: np.ndarray, 
        ts_trajectory: np.ndarray,
        dt: float = 0.05
    ) -> Tuple[float, float]:
        """
        Computes minimum Euclidean distance (DCPA) in meters and time (TCPA) in seconds
        across synchronized future discrete look-ahead arrays.
        """
        min_len = min(len(os_trajectory), len(ts_trajectory))
        if min_len == 0:
            return float("inf"), 0.0

        diffs = os_trajectory[:min_len, :2] - ts_trajectory[:min_len, :2]
        distances = np.linalg.norm(diffs, axis=1)

        min_idx = int(np.argmin(distances))
        dcpa = float(distances[min_idx])
        tcpa = float(min_idx * dt)
        
        return dcpa, tcpa

    @staticmethod
    def project_ts_on_shared_trajectory(
        x_ts_pos: np.ndarray, 
        w_ts: np.ndarray, 
        u_ts: float = 0.5
    ) -> Tuple[int, float, float]:
        """Projects TS position onto delayed intention waypoints (Eq. 3.25 - 3.32)."""
        if len(w_ts) < 2:
            return 0, 0.0, 0.0

        best_j, min_cte, best_d_seg = 0, float("inf"), 0.0
        for j in range(len(w_ts) - 1):
            L_j = w_ts[j + 1] - w_ts[j]
            L_norm_sq = np.dot(L_j, L_j)
            if L_norm_sq < 1e-6:
                continue

            P_j = x_ts_pos - w_ts[j]
            c_j = np.clip(np.dot(P_j, L_j) / L_norm_sq, 0.0, 1.0)
            e_cte = np.linalg.norm(P_j - c_j * L_j)

            if e_cte < min_cte:
                min_cte, best_j = e_cte, j
                best_d_seg = c_j * np.sqrt(L_norm_sq)

        # Eq. 3.32: t_progress along current segment
        t_progress = best_d_seg / max(u_ts, 1e-3)

        return best_j, float(best_d_seg), float(t_progress)