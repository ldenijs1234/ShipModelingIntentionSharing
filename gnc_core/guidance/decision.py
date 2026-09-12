#!/usr/bin/env python3
from typing import Tuple, Optional
import time
import numpy as np
import rclpy.logging

from gnc_core.config.vessel_params import VesselParams
from gnc_core.navigation.risk import RiskCalculator


class DecisionLayer:
    _mode_a_active = False
    _mode_b_active = False
    _w_latched = None

    # Akdağ's asymmetric course penalty weights (k_ci)
    k_chi_stb = 5.0
    k_chi_port = 5.2

    @staticmethod
    def _compute_cross_track_error(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Computes Euclidean cross-track error to segment [a, b]."""
        ab = b - a
        norm_ab = np.linalg.norm(ab)
        if norm_ab < 1e-6:
            return float(np.linalg.norm(p - a))
        return float(abs(ab[0] * (a[1] - p[1]) - ab[1] * (a[0] - p[0])) / norm_ab)

    @classmethod
    def evaluate_reactive_offset(
        cls, x_os: np.ndarray, x_ts: np.ndarray, scenario: str
    ) -> float:
        """Evaluates candidate offsets (Eq. 3.43) against linear projection."""
        if scenario == "Head-On":
            candidates = np.radians([30.0, 45.0, 60.0, 75.0])
        elif scenario in ["Crossing_A", "Crossing_B"]:
            candidates = np.radians([45.0, 60.0, 75.0, 85.0])
        elif scenario == "Overtaking":
            candidates = np.radians([-30.0, 30.0])
        else:
            candidates = np.radians([30.0, 45.0, 60.0])

        u_os = max(x_os[3], 0.1)
        u_ts = max(x_ts[3], 0.1)
        psi_ts = x_ts[2]
        best_chi = candidates[-1]

        for chi in candidates:
            psi_os_cand = x_os[2] + chi
            vx_os = u_os * np.cos(psi_os_cand)
            vy_os = u_os * np.sin(psi_os_cand)
            vx_ts = u_ts * np.cos(psi_ts)
            vy_ts = u_ts * np.sin(psi_ts)

            dx = x_ts[0] - x_os[0]
            dy = x_ts[1] - x_os[1]
            dvx = vx_ts - vx_os
            dvy = vy_ts - vy_os
            v_rel_sq = dvx**2 + dvy**2

            if v_rel_sq > 1e-4:
                tcpa_cand = -(dx * dvx + dy * dvy) / v_rel_sq
                if tcpa_cand > 0.0:
                    dcpa_cand = np.hypot(dx + dvx * tcpa_cand, dy + dvy * tcpa_cand)
                else:
                    dcpa_cand = np.hypot(dx, dy)
            else:
                dcpa_cand = np.hypot(dx, dy)

            if dcpa_cand >= (VesselParams.R_lateral + VesselParams.B):
                best_chi = chi
                break

        return float(best_chi)

    @classmethod
    def evaluate(
        cls,
        x_os: np.ndarray,
        x_ts: Optional[np.ndarray],
        w_os: np.ndarray,
        w_ts_delayed: Optional[np.ndarray],
        dcpa: float,
        tcpa: float,
    ) -> Tuple[np.ndarray, float, str]:
        logger = rclpy.logging.get_logger("DecisionLayer")

        if x_ts is None:
            return np.copy(w_os), 0.0, "STAND_ON"

        # ----------------------------------------------------------------------
        # Mode A: Collaborative Shared Intent (Route Available)
        # ----------------------------------------------------------------------
        if w_ts_delayed is not None and len(w_ts_delayed) >= 2:
            cls._mode_b_active = False

            # If active corridor is already executing, track it until passed
            if cls._mode_a_active and cls._w_latched is not None:
                if x_os[0] >= cls._w_latched[2, 0]:
                    return np.copy(w_os), 0.0, "State A.2"
                return cls._w_latched, 0.0, "State A.1"

            # Setup forward trajectory projection (Akdağ ship_trajectory)
            u_os = max(x_os[3], 0.45)
            u_ts = max(x_ts[3], 0.45)
            hor_time = 70.0
            dt_sim = 1.0
            steps = int(hor_time / dt_sim)

            # Target vessel planned trajectory forward integration
            ts_traj = np.zeros((steps, 2))
            diffs = np.diff(w_ts_delayed[:, :2], axis=0)
            seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
            total_route_len = np.sum(seg_lengths)

            for k in range(steps):
                s = min(u_ts * k * dt_sim, total_route_len)
                accum = 0.0
                pt = w_ts_delayed[-1, :2]
                for seg_idx, length in enumerate(seg_lengths):
                    if accum + length >= s:
                        ratio = (s - accum) / max(length, 1e-4)
                        pt = w_ts_delayed[seg_idx, :2] + ratio * diffs[seg_idx]
                        break
                    accum += length
                ts_traj[k] = pt

            d_safe = VesselParams.DCPA_safe

            # ------------------------------------------------------------------
            # Gate: Evaluate Nominal Track (chi = 0.0) first
            # ------------------------------------------------------------------
            vx_nom = u_os * np.cos(x_os[2])
            vy_nom = u_os * np.sin(x_os[2])
            os_nom_x = x_os[0] + vx_nom * np.arange(steps) * dt_sim
            os_nom_y = x_os[1] + vy_nom * np.arange(steps) * dt_sim
            nom_dists = np.hypot(os_nom_x - ts_traj[:, 0], os_nom_y - ts_traj[:, 1])
            nom_min_dist = float(np.min(nom_dists))

            # If nominal track maintains safe separation, stay on original route
            if nom_min_dist >= d_safe:
                return np.copy(w_os), 0.0, "State A.2"

            # ------------------------------------------------------------------
            # Conflict detected: Optimize evasive action (Eq. 3.33 - 3.41)
            # ------------------------------------------------------------------
            cls._mode_a_active = True
            t_start = time.perf_counter()

            chi_candidates = np.radians(np.array(
                [-75.0, -60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0]
            ))

            best_cost = float("inf")
            best_chi = None
            best_min_dist = 0.0

            for chi in chi_candidates:
                psi_cand = x_os[2] + chi
                vx = u_os * np.cos(psi_cand)
                vy = u_os * np.sin(psi_cand)

                os_traj_x = x_os[0] + vx * np.arange(steps) * dt_sim
                os_traj_y = x_os[1] + vy * np.arange(steps) * dt_sim
                dists = np.hypot(os_traj_x - ts_traj[:, 0], os_traj_y - ts_traj[:, 1])
                min_dist = float(np.min(dists))

                if min_dist <= d_safe:
                    j_safety = 1000.0 * ((d_safe / max(min_dist, 0.05)) ** 4.0)
                else:
                    j_safety = 50.0 * (d_safe / min_dist)

                k_weight = cls.k_chi_port if chi < 0.0 else cls.k_chi_stb
                j_control = k_weight * (chi ** 2)

                cost = j_safety + j_control

                if cost < best_cost:
                    best_cost = cost
                    best_chi = chi
                    best_min_dist = min_dist

            if best_chi is None or abs(best_chi) < 1e-4:
                cls._mode_a_active = False
                return np.copy(w_os), 0.0, "State A.2"

            # Generate 4-point evasion corridor
            lat_sign = -1.0 if best_chi < 0.0 else 1.0
            y_corridor = x_os[1] + (lat_sign * 2.0 * d_safe)

            wp_start = np.array([x_os[0], x_os[1]])
            wp_evade = np.array([x_os[0] + 6.0, y_corridor])
            wp_pass = np.array([12.0, y_corridor])
            wp_end = w_os[-1, :].copy()

            cls._w_latched = np.vstack([wp_start, wp_evade, wp_pass, wp_end])

            t_comp_ms = (time.perf_counter() - t_start) * 1000.0
            logger.info(
                f"\n===================================================\n"
                f"[DecisionLayer] FIRST EVASIVE ROUTE GENERATED:\n"
                f"  Computation Time : {t_comp_ms:.4f} ms\n"
                f"  Optimal chi*     : {np.degrees(best_chi):+.1f} deg\n"
                f"  Min Distance     : {best_min_dist:.2f} m\n"
                f"  Total Cost       : {best_cost:.2f}\n"
                f"===================================================\n"
            )

            return cls._w_latched, 0.0, "State A.1"

        # ----------------------------------------------------------------------
        # Mode B: Uncollaborative Reactive Fallback (No Intent)
        # ----------------------------------------------------------------------
        cls._mode_a_active = False
        cls._w_latched = None

        domain_breached, _ = RiskCalculator.check_ship_domain_breach(x_os, x_ts)
        threshold_cpa = VesselParams.R_lateral + VesselParams.B
        cpa_risk = (dcpa < threshold_cpa) and (0.0 <= tcpa <= VesselParams.TCPA_safe)
        risk_active = cpa_risk or domain_breached

        if cls._mode_b_active:
            if x_os[0] > (x_ts[0] + 1.0) or (tcpa <= 0.0 and dcpa >= threshold_cpa):
                cls._mode_b_active = False
        else:
            if risk_active and (x_ts[0] > x_os[0]):
                cls._mode_b_active = True

        if cls._mode_b_active:
            beta = RiskCalculator.calculate_relative_bearing(x_os, x_ts)
            scenario = RiskCalculator.classify_colreg_scenario(beta)
            psi_ca = cls.evaluate_reactive_offset(x_os, x_ts, scenario)
            return np.copy(w_os), psi_ca, "State B.1"

        return np.copy(w_os), 0.0, "State B.2"