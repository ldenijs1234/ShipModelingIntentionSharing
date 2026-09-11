from typing import Tuple, Optional
import time
import sys
import numpy as np
from gnc_core.config.vessel_params import VesselParams
from gnc_core.navigation.risk import RiskCalculator


class DecisionLayer:
    _mode_a_active = False
    _w_latched = None

    @staticmethod
    def _compute_cross_track_error(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Calculates orthogonal cross-track distance from 2D point p to line segment a-b."""
        ab = b - a
        norm_ab = np.linalg.norm(ab)
        if norm_ab < 1e-6:
            return float(np.linalg.norm(p - a))
        return float(abs(ab[0] * (a[1] - p[1]) - ab[1] * (a[0] - p[0])) / norm_ab)

    @classmethod
    def evaluate(
        cls,
        x_os: np.ndarray,
        x_ts: np.ndarray,
        w_os: np.ndarray,
        w_ts_delayed: Optional[np.ndarray],
        dcpa: float,
        tcpa: float,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Evaluates deterministic surrogate decision logic.
        Generates a 4-point evasive corridor and latches until OS re-merges onto the original mission.
        """
        initial_risk = (dcpa <= VesselParams.DCPA_safe) and (0.0 <= tcpa <= VesselParams.TCPA_safe)

        # ----------------------------------------------------------------------
        # Mode A: Informed CPA Evaluation (Cooperative Track)
        # ----------------------------------------------------------------------
        if w_ts_delayed is not None and len(w_ts_delayed) >= 2:
            if cls._mode_a_active:
                # 1. Encounter clearance condition
                ts_cleared = (tcpa <= 0.0) or (x_ts[0] < x_os[0] - 5.0)

                # 2. General cross-track error check relative to original mission final leg
                p_os = np.array([x_os[0], x_os[1]])
                leg_start = w_os[-2, :2] if len(w_os) >= 2 else w_os[0, :2]
                leg_end = w_os[-1, :2]
                xt_error = cls._compute_cross_track_error(p_os, leg_start, leg_end)

                # Re-merged when close to original mission path and past the evasion midpoint
                remerged_on_original_mission = (xt_error < 0.3) and (x_os[0] > (cls._w_latched[1, 0] + 5.0))

                # 3. Near final mission waypoint
                dist_to_end = np.linalg.norm(p_os - leg_end)
                near_destination = dist_to_end <= 2.0

                if near_destination or (ts_cleared and remerged_on_original_mission):
                    cls._mode_a_active = False
                    cls._w_latched = None
            else:
                if initial_risk:
                    cls._mode_a_active = True
                    t_start = time.perf_counter()

                    # Select candidate heading alteration chi*
                    chi_candidates = np.array([
                        -90.0, -75.0, -60.0, -45.0, -30.0, -15.0,
                        0.0,
                        15.0, 30.0, 45.0, 60.0, 75.0, 90.0
                    ])
                    p_port = 1000.0
                    k_chi = 1.0

                    best_cost = float("inf")
                    best_chi = 30.0

                    for chi in chi_candidates:
                        proj_dcpa = dcpa + (np.radians(chi) * 10.0)
                        if proj_dcpa < VesselParams.DCPA_safe and chi < 0:
                            j_safety = float("inf")
                        else:
                            j_safety = VesselParams.DCPA_safe / max(proj_dcpa, 1e-5)

                        j_colregs = p_port if chi < 0 else 0.0
                        j_smooth = k_chi * abs(np.radians(chi))

                        total_cost = j_safety + j_colregs + j_smooth
                        if total_cost < best_cost:
                            best_cost = total_cost
                            best_chi = chi

                    # Compute lateral shift perpendicular to encounter geometry
                    x_cpa = 0.5 * (x_os[0] + x_ts[0])
                    y_offset = 1.2 * VesselParams.DCPA_safe * np.sign(best_chi if best_chi != 0 else 1.0)

                    # Build complete 4-point evasive corridor
                    wp_start = np.array([x_os[0], x_os[1]])
                    wp_evade = np.array([x_cpa, y_offset])
                    wp_pass = np.array([x_cpa + 6.0, y_offset])
                    wp_end = w_os[-1, :].copy()

                    cls._w_latched = np.vstack([wp_start, wp_evade, wp_pass, wp_end])

                    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    sys.stderr.write(
                        f"\n\033[1;92m>>> [DecisionLayer] Mode A.1 corridor created in: {t_elapsed_ms:.4f} ms (chi* = {best_chi:+.1f} deg)\033[0m\n\n"
                    )
                    sys.stderr.flush()

            if cls._mode_a_active and cls._w_latched is not None:
                return cls._w_latched, 0.0, "State A.1"

            return np.copy(w_os), 0.0, "State A.2"

        # ----------------------------------------------------------------------
        # Mode B: Reactive CPA Evaluation (Uncollaborative Fallback)
        # ----------------------------------------------------------------------
        cls._mode_a_active = False
        cls._w_latched = None

        if initial_risk:
            beta = RiskCalculator.calculate_relative_bearing(x_os, x_ts)
            scenario = RiskCalculator.classify_colreg_scenario(beta)
            offset = 45.0 if scenario in ["Head-On", "Crossing_A"] else (30.0 if scenario == "Overtaking" else 0.0)
            return np.copy(w_os), float(np.radians(offset)), "State B.1"

        return np.copy(w_os), 0.0, "State B.2"