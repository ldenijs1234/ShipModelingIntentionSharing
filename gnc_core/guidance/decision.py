#!/usr/bin/env python3
from typing import Optional, Tuple
import numpy as np
import time
from shapely.geometry import LineString

from gnc_core.config.vessel_params import VesselParams
from gnc_core.navigation.risk import RiskCalculator


class DecisionLayer:
    _mode_a_active = False
    _mode_b_active = False
    _w_evasive_latched = None
    _p_evasive_latched = 1.0

    k_chi_stb = 5.0
    k_chi_port = 5.2
    k_p = 10.0
    
    # New penalty weights defined matching LaTeX
    k_safety = 1000.0
    k_grounding = 5.0
    k_colregs = 10.0

    # Discrete search set matching Akdag (radians)
    chi_candidates = np.radians(
        np.array([
            -90.0, -75.0, -60.0, -45.0, -30.0, -15.0,
            0.0,
            15.0, 30.0, 45.0, 60.0, 75.0, 90.0,
        ])
    )
    
    # Discrete speed multiplier candidates matching Akdag
    p_candidates = np.array([1.0, 0.5, 0.0])

    @classmethod
    def _evaluate_candidate_hazard(
        cls,
        x_os: np.ndarray,
        chi: float,
        p_cand: float,
        ts_traj: np.ndarray,
        steps: int,
        dt_sim: float,
        d_safe: float,
        u_nominal: float = 0.45,
        canal_polygons = None,
        scenario: str = "Unknown"
    ) -> Tuple[float, float, int]:
        
        u_os = (float(x_os[3]) if abs(x_os[3]) > 0.05 else float(u_nominal)) * p_cand
        psi_cand = x_os[2] + chi

        vx = u_os * np.cos(psi_cand)
        vy = u_os * np.sin(psi_cand)

        t_steps = np.arange(steps) * dt_sim
        os_x = x_os[0] + vx * t_steps
        os_y = x_os[1] + vy * t_steps

        # 1. Safety Cost (J_safety) - Fixed to strict piece-wise zero tail
        dists = np.hypot(os_x - ts_traj[:, 0], os_y - ts_traj[:, 1])
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist <= d_safe:
            j_safety = cls.k_safety * ((d_safe / max(min_dist, 0.05)) ** 4.0)
        else:
            j_safety = 0.0

        # 2. Grounding Cost (J_grounding)
        j_grounding = 0.0
        if canal_polygons is not None:
            traj_line = LineString(np.column_stack((os_x, os_y)))
            min_static_dist = traj_line.distance(canal_polygons)
            
            # Add this: Hard physical boundary truncation (Thesis Eq 2.12 constraints)
            if min_static_dist <= VesselParams.R_lateral:
                return float("inf"), min_dist, min_idx
                
            if min_static_dist <= VesselParams.d_safe_static:
                j_grounding = cls.k_grounding * ((VesselParams.d_safe_static / (min_static_dist + 0.05)) ** 2.5)

        # 3. Control Effort Cost (J_control & J_speed)
        k_w = cls.k_chi_stb if chi <= 0.0 else cls.k_chi_port
        j_control = k_w * (chi**2)
        j_speed = cls.k_p * (1.0 - p_cand)

        # 4. COLREGs Cost (J_colregs)
        j_colregs = 0.0
        if scenario != "Unknown":
            # Vectorized mapping to [-180, 180] relative bearing
            phi_array = np.arctan2(ts_traj[:, 1] - os_y, ts_traj[:, 0] - os_x)
            beta_array = np.degrees(phi_array - psi_cand)
            beta_array = (beta_array + 180.0) % 360.0 - 180.0

            if scenario == "Head-On":
                mu_array = (beta_array <= 13.0)
            elif scenario in ["Crossing_A", "Crossing_B"]:
                mu_array = (beta_array <= 0.0)
            elif scenario == "Overtaking":
                mu_array = (np.abs(beta_array) <= 22.5)
            else:
                mu_array = np.zeros_like(beta_array, dtype=bool)

            if np.any(mu_array):
                j_colregs = cls.k_colregs

        total_cost = j_safety + j_grounding + j_control + j_speed + j_colregs
        return total_cost, min_dist, min_idx

    @classmethod
    def _sample_route(
        cls, wps: np.ndarray, pos: np.ndarray, speed: float, steps: int, dt: float
    ) -> Tuple[np.ndarray, float]:
        """Samples along the route starting from the orthogonal projection of pos."""
        diffs = np.diff(wps[:, :2], axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        total_len = float(np.sum(seg_lens))

        s_start = 0.0
        min_d = float('inf')
        accum = 0.0
        for i in range(len(seg_lens)):
            v = pos[:2] - wps[i, :2]
            u_vec = diffs[i] / max(seg_lens[i], 1e-4)
            proj = max(0.0, min(float(np.dot(v, u_vec)), seg_lens[i]))
            pt = wps[i, :2] + u_vec * proj
            dist = float(np.linalg.norm(pos[:2] - pt))
            if dist < min_d:
                min_d = dist
                s_start = accum + proj
            accum += seg_lens[i]

        traj = np.zeros((steps, 2))
        for k in range(steps):
            s = min(s_start + speed * k * dt, total_len)
            accum_dist = 0.0
            pt = wps[-1, :2]
            for i, seg_len in enumerate(seg_lens):
                if accum_dist + seg_len >= s:
                    ratio = (s - accum_dist) / max(seg_len, 1e-4)
                    pt = wps[i, :2] + ratio * diffs[i]
                    break
                accum_dist += seg_len
            traj[k] = pt

        return traj, s_start

    @classmethod
    def evaluate(
        cls,
        x_os: np.ndarray,
        x_ts: Optional[np.ndarray],
        w_os: np.ndarray,
        w_ts_delayed: Optional[np.ndarray],
        dcpa: float,
        tcpa: float,
        u_nominal: float = 0.45,
        canal_polygons = None
    ) -> Tuple[np.ndarray, float, float, str]:
        
        if x_ts is None:
            cls._mode_a_active = False
            cls._mode_b_active = False
            cls._w_evasive_latched = None
            return np.copy(w_os), 0.0, 1.0, "STAND_ON"

        u_os = float(x_os[3]) if abs(x_os[3]) > 0.05 else float(u_nominal)
        u_ts = float(x_ts[3]) if abs(x_ts[3]) > 0.05 else float(u_nominal)

        if u_os < 0.10 or u_ts < 0.10:
            cls._mode_a_active = False
            cls._mode_b_active = False
            cls._w_evasive_latched = None
            return np.copy(w_os), 0.0, 1.0, "State B.2"

        d_safe = VesselParams.DCPA_safe
        curr_dist = float(np.hypot(x_os[0] - x_ts[0], x_os[1] - x_ts[1]))
        
        # Identify fixed COLREGs scenario for the evaluation step
        scenario = "Unknown"
        if x_ts is not None:
            beta_init = RiskCalculator.calculate_relative_bearing(x_os, x_ts)
            scenario = RiskCalculator.classify_colreg_scenario(beta_init)

        # ----------------------------------------------------------------------
        # Mode A: Shared Intent Route Available
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # Mode A: Shared Intent Route Available
        # ----------------------------------------------------------------------
        if w_ts_delayed is not None and len(w_ts_delayed) >= 2:
            # 1. Check exit conditions ONLY after vessels have truly passed each other
            if cls._mode_a_active:
                # Relative position vector from OS to TS
                dx = x_ts[0] - x_os[0]
                dy = x_ts[1] - x_os[1]
                
                # Check if TS is astern of OS relative to OS heading
                # dot product of OS forward direction and vector to TS < 0 means TS is behind OS
                cos_psi = np.cos(x_os[2])
                sin_psi = np.sin(x_os[2])
                longitudinal_rel = dx * cos_psi + dy * sin_psi
                
                # Truly passed CPA: TS is physically behind OS AND vessels are separating
                is_astern = longitudinal_rel < -0.5
                passed_cpa = (tcpa < -1.0 and curr_dist < 6.0) or is_astern
                cleared_distance = curr_dist > (d_safe * 1.8)

                if passed_cpa and cleared_distance:
                    cls._mode_a_active = False
                    cls._w_evasive_latched = None
                    cls._p_evasive_latched = 1.0
                    return np.copy(w_os), 0.0, 1.0, "State A.2"

                # Hold the active evasive route until safely past
                if cls._w_evasive_latched is not None:
                    return cls._w_evasive_latched, 0.0, cls._p_evasive_latched, "State A.1"
                
            cls._mode_b_active = False

            # Horizon planning steps
            hor_time = max(100.0, tcpa + 20.0) if tcpa > 0.0 else 60.0
            dt_sim = 0.5
            steps = int(hor_time / dt_sim)

            ts_traj, _ = cls._sample_route(w_ts_delayed, x_ts[:2], u_ts, steps, dt_sim)

            diffs_os = np.diff(w_os[:, :2], axis=0)
            downstream_indices = []
            for i in range(len(w_os) - 1):
                seg_dir = diffs_os[i]
                v_to_wp = w_os[i + 1, :2] - x_os[:2]
                if np.dot(seg_dir, v_to_wp) > 0.5:
                    downstream_indices.append(i + 1)

            unsailed_wps = w_os[downstream_indices[0]:, :2] if len(downstream_indices) > 0 else w_os[-1:, :2]
            w_os_base = np.vstack([x_os[:2], unsailed_wps])

            os_traj, s_os = cls._sample_route(w_os_base, x_os[:2], u_os, steps, dt_sim)
            dists = np.hypot(os_traj[:, 0] - ts_traj[:, 0], os_traj[:, 1] - ts_traj[:, 1])
            k_cpa = int(np.argmin(dists))
            min_dist = float(dists[k_cpa])

            # Trigger condition: Only activate if an actual CPA risk exists
            if min_dist >= (d_safe * 1.5) and curr_dist > (d_safe * 1.5):
                return np.copy(w_os), 0.0, 1.0, "State A.2"

            t_start_a1 = time.perf_counter()

            best_cost = float("inf")
            best_chi = 0.0
            best_p = 1.0

            for chi in cls.chi_candidates:
                for p_cand in cls.p_candidates:
                    cost, _, _ = cls._evaluate_candidate_hazard(
                        x_os, chi, p_cand, ts_traj, steps, dt_sim, d_safe, u_nominal, canal_polygons, scenario
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_chi = chi
                        best_p = p_cand

            # Head-On / Starboard evasion offset
            req_offset = 2.0  # Guarantees clearing TS (y=0) while keeping inside canal wall (y=3.0)

            P_cpa = os_traj[k_cpa]
            s_cpa = s_os + (u_os * k_cpa * dt_sim)

            diffs_base = np.diff(w_os_base[:, :2], axis=0)
            seg_lens_base = np.hypot(diffs_base[:, 0], diffs_base[:, 1])
            accum = 0.0
            cpa_seg_idx = 0
            for idx, length in enumerate(seg_lens_base):
                if accum + length >= s_cpa:
                    cpa_seg_idx = idx
                    break
                accum += length

            seg_dir = diffs_base[cpa_seg_idx] / max(seg_lens_base[cpa_seg_idx], 1e-4)
            n_stb = np.array([-seg_dir[1], seg_dir[0]])  # Right / Starboard normal

            # Latch a stable evasive polyline: [Start, Evade, Rejoin]
            W_evade = P_cpa + req_offset * n_stb
            
            # Anchor start of evasive leg at initial decision point, not continuously shifting with x_os
            rejoin_idx = min(cpa_seg_idx + 1, len(w_os_base) - 1)
            remaining_wps = w_os_base[rejoin_idx:, :2]

            cls._w_evasive_latched = np.vstack([x_os[:2], W_evade, remaining_wps])
            cls._mode_a_active = True
            cls._p_evasive_latched = best_p

            calc_duration_ms = (time.perf_counter() - t_start_a1) * 1000.0
            print(
                f"\033[93m[State A.1 Route Plan] Computation Time: {calc_duration_ms:6.2f} ms\033[0m",
                flush=True
            )

            return cls._w_evasive_latched, 0.0, best_p, "State A.1"

        # ----------------------------------------------------------------------
        # Mode B: Reactive Fallback (No Intent)
        # ----------------------------------------------------------------------
        cls._mode_a_active = False
        cls._w_evasive_latched = None

        domain_breached, _ = RiskCalculator.check_ship_domain_breach(x_os, x_ts)
        
        threshold_cpa = VesselParams.R_lateral + VesselParams.B
        cpa_risk = (dcpa < threshold_cpa) and (0.0 <= tcpa <= VesselParams.TCPA_safe)
        
        close_quarters = (curr_dist < d_safe * 1.5) and (dcpa < threshold_cpa) and (tcpa > -3.0)
        risk_active = cpa_risk or domain_breached or close_quarters

        if cls._mode_b_active:
            # Exit Mode B only after TCPA has passed AND vessels are separating
            # (Independent of North/South travel direction)
            has_passed = (tcpa < -2.0) and (curr_dist > max(d_safe * 1.5, 3.0))
            if has_passed:
                cls._mode_b_active = False
        else:
            if risk_active:
                cls._mode_b_active = True

        if cls._mode_b_active:
            urgency = np.clip((d_safe - dcpa) / max(d_safe, 1e-3), 0.0, 1.0)
            psi_headon = np.radians(30.0 + (60.0 - 30.0) * urgency)

            if scenario == "Head-On":
                psi_ca = psi_headon  # +offset = Starboard turn
                p_ca = 0.5 if urgency > 0.6 else 1.0
            elif scenario in ["Crossing_A", "Crossing_B"]:
                # +45.0 deg (Starboard turn to pass astern) per Equation (3.50)
                psi_ca = np.radians(45.0)
                p_ca = 0.5 if curr_dist < (d_safe * 1.5) else 1.0
            elif scenario == "Overtaking":
                psi_ca = np.radians(30.0)
                p_ca = 1.0
            else:
                psi_ca = np.radians(30.0)
                p_ca = 1.0

            if curr_dist < d_safe * 0.8:
                p_ca = 0.0

            # --- NEW: REACTIVE GROUNDING AVOIDANCE ---
            if canal_polygons is not None:
                # Project the OS position 5 seconds into the future using the proposed heading
                vx = u_os * np.cos(x_os[2] + psi_ca)
                vy = u_os * np.sin(x_os[2] + psi_ca)
                future_pos = (x_os[0] + vx * 5.0, x_os[1] + vy * 5.0)
                
                # Check distance to canal walls along this projected path
                projected_line = LineString([(x_os[0], x_os[1]), future_pos])
                d_static_future = float(projected_line.distance(canal_polygons))

                # If the evasive turn puts us into the wall, kill speed and straighten out
                if d_static_future <= VesselParams.d_safe_static:
                    p_ca = 0.0    # Emergency stop to yield
                    psi_ca = 0.0  # Re-align parallel to the channel to minimize footprint
            # -----------------------------------------

            return np.copy(w_os), float(psi_ca), float(p_ca), "State B.1"

        return np.copy(w_os), 0.0, 1.0, "State B.2"