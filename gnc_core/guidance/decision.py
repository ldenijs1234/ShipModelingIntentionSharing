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
    _w_nominal = None
    _latched_scenario = None
    _t_offset_latched = 20.0
    _chi_ca_latched = 0.0

    k_chi_stb = 5.0
    k_chi_port = 5.2
    k_dchi_stb = 4.0   
    k_dchi_port = 4.2   
    k_p = 10.0
    
    k_safety = 1000.0
    k_grounding = 5.0
    k_colregs = 10.0

    chi_candidates = np.radians(
        np.array([
            -90.0, -75.0, -60.0, -45.0, -30.0, -15.0,
            0.0,
            15.0, 30.0, 45.0, 60.0, 75.0, 90.0,
        ])
    )

    p_candidates = np.array([1.0, 0.5, 0.0])

    # Tactical lookahead offset candidates relative to TCPA (seconds)
    t_offset_candidates = np.array([5.0, 10.0, 15.0, 20.0])

    @classmethod
    def _get_active_nominal_track(cls, path_wps: np.ndarray, pos: np.ndarray) -> np.ndarray:
        if len(path_wps) < 2:
            return path_wps
        
        diffs = np.diff(path_wps[:, :2], axis=0)
        for i in range(len(path_wps) - 1):
            seg_dir = diffs[i]
            v_to_wp = path_wps[i + 1, :2] - pos[:2]
            if np.dot(seg_dir, v_to_wp) > 0.0: 
                return path_wps[i:]
        
        return path_wps[-2:] if len(path_wps) >= 2 else path_wps

    @classmethod
    def _slice_path_forward(cls, path_wps: np.ndarray, pos: np.ndarray) -> np.ndarray:
        if len(path_wps) < 2:
            return path_wps
        
        diffs = np.diff(path_wps[:, :2], axis=0)
        for i in range(len(path_wps) - 1):
            seg_dir = diffs[i]
            v_to_wp = path_wps[i + 1, :2] - pos[:2]
            if np.dot(seg_dir, v_to_wp) > 0.0: 
                return path_wps[i + 1:]
        
        return path_wps[-1:]

    @classmethod
    def calculate_adaptive_tcpa_trigger(
        cls,
        u_os: float,
        x_os: np.ndarray,
        x_ts: np.ndarray,
        d_safe: float,
        scenario: str
    ) -> float:
        """
        Dynamically calculates the initiation lookahead TCPA_trigger based on
        hydrodynamic lateral transfer limits and relative encounter velocity.
        """
        # 1. Lateral transfer time to clear d_safe at nominal 15 deg heading change
        chi_nominal_rad = np.radians(15.0)
        u_effective = max(u_os, 0.15)
        tau_lag = 2.5  # Hull yaw inertia + rudder response lag (seconds)
        t_transfer = (d_safe / (u_effective * np.sin(chi_nominal_rad))) + tau_lag

        # 2. Closing velocity calculation
        v_os = np.array([u_os * np.cos(x_os[2]), u_os * np.sin(x_os[2])])
        u_ts = float(x_ts[3])
        v_ts = np.array([u_ts * np.cos(x_ts[2]), u_ts * np.sin(x_ts[2])])
        v_rel = float(np.linalg.norm(v_os - v_ts))

        # 3. Approach time across the tactical boundary
        d_buffer = d_safe * 4.0  # Safe longitudinal buffer distance
        t_approach = d_buffer / max(v_rel, 0.15)

        # 4. Scenario-dependent baseline
        if scenario == "Overtaking":
            # For overtaking, relative speed is small; avoid triggering at huge distances
            raw_trigger = max(t_transfer, 5.0)
        else:
            raw_trigger = max(t_transfer, t_approach)

        # 5. Bound trigger to inland fairway limits (between 5s and 20s)
        tcpa_trigger = float(np.clip(raw_trigger, 5.0, 20.0))
        return tcpa_trigger

    @classmethod
    def delta_chi(cls, chi_ca: float, chi_ca_last: float) -> float:
        """
        Exact transition cost formulation from Akdag (2022).
        Penalizes changes in heading relative to the actively executed maneuver.
        """
        d_chi = chi_ca - chi_ca_last
        
        # Check if Own Ship is actively turned (>5 deg) AND attempting to cross over to the opposite side
        is_reversing = (abs(chi_ca_last) > np.radians(5.0)) and (np.sign(chi_ca) * np.sign(chi_ca_last) < -0.5)
        
        if is_reversing:
            # Reversal penalty to suppress S-curve snap-back
            k_weight = 20.0
        elif d_chi >= 0.0:
            k_weight = cls.k_dchi_stb
        else:
            k_weight = cls.k_dchi_port
            
        return float(k_weight * (d_chi ** 2))

    @classmethod
    def _evaluate_candidate_hazard(
        cls,
        x_os: np.ndarray,
        chi: float,
        p_cand: float,
        ts_traj: np.ndarray,
        cand_traj: np.ndarray,
        steps: int,
        dt_sim: float,
        d_safe: float,
        d_warn: float,
        canal_polygons = None,
        scenario: str = "Unknown",
        k_w1: int = 0,
        chi_active: float = 0.0 
    ) -> Tuple[float, float, int]:
        
        os_x = cand_traj[:, 0]
        os_y = cand_traj[:, 1]

        dists = np.hypot(os_x - ts_traj[:, 0], os_y - ts_traj[:, 1])
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist <= d_safe:
            # Physical domain breach: heavy penalty to forbid near-collisions
            j_safety = 500.0 + 500.0 * ((d_safe / max(min_dist, 0.05)) ** 2.0)
        elif min_dist < d_warn:
            # Soft proximity warning: scales from 0.0 (at 2.0m) up to 20.0 (at 1.0m)
            # This mirrors Akdag's H1 ~ 20-30 hazard when cutting close to TS
            proximity_factor = (d_warn - min_dist) / (d_warn - d_safe)
            j_safety = 20.0 * (proximity_factor ** 2.0)
        else:
            j_safety = 0.0
            
        j_grounding = 0.0
        if canal_polygons is not None:
            eval_steps = min(steps, min_idx + int(10.0 / dt_sim))
            eval_steps = max(eval_steps, 10)  
            
            traj_line = LineString(np.column_stack((os_x[:eval_steps], os_y[:eval_steps])))
            min_static_dist = float(traj_line.distance(canal_polygons))
            
            if min_static_dist <= VesselParams.R_lateral:
                j_grounding = 50000.0  
            elif min_static_dist <= VesselParams.d_safe_static:
                j_grounding = cls.k_grounding * ((VesselParams.d_safe_static / max(min_static_dist, 0.05)) ** 2.5)

        if scenario in ["Overtaking"]:
            k_w = cls.k_chi_stb
        else:
            k_w = cls.k_chi_stb if chi >= 0.0 else cls.k_chi_port
        j_control = k_w * (chi**2)
        j_speed = cls.k_p * (1.0 - p_cand)

        j_colregs = 0.0
        if scenario != "Unknown":
            dx = np.diff(os_x)
            dy = np.diff(os_y)
            dx = np.append(dx, dx[-1])
            dy = np.append(dy, dy[-1])
            psi_cand_array = np.arctan2(dy, dx)

            phi_array = np.arctan2(ts_traj[:, 1] - os_y, ts_traj[:, 0] - os_x)
            beta_array = np.degrees(phi_array - psi_cand_array)
            beta_array = (beta_array + 180.0) % 360.0 - 180.0

            beta_eval = beta_array[k_w1:]

            if scenario == "Head-On":
                mu_array = (beta_array >= -13.0) | (abs(chi) < 1e-3)
            elif scenario in ["Crossing_A", "Crossing_B"]:
                mu_array = (beta_array >= 0.0) 
            else:
                mu_array = np.zeros_like(beta_array, dtype=bool)

            if np.any(mu_array):
                j_colregs = cls.k_colregs

        j_trans = cls.delta_chi(chi, chi_active)

        total_cost = j_safety + j_grounding + j_control + j_speed + j_colregs + j_trans
        return total_cost, min_dist, min_idx, (j_safety, j_speed, j_trans, j_grounding, j_control)

    @classmethod
    def _sample_route(
        cls, wps: np.ndarray, pos: np.ndarray, speed: float, steps: int, dt: float
    ) -> Tuple[np.ndarray, float]:
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
        canal_polygons = None,
        canal_width: float = 6.0
    ) -> Tuple[np.ndarray, float, float, str]:
        
        if not cls._mode_a_active and not cls._mode_b_active:
            cls._w_nominal = np.copy(w_os)

        nom = cls._w_nominal if cls._w_nominal is not None else w_os

        if x_ts is None:
            cls._mode_a_active = False
            cls._mode_b_active = False
            cls._w_evasive_latched = None
            cls._p_evasive_latched = 1.0
            cls._latched_scenario = None
            cls._chi_ca_latched = 0.0
            active_track = cls._get_active_nominal_track(nom, x_os[:2])
            return active_track, 0.0, 1.0, "State B.2"

        u_os = float(x_os[3]) if abs(x_os[3]) > 0.05 else float(u_nominal)
        u_ts = float(x_ts[3]) if abs(x_ts[3]) > 0.05 else float(u_nominal)

        if u_os < 0.10 or u_ts < 0.10:
            cls._mode_a_active = False
            cls._mode_b_active = False
            cls._w_evasive_latched = None
            cls._p_evasive_latched = 1.0
            cls._latched_scenario = None
            cls._chi_ca_latched = 0.0
            active_track = cls._get_active_nominal_track(nom, x_os[:2])
            return active_track, 0.0, 1.0, "State B.2"

        d_safe = VesselParams.DCPA_safe
        curr_dist = float(np.hypot(x_os[0] - x_ts[0], x_os[1] - x_ts[1]))
        
        scenario = "Unknown"
        if x_ts is not None:
            beta_init = RiskCalculator.calculate_relative_bearing(x_os, x_ts)
            beta_deg = np.degrees(beta_init) % 360.0
            raw_scenario = RiskCalculator.classify_colreg_scenario(
                beta_deg, psi_os=float(x_os[2]), psi_ts=float(x_ts[2])
            )
            
            if cls._mode_a_active and cls._latched_scenario is not None:
                scenario = cls._latched_scenario
            else:
                scenario = raw_scenario
                cls._latched_scenario = scenario

        # ----------------------------------------------------------------------
        # Unified Predictive MPC (Used for both Intent and No Intent)
        # ----------------------------------------------------------------------
        cls._mode_b_active = False

        hor_time = max(100.0, tcpa + 20.0) if tcpa > 0.0 else 60.0
        dt_sim = 0.5
        steps = int(hor_time / dt_sim)

        # 1. Generate Target Ship Trajectory & Define State Prefix
        if w_ts_delayed is not None and len(w_ts_delayed) >= 2:
            ts_traj, _ = cls._sample_route(w_ts_delayed, x_ts[:2], u_ts, steps, dt_sim)
            state_prefix = "A"
        else:
            extrapolation_time = steps * dt_sim
            ts_p0 = x_ts[:2]
            ts_p1 = ts_p0 + np.array([np.cos(x_ts[2]), np.sin(x_ts[2])]) * max(u_ts, 0.1) * extrapolation_time
            w_ts_extrapolated = np.vstack((ts_p0, ts_p1))
            ts_traj, _ = cls._sample_route(w_ts_extrapolated, x_ts[:2], u_ts, steps, dt_sim)
            state_prefix = "B"

        w_os_base = cls._get_active_nominal_track(nom, x_os[:2])

        # 2. Check nominal route clearance
        os_traj_nom, s_os = cls._sample_route(w_os_base, x_os[:2], u_os, steps, dt_sim)
        dists_nom = np.hypot(os_traj_nom[:, 0] - ts_traj[:, 0], os_traj_nom[:, 1] - ts_traj[:, 1])
        k_cpa = int(np.argmin(dists_nom))
        min_dist_nom = float(dists_nom[k_cpa])
        s_cpa = s_os + (u_os * k_cpa * dt_sim)

        # 3. Encounter exit conditions
        dx = x_ts[0] - x_os[0]
        dy = x_ts[1] - x_os[1]
        longitudinal_rel = dx * np.cos(x_os[2]) + dy * np.sin(x_os[2])
        is_astern = longitudinal_rel < -0.5

        if scenario in ["Overtaking"]:
            has_passed = longitudinal_rel < -(5.0 * VesselParams.L)
            if (tcpa < -10.0) or (has_passed and curr_dist > 4.0):
                cls._mode_a_active = False
                cls._w_evasive_latched = None
                cls._chi_ca_latched = 0.0
                active_track = cls._get_active_nominal_track(nom, x_os[:2])
                return active_track, 0.0, 1.0, f"State {state_prefix}.2"
        else:
            if state_prefix == "A":
                passed_cpa = (tcpa < -2.5) or (is_astern and curr_dist < 4.0)
                separating = curr_dist > (d_safe * 1.5) and (tcpa < 0.0 or tcpa > 100.0)
                threat_cleared = (min_dist_nom >= d_safe * 2.0 and curr_dist > d_safe * 2.0)
                
                if (passed_cpa and separating) or threat_cleared:
                    cls._mode_a_active = False
                    cls._w_evasive_latched = None
                    cls._chi_ca_latched = 0.0
                    active_track = cls._get_active_nominal_track(nom, x_os[:2])
                    return active_track, 0.0, 1.0, f"State {state_prefix}.2"
            else:
                passed_cpa_strict = (tcpa <= -2.0) and (curr_dist > d_safe * 1.5)
                physically_clear = (curr_dist > d_safe * 3.0) and is_astern
                
                if passed_cpa_strict or physically_clear:
                    cls._mode_a_active = False
                    cls._w_evasive_latched = None
                    cls._chi_ca_latched = 0.0
                    active_track = cls._get_active_nominal_track(nom, x_os[:2])
                    return active_track, 0.0, 1.0, f"State {state_prefix}.2"

        # 4. Prospective Safety Verification of the Latched Plan
        if cls._mode_a_active and cls._w_evasive_latched is not None:
            os_traj_latched, _ = cls._sample_route(
                cls._w_evasive_latched, x_os[:2], u_os * cls._p_evasive_latched, steps, dt_sim
            )
            dists_latched = np.hypot(
                os_traj_latched[:, 0] - ts_traj[:, 0],
                os_traj_latched[:, 1] - ts_traj[:, 1]
            )
            min_dist_latched = float(np.min(dists_latched))

            if min_dist_latched >= d_safe:
                sliced_latched = cls._slice_path_forward(cls._w_evasive_latched, x_os[:2])
                cls._w_evasive_latched = np.vstack([x_os[:2], sliced_latched])
                return cls._w_evasive_latched, 0.0, cls._p_evasive_latched, f"State {state_prefix}.1"
            else:
                print(
                    f"\033[91m[State {state_prefix}.1 Early Re-evaluation] Latched route compromised "
                    f"(projected min_dist={min_dist_latched:.2f}m < {d_safe:.2f}m). Recalculating now...\033[0m",
                    flush=True
                )
                cls._w_evasive_latched = None
            
        # 5. Planning Trigger Check
        threat_exists = min_dist_nom < d_safe
        if not cls._mode_a_active and not threat_exists:
            active_track = cls._get_active_nominal_track(nom, x_os[:2])
            return active_track, 0.0, 1.0, f"State {state_prefix}.2"

        # ----------------------------------------------------------------------
        # 6. Candidate Evasion Generation with Dynamic t_offset Optimization
        # ----------------------------------------------------------------------
        t_start_a1 = time.perf_counter()

        diffs_base = np.diff(cls._w_nominal[:, :2], axis=0)
        seg_lens_base = np.hypot(diffs_base[:, 0], diffs_base[:, 1])

        d_warn = float(np.clip(0.25 * canal_width, d_safe * 1.5, d_safe * 3.5))

        # Calculate adaptive initiation trigger based on physics
        adaptive_tcpa = cls.calculate_adaptive_tcpa_trigger(
            u_os=u_os,
            x_os=x_os,
            x_ts=x_ts,
            d_safe=d_safe,
            scenario=scenario
        )

        # Formulate candidate offsets centered tightly around the adaptive trigger
        # e.g., if adaptive_tcpa = 12.0s -> candidates = [10.0, 12.0, 14.0]
        # cls.t_offset_candidates = np.array([adaptive_tcpa])
        cls.t_offset_candidates = np.array([
            max(5.0, adaptive_tcpa - 5.0),
            adaptive_tcpa,
            min(20.0, adaptive_tcpa + 5.0)
        ])

        # Baseline tactical runway D
        dist_to_cpa_actual = max(0.0, s_cpa - s_os)
        D = float(min(adaptive_tcpa * u_os, max(dist_to_cpa_actual, 3.0)))

        best_cost = float("inf")
        best_chi = 0.0
        best_p = 1.0
        best_t_offset = 20.0
        best_wps = None

        all_evals = []

        dist_to_cpa_actual = max(0.0, s_cpa - s_os)

        # Iterate through tactical time-to-CPA offsets
        for t_off in cls.t_offset_candidates:
            k_offset = int(t_off / dt_sim)
            if cls._mode_a_active:
                # Mid-turn replanning: Start evasion immediately from current vessel position
                k_w1 = 0
                W_1 = np.copy(x_os[:2])
                min_clear_x = float(x_ts[0] + 8.0) if x_ts is not None else float(x_os[0] + 15.0)
                # Find nominal waypoint index that is safely past TS
                k_w3 = len(os_traj_nom) - 1
                for idx, pt in enumerate(os_traj_nom):
                    if pt[0] > min_clear_x:
                        k_w3 = idx
                        break
                W_3 = os_traj_nom[k_w3]
                
                # Use mission path tangent forward as base orientation
                diff_nom = cls._w_nominal[min(1, len(cls._w_nominal)-1), :2] - cls._w_nominal[0, :2]
                norm_base = float(np.hypot(diff_nom[0], diff_nom[1]))
                u_nom = (diff_nom / norm_base) if norm_base > 1e-3 else np.array([1.0, 0.0])
            else:
                # Initial plan from nominal mission track
                k_w1 = max(0, k_cpa - k_offset)
                W_1 = os_traj_nom[k_w1]

                k_w3 = min(len(os_traj_nom) - 1, k_cpa + k_offset)
                W_3 = os_traj_nom[k_w3]

                v_nom = os_traj_nom[k_cpa] - W_1
                norm = float(np.hypot(v_nom[0], v_nom[1]))
                u_nom = (v_nom / norm) if norm > 1e-3 else np.array([1.0, 0.0])

            s_w1 = s_os + (u_os * k_w1 * dt_sim)
            w1_idx = 0
            accum_w1 = 0.0
            for idx, length in enumerate(seg_lens_base):
                accum_w1 += length
                if accum_w1 > s_w1:
                    w1_idx = idx + 1
                    break
            w1_idx = min(w1_idx, len(cls._w_nominal) - 1)
            pre_w1_wps = cls._w_nominal[:w1_idx, :2]

            s_w3 = s_os + (u_os * k_w3 * dt_sim)
            rejoin_idx = len(cls._w_nominal) - 1
            accum_rejoin = 0.0
            for idx, length in enumerate(seg_lens_base):
                accum_rejoin += length
                if accum_rejoin > s_w3:
                    rejoin_idx = idx + 1
                    break
            rejoin_idx = min(rejoin_idx, len(cls._w_nominal) - 1)
            remaining_wps = cls._w_nominal[rejoin_idx:, :2]

            for chi in cls.chi_candidates:
                for p_cand in cls.p_candidates:
                    cand_wps = None
                    # Nominal track (chi == 0) does not depend on t_offset; evaluate once
                    if abs(chi) < 1e-3 and p_cand > 0.99 and not cls._mode_a_active:
                        if t_off != cls.t_offset_candidates[0]:
                            continue
                        cand_wps = np.copy(w_os_base)
                    else:
                        cos_chi = np.cos(chi)
                        sin_chi = np.sin(chi)
                        u_evade = np.array([
                            u_nom[0] * cos_chi - u_nom[1] * sin_chi,
                            u_nom[0] * sin_chi + u_nom[1] * cos_chi
                        ])
                        
                        W_2 = W_1 + u_evade * D

                        if scenario in ["Overtaking"]:
                            u_cand_os = u_os * p_cand
                            u_rel = max(u_cand_os - u_ts, 0.05)
                            t_clear = (2.5 * VesselParams.L) / u_rel
                            d_pass = max(u_cand_os * t_clear, 4.0 * VesselParams.L)

                            s_w1_ov = max(0.0, s_cpa - D)
                            s_w2_ov = s_cpa
                            s_wpass = s_cpa + d_pass
                            s_wrejoin = s_wpass + D

                            def get_curved_offset_point(s_target, lateral_offset):
                                accum = 0.0
                                for i, seg_len in enumerate(seg_lens_base):
                                    if accum + seg_len >= s_target or i == len(seg_lens_base) - 1:
                                        ratio = np.clip((s_target - accum) / max(seg_len, 1e-4), 0.0, 1.0)
                                        pt_centerline = cls._w_nominal[i, :2] + ratio * diffs_base[i]
                                        t_vec = diffs_base[i] / max(seg_len, 1e-4)
                                        n_vec = np.array([-t_vec[1], t_vec[0]])
                                        return pt_centerline + n_vec * lateral_offset
                                    accum += seg_len
                                return cls._w_nominal[-1, :2]

                            lat_offset = np.sign(chi) * max(d_safe * 1.2, 1.2) if abs(chi) > 1e-3 else 0.0

                            W_1_curved = get_curved_offset_point(s_w1_ov, 0.0)
                            W_2_curved = get_curved_offset_point(s_w2_ov, lat_offset)
                            W_pass_curved = get_curved_offset_point(s_wpass, lat_offset)
                            W_rejoin_curved = get_curved_offset_point(s_wrejoin, 0.0)

                            rejoin_idx_ov = len(cls._w_nominal) - 1
                            accum_track = 0.0
                            for idx, length in enumerate(seg_lens_base):
                                accum_track += length
                                if accum_track > s_wrejoin:
                                    rejoin_idx_ov = idx + 1
                                    break
                            remaining_wps_curved = cls._w_nominal[rejoin_idx_ov:, :2]

                            if len(pre_w1_wps) > 0:
                                cand_wps = np.vstack([pre_w1_wps, W_1_curved, W_2_curved, W_pass_curved, W_rejoin_curved, remaining_wps_curved])
                            else:
                                cand_wps = np.vstack([W_1_curved, W_2_curved, W_pass_curved, W_rejoin_curved, remaining_wps_curved])
                        else:
                            if cls._mode_a_active:
                                # When mid-turn, W_1 is already x_os[:2], so route heads directly to W_2 then rejoins
                                cand_wps = np.vstack([W_1, W_2, W_3, remaining_wps])
                            else:
                                if len(pre_w1_wps) > 0:
                                    cand_wps = np.vstack([pre_w1_wps, W_1, W_2, W_3, remaining_wps])
                                else:
                                    cand_wps = np.vstack([W_1, W_2, W_3, remaining_wps])
                                sliced_route = cls._slice_path_forward(cand_wps, x_os[:2])
                                cand_wps = np.vstack([x_os[:2], sliced_route])

                    cand_traj, _ = cls._sample_route(cand_wps, x_os[:2], u_os * p_cand, steps, dt_sim)

                    cost, min_dist, min_idx, cost_terms = cls._evaluate_candidate_hazard(
                        x_os, chi, p_cand, ts_traj, cand_traj, steps, dt_sim, d_safe, d_warn, canal_polygons, scenario, k_w1, chi_active=cls._chi_ca_latched
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_chi = chi
                        best_p = p_cand
                        best_t_offset = t_off
                        best_wps = cand_wps

                    all_evals.append({
                        'chi': np.degrees(chi),
                        'p': p_cand,
                        't_off': t_off,
                        'cost': cost,
                        'j_safe': cost_terms[0],
                        'd_min': min_dist,
                        'j_spd': cost_terms[1],
                        'j_trans': cost_terms[2],
                        'j_ground': cost_terms[3],
                        'wps': cand_wps
                    })

        calc_duration_ms = (time.perf_counter() - t_start_a1) * 1000.0
        # -------------------------------------------------------------
        # DEBUG INSTRUMENTATION: Log Top Candidates & Cost Breakdown
        # -------------------------------------------------------------
        # if cls._mode_a_active:
        #     print(f"\n{'='*75}", flush=True)
        #     print(f"[DEBUG RE-EVALUATION] OS: x={x_os[0]:.2f}, y={x_os[1]:.2f}, psi={np.degrees(x_os[2]):.1f}°, u={x_os[3]:.2f} m/s", flush=True)
        #     if x_ts is not None:
        #         print(f"[DEBUG RE-EVALUATION] TS: x={x_ts[0]:.2f}, y={x_ts[1]:.2f}, psi={np.degrees(x_ts[2]):.1f}°, u={x_ts[3]:.2f} m/s", flush=True)
        #     print(f"[DEBUG RE-EVALUATION] Active Latched: Chi={np.degrees(cls._chi_ca_latched):.1f}°, P={cls._p_evasive_latched:.2f}", flush=True)
        #     print(f"{'-'*75}", flush=True)
        #     print(f"{'Chi':>6} | {'P':>4} | {'t_off':>5} | {'Cost':>7} | {'J_safe':>7} | {'d_min':>5} | {'J_spd':>5} | {'J_trans':>7} | {'J_ground':>8}", flush=True)
        #     print(f"{'-'*75}", flush=True)
        log_msg = (
            f"Opt Chi: {np.degrees(best_chi):.1f}° | Opt P: {best_p:.2f} | "
            f"Opt t_off: {best_t_offset:.1f}s | Cost: {best_cost:.2f} | {calc_duration_ms:6.2f} ms"
        )
        if cls._mode_a_active:
            print(f"\033[93m[State {state_prefix}.1 Re-planned Early] {log_msg}\033[0m", flush=True)
        else:
            print(f"\033[92m[State {state_prefix}.1 Early Plan] {log_msg}\033[0m", flush=True)

        # # <--- [PASTE YOUR SNIPPET HERE] ---------------------------------
        # if cls._mode_a_active:
        #     # Sort and print the top 5 candidates
        #     sorted_cands = sorted(all_evals, key=lambda c: c['cost'])[:5]
        #     print(f"\n{'Chi':>6} | {'P':>4} | {'t_off':>5} | {'Cost':>7} | {'J_safe':>7} | {'d_min':>5} | {'J_spd':>5} | {'J_trans':>7} | {'J_ground':>8}", flush=True)
        #     print(f"{'-'*75}", flush=True)
        #     for c in sorted_cands:
        #         print(f"{c['chi']:6.1f}° | {c['p']:4.2f} | {c['t_off']:5.1f} | {c['cost']:7.2f} | "
        #             f"{c['j_safe']:7.2f} | {c['d_min']:5.2f}m | {c['j_spd']:5.2f} | {c['j_trans']:7.2f} | {c['j_ground']:8.2f}", flush=True)

        #     # Print the exact waypoints of the winning plan
        #     print(f"{'-'*75}", flush=True)
        #     print(f"[DEBUG WINNER WPs]:\n{best_wps[:4]}", flush=True)
        #     print(f"{'='*75}\n", flush=True)
        # # -----------------------------------------------------------------
        
        cls._chi_ca_latched = float(best_chi)
        cls._p_evasive_latched = float(best_p)
        cls._t_offset_latched = float(best_t_offset)
        cls._w_evasive_latched = best_wps
        cls._mode_a_active = True

        return cls._w_evasive_latched, 0.0, cls._p_evasive_latched, f"State {state_prefix}.1"


        # # ----------------------------------------------------------------------
        # # Mode B: Reactive Fallback (No Intent)
        # # ----------------------------------------------------------------------
        # cls._mode_a_active = False
        # cls._w_evasive_latched = None

        # domain_breached, _ = RiskCalculator.check_ship_domain_breach(x_os, x_ts)
        
        # cpa_risk = (dcpa < d_safe) and (0.0 <= tcpa <= VesselParams.TCPA_safe)
        
        # close_quarters = (curr_dist < d_safe * 1.5) and (dcpa < d_safe) and (tcpa > -3.0)
        # risk_active = cpa_risk or domain_breached or close_quarters

        # # Relative longitudinal distance along OS heading (+ = TS is ahead of OS, - = TS is behind OS)
        # os_heading = x_os[2]
        # dx_rel = (x_ts[0] - x_os[0]) * np.cos(os_heading) + (x_ts[1] - x_os[1]) * np.sin(os_heading)
        # dy_rel = abs(x_os[1] - x_ts[1])

        # # --- Hysteresis Latching Logic for Mode B ---
        # if cls._mode_b_active:
        #     # Check exit conditions using the latched scenario
        #     if scenario in "Overtaking":
        #         # Exit ONLY when OS has physically pulled ahead of TS
        #         if dx_rel < (-VesselParams.L * 1.5):
        #             cls._mode_b_active = False
        #             cls._latched_scenario = None
        #     else:
        #         passed_cpa = (tcpa <= -2.0) and (curr_dist > (d_safe * 1.3))
        #         if passed_cpa:
        #             cls._mode_b_active = False
        #             cls._latched_scenario = None
        # else:
        #     if risk_active:
        #         cls._mode_b_active = True
        #         cls._latched_scenario = scenario

        # if cls._mode_b_active:
        #     urgency = np.clip((d_safe - dcpa) / max(d_safe, 1e-3), 0.0, 1.0)
        #     psi_headon = np.radians(30.0 + (60.0 - 30.0) * urgency)
        #     psi_crossing = np.radians(20.0 + (45.0 - 20.0) * urgency)
        #     psi_overtaking_mag = np.radians(15.0 + (30.0 - 15.0) * urgency)

        #     if scenario == "Head-On":
        #         psi_ca = psi_headon 
        #         p_ca = 0.5 if urgency > 0.6 else 1.0
        #     elif scenario == "Crossing_A":  # Give-Way (TS on Starboard)
        #         # Alter course substantially to starboard to pass astern of TS
        #         psi_ca = psi_crossing
        #         p_ca = 0.5 if urgency > 0.5 else 1.0

        #     elif scenario == "Crossing_B":  # Stand-On (TS on Port)
        #         # Under Rule 17, Stand-On holds course/speed unless TS fails to act.
        #         # If Mode B triggers for Stand-On, it is an in-extremis maneuver:
        #         # Steer starboard, but avoid turning port toward TS.
        #         psi_ca = psi_crossing
        #         p_ca = 0.5 if urgency > 0.7 else 1.0
        #     elif scenario == "Overtaking":
        #         p_ca = 1.0
        #         psi_os = x_os[2]
        #         probe_len = 10.0  # Sufficient reach to intersect canal boundaries

        #         # Default preference: Port overtaking (-25 deg)
        #         chosen_dir = -1.0  # -1.0 = Port, +1.0 = Starboard

        #         if canal_polygons is not None:
        #             # Starboard lateral probe (+90 deg perpendicular to heading)
        #             pt_stbd = (
        #                 x_os[0] + probe_len * np.cos(psi_os + np.pi / 2.0),
        #                 x_os[1] + probe_len * np.sin(psi_os + np.pi / 2.0)
        #             )
        #             # Port lateral probe (-90 deg perpendicular to heading)
        #             pt_port = (
        #                 x_os[0] + probe_len * np.cos(psi_os - np.pi / 2.0),
        #                 x_os[1] + probe_len * np.sin(psi_os - np.pi / 2.0)
        #             )

        #             dist_stbd = float(LineString([(x_os[0], x_os[1]), pt_stbd]).distance(canal_polygons))
        #             dist_port = float(LineString([(x_os[0], x_os[1]), pt_port]).distance(canal_polygons))

        #             # Minimum lateral clearance required to execute an evasive lane shift
        #             min_room_needed = VesselParams.DCPA_safe * 1.2 + VesselParams.d_safe_static

        #             # If port side is constrained and starboard offers more navigable water:
        #             # Overtake on Starboard (+15-30 deg)
        #             if dist_port < min_room_needed and dist_stbd > dist_port:
        #                 chosen_dir = 1.0
        #             else:
        #                 # If both sides are clear or port offers more room:
        #                 # Maintain Port overtaking (-15-30 deg) per inland regulations
        #                 chosen_dir = -1.0

        #         # 1. Step out toward the chosen side (-15-30 deg for Port, +15-30 deg for Starboard)
        #         if dy_rel < (VesselParams.DCPA_safe * 1.2):
        #             psi_ca = psi_overtaking_mag * chosen_dir
        #         else:
        #             # 2. Once clear in the overtaking lane: run parallel to pass TS safely
        #             psi_ca = 0.0
        #     else:
        #         psi_ca = np.radians(30.0)
        #         p_ca = 1.0

        #     if scenario != "Overtaking" and curr_dist < d_safe * 0.8:
        #         p_ca = 0.0

        #     if canal_polygons is not None:
        #         vx = u_os * np.cos(x_os[2] + psi_ca)
        #         vy = u_os * np.sin(x_os[2] + psi_ca)
        #         future_pos = (x_os[0] + vx * 5.0, x_os[1] + vy * 5.0)
                
        #         projected_line = LineString([(x_os[0], x_os[1]), future_pos])
        #         d_static_future = float(projected_line.distance(canal_polygons))

        #         if d_static_future <= VesselParams.d_safe_static:
        #             for scale_factor in [0.75, 0.5, 0.25]:
        #                 test_psi = psi_ca * scale_factor
        #                 test_line = LineString([
        #                     (x_os[0], x_os[1]),
        #                     (x_os[0] + u_os * np.cos(x_os[2] + test_psi) * 5.0,
        #                      x_os[1] + u_os * np.sin(x_os[2] + test_psi) * 5.0)
        #                 ])
        #                 if test_line.distance(canal_polygons) > VesselParams.d_safe_static:
        #                     psi_ca = test_psi
        #                     p_ca = 0.5 
        #                     break
        #             else:
        #                 p_ca = 0.0
        #                 psi_ca = 0.0

        #     return np.copy(w_os), float(psi_ca), float(p_ca), "State B.1"

        # active_track = cls._get_active_nominal_track(nom, x_os[:2])
        # return active_track, 0.0, 1.0, "State B.2"