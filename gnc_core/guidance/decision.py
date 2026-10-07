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

    k_chi_stb = 5.0
    k_chi_port = 5.2
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
        canal_polygons = None,
        scenario: str = "Unknown"
    ) -> Tuple[float, float, int]:
        
        os_x = cand_traj[:, 0]
        os_y = cand_traj[:, 1]

        dists = np.hypot(os_x - ts_traj[:, 0], os_y - ts_traj[:, 1])
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist <= (d_safe):
            j_safety = cls.k_safety * ((d_safe / max(min_dist, 0.05)) ** 4.0)
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

            if scenario == "Head-On":
                mu_array = (beta_array >= -2.0) | (abs(chi) < 1e-3)
            elif scenario in ["Crossing_A", "Crossing_B"]:
                mu_array = (beta_array >= 0.0) 
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
        
        if not cls._mode_a_active and not cls._mode_b_active:
            cls._w_nominal = np.copy(w_os)

        nom = cls._w_nominal if cls._w_nominal is not None else w_os

        if x_ts is None:
            cls._mode_a_active = False
            cls._mode_b_active = False
            cls._w_evasive_latched = None
            cls._latched_scenario = None
            active_track = cls._get_active_nominal_track(nom, x_os[:2])
            return active_track, 0.0, 1.0, "State B.2"

        u_os = float(x_os[3]) if abs(x_os[3]) > 0.05 else float(u_nominal)
        u_ts = float(x_ts[3]) if abs(x_ts[3]) > 0.05 else float(u_nominal)

        if u_os < 0.10 or u_ts < 0.10:
            cls._mode_a_active = False
            cls._mode_b_active = False
            cls._w_evasive_latched = None
            cls._latched_scenario = None
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
        cls._mode_b_active = False  # Deactivate legacy Mode B logic

        hor_time = max(100.0, tcpa + 20.0) if tcpa > 0.0 else 60.0
        dt_sim = 0.5
        steps = int(hor_time / dt_sim)

        # 1. Generate Target Ship Trajectory & Define State Prefix
        if w_ts_delayed is not None and len(w_ts_delayed) >= 2:
            # IS MODE (State A): Use shared route intent
            ts_traj, _ = cls._sample_route(w_ts_delayed, x_ts[:2], u_ts, steps, dt_sim)
            state_prefix = "A"
        else:
            # RA MODE (State B): Constant Velocity Assumption (Linear Extrapolation)
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
                active_track = cls._get_active_nominal_track(nom, x_os[:2])
                return active_track, 0.0, 1.0, f"State {state_prefix}.2"
        else:
            # ------------------------------------------------------------------
            # MODE A (Intent Shared): Allows dynamic early aborts because 
            # the downstream route intent is stable and reliable.
            # ------------------------------------------------------------------
            if state_prefix == "A":
                passed_cpa = (tcpa < -0.5) or (is_astern and curr_dist < 4.0)
                separating = curr_dist > (d_safe * 1.5) and (tcpa < 0.0 or tcpa > 100.0)
                threat_cleared = (min_dist_nom >= d_safe * 2.0 and curr_dist > d_safe * 2.0)
                
                if (passed_cpa and separating) or threat_cleared:
                    cls._mode_a_active = False
                    cls._w_evasive_latched = None
                    active_track = cls._get_active_nominal_track(nom, x_os[:2])
                    return active_track, 0.0, 1.0, f"State {state_prefix}.2"
            
            # ------------------------------------------------------------------
            # MODE B (Reactive RA): Requires strict physical hysteresis.
            # Volatile linear extrapolations will falsely trigger 'threat_cleared' 
            # on curves, so we force OS to hold the evasion until physically clear.
            # ------------------------------------------------------------------
            else:
                passed_cpa_strict = (tcpa <= -2.0) and (curr_dist > d_safe * 1.5)
                physically_clear = (curr_dist > d_safe * 3.0) and is_astern
                
                if passed_cpa_strict or physically_clear:
                    cls._mode_a_active = False
                    cls._w_evasive_latched = None
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

        # 6. Candidate Evasion Generation
        t_start_a1 = time.perf_counter()
        tactical_tcpa = 20.0
        k_offset = int(tactical_tcpa / dt_sim)

        k_w1 = max(0, k_cpa - k_offset)
        W_1 = os_traj_nom[k_w1]

        k_w3 = min(len(os_traj_nom) - 1, k_cpa + k_offset)
        W_3 = os_traj_nom[k_w3]

        dist_to_cpa_actual = max(0.0, s_os + (u_os * k_cpa * dt_sim) - s_os)
        D = min(tactical_tcpa * u_os, max(dist_to_cpa_actual, 3.0))

        v_nom = os_traj_nom[k_cpa] - W_1
        norm = float(np.hypot(v_nom[0], v_nom[1]))
        u_nom = (v_nom / norm) if norm > 1e-3 else np.array([1.0, 0.0])

        diffs_base = np.diff(cls._w_nominal[:, :2], axis=0)
        seg_lens_base = np.hypot(diffs_base[:, 0], diffs_base[:, 1])

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

        best_cost = float("inf")
        best_chi = 0.0
        best_p = 1.0
        best_wps = None

        for chi in cls.chi_candidates:
            for p_cand in cls.p_candidates:
                if abs(chi) < 1e-3 and p_cand > 0.99:
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

                        s_w1 = max(0.0, s_cpa - D)
                        s_w2 = s_cpa
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

                        W_1_curved = get_curved_offset_point(s_w1, 0.0)
                        W_2_curved = get_curved_offset_point(s_w2, lat_offset)
                        W_pass_curved = get_curved_offset_point(s_wpass, lat_offset)
                        W_rejoin_curved = get_curved_offset_point(s_wrejoin, 0.0)

                        rejoin_idx = len(cls._w_nominal) - 1
                        accum_track = 0.0
                        for idx, length in enumerate(seg_lens_base):
                            accum_track += length
                            if accum_track > s_wrejoin:
                                rejoin_idx = idx + 1
                                break
                        remaining_wps_curved = cls._w_nominal[rejoin_idx:, :2]

                        if len(pre_w1_wps) > 0:
                            full_route = np.vstack([pre_w1_wps, W_1_curved, W_2_curved, W_pass_curved, W_rejoin_curved, remaining_wps_curved])
                        else:
                            full_route = np.vstack([W_1_curved, W_2_curved, W_pass_curved, W_rejoin_curved, remaining_wps_curved])
                    else:
                        if len(pre_w1_wps) > 0:
                            full_route = np.vstack([pre_w1_wps, W_1, W_2, W_3, remaining_wps])
                        else:
                            full_route = np.vstack([W_1, W_2, W_3, remaining_wps])

                    sliced_route = cls._slice_path_forward(full_route, x_os[:2])
                    cand_wps = np.vstack([x_os[:2], sliced_route])

                cand_traj, _ = cls._sample_route(cand_wps, x_os[:2], u_os * p_cand, steps, dt_sim)

                cost, _, _ = cls._evaluate_candidate_hazard(
                    x_os, chi, p_cand, ts_traj, cand_traj, steps, dt_sim, d_safe, canal_polygons, scenario
                )

                if cost < best_cost:
                    best_cost = cost
                    best_chi = chi
                    best_p = p_cand
                    best_wps = cand_wps

        calc_duration_ms = (time.perf_counter() - t_start_a1) * 1000.0
        if cls._mode_a_active:
            print(
                f"\033[93m[State {state_prefix}.1 Re-planned Early] Opt Chi: {np.degrees(best_chi):.1f}° | Opt P: {best_p:.2f} | Cost: {best_cost:.2f} | {calc_duration_ms:6.2f} ms\033[0m",
                flush=True
            )
        else:
            print(
                f"\033[92m[State {state_prefix}.1 Early Plan] Opt Chi: {np.degrees(best_chi):.1f}° | Opt P: {best_p:.2f} | Cost: {best_cost:.2f} | {calc_duration_ms:6.2f} ms\033[0m",
                flush=True
            )

        cls._chi_ca_latched = float(best_chi)
        cls._p_ca_latched = float(best_p)
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