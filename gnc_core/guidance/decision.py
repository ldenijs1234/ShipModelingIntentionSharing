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
    def _slice_path_forward(cls, path_wps: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Slices a waypoint array to return only the points physically ahead of the vessel."""
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

        # 1. Collision Hazard 
        dists = np.hypot(os_x - ts_traj[:, 0], os_y - ts_traj[:, 1])
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist <= (d_safe * 2.5):
            j_safety = cls.k_safety * ((d_safe / max(min_dist, 0.05)) ** 4.0)
        else:
            j_safety = 0.0

        # 2. Grounding Penalty 
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

        # 3. Control Cost
        k_w = cls.k_chi_stb if chi > 0.0 else cls.k_chi_port
        j_control = k_w * (chi**2)
        j_speed = cls.k_p * (1.0 - p_cand)

        # 4. COLREGs Cost
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

        # Bypass broken own_ship_node.py logic by safely slicing here and returning STAND_ON
        if x_ts is None:
            cls._mode_a_active = False
            cls._mode_b_active = False
            nom = cls._w_nominal if cls._w_nominal is not None else w_os
            unsailed = cls._slice_path_forward(nom, x_os[:2])
            return np.vstack([x_os[:2], unsailed]), 0.0, 1.0, "STAND_ON"

        u_os = float(x_os[3]) if abs(x_os[3]) > 0.05 else float(u_nominal)
        u_ts = float(x_ts[3]) if abs(x_ts[3]) > 0.05 else float(u_nominal)

        # This previously returned State B.2 and triggered the backward steering bug when TS arrived at terminal WP
        if u_os < 0.10 or u_ts < 0.10:
            cls._mode_a_active = False
            cls._mode_b_active = False
            unsailed = cls._slice_path_forward(cls._w_nominal, x_os[:2])
            return np.vstack([x_os[:2], unsailed]), 0.0, 1.0, "STAND_ON"

        d_safe = VesselParams.DCPA_safe
        curr_dist = float(np.hypot(x_os[0] - x_ts[0], x_os[1] - x_ts[1]))
        
        scenario = "Unknown"
        if x_ts is not None:
            beta_init = RiskCalculator.calculate_relative_bearing(x_os, x_ts)
            scenario = RiskCalculator.classify_colreg_scenario(beta_init)

        # ----------------------------------------------------------------------
        # Mode A: Shared Intent (Latch and Verify)
        # ----------------------------------------------------------------------
        if w_ts_delayed is not None and len(w_ts_delayed) >= 2:
            cls._mode_b_active = False

            hor_time = max(100.0, tcpa + 20.0) if tcpa > 0.0 else 60.0
            dt_sim = 0.5
            steps = int(hor_time / dt_sim)

            ts_traj, _ = cls._sample_route(w_ts_delayed, x_ts[:2], u_ts, steps, dt_sim)

            unsailed_nominal = cls._slice_path_forward(cls._w_nominal, x_os[:2])
            w_os_base = np.vstack([x_os[:2], unsailed_nominal])

            os_traj, s_os = cls._sample_route(w_os_base, x_os[:2], u_os, steps, dt_sim)
            dists = np.hypot(os_traj[:, 0] - ts_traj[:, 0], os_traj[:, 1] - ts_traj[:, 1])
            k_cpa = int(np.argmin(dists))
            min_dist = float(dists[k_cpa])
            
            dx = x_ts[0] - x_os[0]
            dy = x_ts[1] - x_os[1]
            longitudinal_rel = dx * np.cos(x_os[2]) + dy * np.sin(x_os[2])
            
            is_astern = longitudinal_rel < -0.5
            passed_cpa = (tcpa < -0.5) or (is_astern and curr_dist < 4.0)
            separating = curr_dist > (d_safe * 1.5) and (tcpa < 0.0 or tcpa > 100.0)

            if passed_cpa and separating:
                cls._mode_a_active = False
                cls._w_evasive_latched = None
                return np.copy(w_os_base), 0.0, 1.0, "State A.2"

            threat_cleared = (min_dist >= d_safe * 2.0 and curr_dist > d_safe * 2.0)
            if threat_cleared:
                cls._mode_a_active = False
                cls._w_evasive_latched = None
                return np.copy(w_os_base), 0.0, 1.0, "State A.2"

            if cls._mode_a_active and cls._w_evasive_latched is not None:
                sliced_latched = cls._slice_path_forward(cls._w_evasive_latched, x_os[:2])
                cls._w_evasive_latched = np.vstack([x_os[:2], sliced_latched])
                
                os_traj_latched, _ = cls._sample_route(cls._w_evasive_latched, x_os[:2], u_os * cls._p_evasive_latched, steps, dt_sim)
                dists_latched = np.hypot(os_traj_latched[:, 0] - ts_traj[:, 0], os_traj_latched[:, 1] - ts_traj[:, 1])
                min_dist_latched = float(np.min(dists_latched))
                
                if min_dist_latched >= d_safe * 1.25:
                    return cls._w_evasive_latched, 0.0, cls._p_evasive_latched, "State A.1"
                else:
                    print(f"\033[91m[State A.1] Latched route compromised (min_dist={min_dist_latched:.1f}m)! Recalculating...\033[0m")
            
            if not cls._mode_a_active and min_dist >= d_safe * 1.5:
                return np.copy(w_os_base), 0.0, 1.0, "State A.2"

            t_start_a1 = time.perf_counter()
            
            tactical_tcpa = 20.0
            k_offset = int(tactical_tcpa / dt_sim)
            
            k_w1 = max(0, k_cpa - k_offset)
            W_1 = os_traj[k_w1]
            
            k_w3 = min(len(os_traj) - 1, k_cpa + k_offset)
            W_3 = os_traj[k_w3]

            dist_to_cpa_actual = max(0.0, s_os + (u_os * k_cpa * dt_sim) - s_os)
            D = min(tactical_tcpa * u_os, max(dist_to_cpa_actual, 3.0))

            v_nom = os_traj[k_cpa] - W_1
            norm = float(np.hypot(v_nom[0], v_nom[1]))
            u_nom = (v_nom / norm) if norm > 1e-3 else np.array([1.0, 0.0])

            s_w3 = s_os + (u_os * k_w3 * dt_sim)
            diffs_base = np.diff(cls._w_nominal[:, :2], axis=0)
            seg_lens_base = np.hypot(diffs_base[:, 0], diffs_base[:, 1])
            
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
                        
                        full_triangle = np.vstack([W_1, W_2, W_3, remaining_wps])
                        sliced_triangle = cls._slice_path_forward(full_triangle, x_os[:2])
                        cand_wps = np.vstack([x_os[:2], sliced_triangle])
                    
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
                print(f"\033[93m[State A.1 Recalculating] Opt Chi: {np.degrees(best_chi):.1f}° | Cost: {best_cost:.2f} | {calc_duration_ms:6.2f} ms\033[0m", flush=True)
            else:
                print(f"\033[92m[State A.1 Initial Plan] Opt Chi: {np.degrees(best_chi):.1f}° | Cost: {best_cost:.2f} | {calc_duration_ms:6.2f} ms\033[0m", flush=True)

            cls._chi_ca_latched = float(best_chi)
            cls._p_ca_latched = float(best_p)
            cls._w_evasive_latched = best_wps
            cls._mode_a_active = True
            
            return cls._w_evasive_latched, 0.0, cls._p_evasive_latched, "State A.1"

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
            has_passed = (tcpa < 0.0) and (curr_dist > (d_safe * 1.2))
            if has_passed:
                cls._mode_b_active = False
        else:
            if risk_active:
                cls._mode_b_active = True

        if cls._mode_b_active:
            urgency = np.clip((d_safe - dcpa) / max(d_safe, 1e-3), 0.0, 1.0)
            psi_headon = np.radians(30.0 + (60.0 - 30.0) * urgency)

            if scenario == "Head-On":
                psi_ca = psi_headon 
                p_ca = 0.5 if urgency > 0.6 else 1.0
            elif scenario in ["Crossing_A", "Crossing_B"]:
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

            if canal_polygons is not None:
                vx = u_os * np.cos(x_os[2] + psi_ca)
                vy = u_os * np.sin(x_os[2] + psi_ca)
                future_pos = (x_os[0] + vx * 5.0, x_os[1] + vy * 5.0)
                
                projected_line = LineString([(x_os[0], x_os[1]), future_pos])
                d_static_future = float(projected_line.distance(canal_polygons))

                if d_static_future <= VesselParams.d_safe_static:
                    for scale_factor in [0.75, 0.5, 0.25]:
                        test_psi = psi_ca * scale_factor
                        test_line = LineString([
                            (x_os[0], x_os[1]),
                            (x_os[0] + u_os * np.cos(x_os[2] + test_psi) * 5.0,
                             x_os[1] + u_os * np.sin(x_os[2] + test_psi) * 5.0)
                        ])
                        if test_line.distance(canal_polygons) > VesselParams.d_safe_static:
                            psi_ca = test_psi
                            p_ca = 0.5 
                            break
                    else:
                        p_ca = 0.0
                        psi_ca = 0.0

            return np.copy(w_os), float(psi_ca), float(p_ca), "State B.1"

        return np.copy(cls._w_nominal), 0.0, 1.0, "State B.2"