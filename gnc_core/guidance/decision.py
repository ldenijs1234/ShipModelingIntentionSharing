#!/usr/bin/env python3
from typing import Optional, Tuple
import numpy as np
import rclpy.logging

from gnc_core.config.vessel_params import VesselParams
from gnc_core.navigation.risk import RiskCalculator


class DecisionLayer:
    _mode_a_active = False
    _mode_b_active = False
    _w_evasive_latched = None

    k_chi_stb = 5.0
    k_chi_port = 5.2

    # Discrete search set matching Akdag (radians)
    chi_candidates = np.radians(
        np.array([
            -75.0, -60.0, -45.0, -30.0, -15.0,
            0.0,
            15.0, 30.0, 45.0, 60.0, 75.0,
        ])
    )

    @classmethod
    def _evaluate_candidate_hazard(
        cls,
        x_os: np.ndarray,
        chi: float,
        ts_traj: np.ndarray,
        steps: int,
        dt_sim: float,
        d_safe: float,
    ) -> Tuple[float, float, int]:
        u_os = float(x_os[3]) if abs(x_os[3]) > 0.05 else 0.45
        psi_cand = x_os[2] + chi

        vx = u_os * np.cos(psi_cand)
        vy = u_os * np.sin(psi_cand)

        t_steps = np.arange(steps) * dt_sim
        os_x = x_os[0] + vx * t_steps
        os_y = x_os[1] + vy * t_steps

        dists = np.hypot(os_x - ts_traj[:, 0], os_y - ts_traj[:, 1])
        min_idx = int(np.argmin(dists))
        min_dist = float(dists[min_idx])

        if min_dist <= d_safe:
            j_safety = 1000.0 * ((d_safe / max(min_dist, 0.05)) ** 4.0)
        else:
            j_safety = 50.0 * (d_safe / min_dist)

        k_w = cls.k_chi_stb if chi <= 0.0 else cls.k_chi_port
        j_control = k_w * (chi**2)

        return j_safety + j_control, min_dist, min_idx

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
    ) -> Tuple[np.ndarray, float, str]:
        if x_ts is None:
            cls._mode_a_active = False
            cls._mode_b_active = False
            cls._w_evasive_latched = None
            return np.copy(w_os), 0.0, "STAND_ON"

        if x_os[3] < 0.20 or x_ts[3] < 0.20:
            cls._mode_a_active = False
            cls._mode_b_active = False
            cls._w_evasive_latched = None
            return np.copy(w_os), 0.0, "State B.2"

        d_safe = VesselParams.DCPA_safe
        curr_dist = float(np.hypot(x_os[0] - x_ts[0], x_os[1] - x_ts[1]))

        # ----------------------------------------------------------------------
        # Mode A: Shared Intent Route Available
        # ----------------------------------------------------------------------
        if w_ts_delayed is not None and len(w_ts_delayed) >= 2:
            # 1. Cross-Track Error relative to original nominal mission
            # For a straight North track (x along North, y cross-track):
            e_cte = abs(x_os[1] - w_os[0, 1])

            # If the ship was evading and has returned close to the nominal line,
            # OR if vessels are past CPA, stay strictly in State A.2
            has_passed_cpa = tcpa < 0.0
            is_back_on_route = e_cte < 0.35 and (
                cls._mode_a_active or cls._mode_b_active
            )

            if has_passed_cpa or (is_back_on_route and curr_dist > (d_safe * 1.2)):
                cls._mode_a_active = False
                cls._mode_b_active = False
                cls._w_evasive_latched = None
                return np.copy(w_os), 0.0, "State A.2"

          # 2. Maintain active evasive route while clearing
            if cls._mode_a_active and cls._w_evasive_latched is not None:
                # Release condition: OS has passed TS longitudinally or CPA has cleared
                if (tcpa < -1.0 and curr_dist > (d_safe * 1.3)) or (tcpa < -15.0):
                    cls._mode_a_active = False
                    cls._w_evasive_latched = None
                    return np.copy(w_os), 0.0, "State A.2"
                return cls._w_evasive_latched, 0.0, "State A.1"

            cls._mode_b_active = False

            u_os = float(x_os[3]) if abs(x_os[3]) > 0.05 else float(u_nominal)
            u_ts = float(x_ts[3]) if abs(x_ts[3]) > 0.05 else float(u_nominal)
            hor_time = 70.0
            dt_sim = 0.5
            steps = int(hor_time / dt_sim)

            # Sample TS along broadcast intent route
            ts_traj, _ = cls._sample_route(w_ts_delayed, x_ts[:2], u_ts, steps, dt_sim)

            # 2. Build the OS active trajectory from CURRENT PHYSICAL POSITION
            # Find downstream mission waypoints strictly ahead of OS current position
            diffs_os = np.diff(w_os[:, :2], axis=0)
            downstream_indices = []
            for i in range(len(w_os) - 1):
                seg_dir = diffs_os[i]
                v_to_wp = w_os[i + 1, :2] - x_os[:2]
                if np.dot(seg_dir, v_to_wp) > 1.0:
                    downstream_indices.append(i + 1)

            if len(downstream_indices) > 0:
                unsailed_wps = w_os[downstream_indices[0]:, :2]
            else:
                unsailed_wps = w_os[-1:, :2]

            # Active baseline: Live position -> Unsailed waypoints
            w_os_base = np.vstack([x_os[:2], unsailed_wps])

            # Sample rollout along this live-anchored route
            os_traj, s_os = cls._sample_route(w_os_base, x_os[:2], u_os, steps, dt_sim)
            dists = np.hypot(os_traj[:, 0] - ts_traj[:, 0], os_traj[:, 1] - ts_traj[:, 1])
            k_cpa = int(np.argmin(dists))
            min_dist = float(dists[k_cpa])

            # If already clear of TS, smoothly track to the remaining mission waypoints
            if min_dist >= (d_safe * 1.35) and curr_dist > (d_safe * 1.4):
                return np.copy(w_os), 0.0, "State A.2"

            # 3. Optimize passing direction with Akdag cost function
            best_cost = float("inf")
            best_chi = 0.0
            for chi in cls.chi_candidates:
                cost, _, _ = cls._evaluate_candidate_hazard(x_os, chi, ts_traj, steps, dt_sim, d_safe)
                if cost < best_cost:
                    best_cost = cost
                    best_chi = chi

            # Starboard (+1) vs Port (-1) normal displacement
            lat_sign = 1.0 if best_chi <= 0.0 else -1.0
            req_offset = lat_sign * max(d_safe * 1.4, 1.8)

            # Conflict point on the active live route
            P_cpa = os_traj[k_cpa]

            # Vector of active segment at CPA
            diffs_base = np.diff(w_os_base[:, :2], axis=0)
            seg_lens_base = np.hypot(diffs_base[:, 0], diffs_base[:, 1])
            s_cpa = s_os + (u_os * k_cpa * dt_sim)

            accum = 0.0
            cpa_seg_idx = 0
            for idx, length in enumerate(seg_lens_base):
                if accum + length >= s_cpa:
                    cpa_seg_idx = idx
                    break
                accum += length

            seg_dir = diffs_base[cpa_seg_idx] / max(seg_lens_base[cpa_seg_idx], 1e-4)
            n_stb = np.array([-seg_dir[1], seg_dir[0]])

            # 4. Enforce the "No-Backtrack" safety invariant:
            # If OS is already displaced to Starboard, W_evade must never be closer to the center than current position!
            W_evade = P_cpa + req_offset * n_stb
            if x_os[1] > 0.5 and W_evade[1] < x_os[1]:
                W_evade[1] = x_os[1] + 0.5  # Maintain/expand current clearance
            elif x_os[1] < -0.5 and W_evade[1] > x_os[1]:
                W_evade[1] = x_os[1] - 0.5

            # 5. Assemble and LATCH evasive route
            rejoin_idx = min(cpa_seg_idx + 1, len(w_os_base) - 1)
            remaining_wps = w_os_base[rejoin_idx:, :2]

            cls._w_evasive_latched = np.vstack([x_os[:2], W_evade, remaining_wps])
            cls._mode_a_active = True
            return cls._w_evasive_latched, 0.0, "State A.1"

        # ----------------------------------------------------------------------
        # Mode B: Reactive Fallback (No Intent)
        # ----------------------------------------------------------------------
        cls._mode_a_active = False
        cls._w_evasive_latched = None

        domain_breached, _ = RiskCalculator.check_ship_domain_breach(x_os, x_ts)
        threshold_cpa = max(d_safe * 1.15, VesselParams.R_lateral + VesselParams.B)
        cpa_risk = (dcpa < threshold_cpa) and (0.0 <= tcpa <= VesselParams.TCPA_safe)
        risk_active = cpa_risk or domain_breached

        if cls._mode_b_active:
            if tcpa <= 0.0 and dcpa >= threshold_cpa:
                cls._mode_b_active = False
        else:
            if risk_active:
                cls._mode_b_active = True

        if cls._mode_b_active:
            beta = RiskCalculator.calculate_relative_bearing(x_os, x_ts)
            scenario = RiskCalculator.classify_colreg_scenario(beta)

            urgency = np.clip((d_safe - dcpa) / max(d_safe, 1e-3), 0.0, 1.0)
            psi_headon = np.radians(30.0 + (60.0 - 30.0) * urgency)

            if scenario == "Head-On":
                psi_ca = psi_headon
            elif scenario in ["Crossing_A", "Crossing_B"]:
                psi_ca = np.radians(45.0)
            else:
                psi_ca = np.radians(30.0)

            return np.copy(w_os), float(psi_ca), "State B.1"

        return np.copy(w_os), 0.0, "State B.2"