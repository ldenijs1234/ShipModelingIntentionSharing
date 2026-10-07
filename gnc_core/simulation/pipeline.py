from typing import Any, Dict, Optional, Tuple
import numpy as np

from gnc_core.models.vessel_dynamics import VesselDynamics
from gnc_core.navigation.state_estimation import StateEstimation
from gnc_core.navigation.risk import RiskCalculator
from gnc_core.guidance.decision import DecisionLayer
from gnc_core.guidance.los import LOSGuidance
from gnc_core.control.autopilot import Autopilot


class SynchronousPipeline:

    @staticmethod
    def step(
        internal_state: np.ndarray,
        w_mission_os: np.ndarray,
        x_ts_raw: np.ndarray,
        w_ts_delayed: Optional[np.ndarray],
        cached: Dict[str, Any],
        dt: float = 0.1,       # Default strictly to 10 Hz
        u_nominal: float = 0.5,
        canal_polygons = None
    ) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
        
        # 1. State Estimation (10 Hz)
        x_os = StateEstimation.estimate_own_state(internal_state)
        x_ts = StateEstimation.estimate_target_from_ais(x_ts_raw, dt_ais=dt)

        # 2. Risk & Decision (10 Hz)
        cached["dcpa"], cached["tcpa"] = RiskCalculator.calculate_cpa(x_os, x_ts, u_nominal)

        prev_state = cached.get("state", "State B.2")
        prev_route_len = len(cached.get("w_active", w_mission_os))

        # Unpack the 4 returned values, including the speed multiplier 'p_ca'
        cached["w_active"], cached["psi_ca"], cached["p_ca"], cached["state"] = DecisionLayer.evaluate(
            x_os, x_ts, w_mission_os, w_ts_delayed, cached["dcpa"], cached["tcpa"], u_nominal, canal_polygons
        )

        # Re-anchor downstream waypoint index upon state transitions back to nominal mission
        transitioned_from_ca = (
            (prev_state == "State B.1" and cached["state"] == "State B.2") or
            (prev_state == "State A.1" and cached["state"] == "State A.2")
        )

        if transitioned_from_ca:
            # Rejoin nearest unsailed segment ahead along the nominal mission
            best_idx = len(w_mission_os) - 1
            for idx in range(len(w_mission_os) - 1):
                seg_vec = w_mission_os[idx + 1, :2] - w_mission_os[idx, :2]
                v_ship = x_os[:2] - w_mission_os[idx, :2]
                seg_len = float(np.hypot(seg_vec[0], seg_vec[1]))
                proj = float(np.dot(v_ship, seg_vec)) / max(seg_len, 1e-4)

                # Segment is ahead if the projection has not reached the end of the segment
                if proj < (seg_len - 0.5):
                    best_idx = idx + 1
                    break
            cached["wp_idx"] = max(1, best_idx)
            
        # Reset waypoint index when entering ANY evasion mode or when the route dynamically updates
        elif (cached["state"] in ["State A.1", "State B.1"] and prev_state not in ["State A.1", "State B.1"]):
            cached["wp_idx"] = 1
        elif len(cached["w_active"]) != prev_route_len:
            cached["wp_idx"] = 1

        # 3. Guidance Layer (LOS Waypoint Tracking applies universally now)
        cached["psi_wp"], _, cached["wp_idx"] = LOSGuidance.compute_heading_reference(
            x_os, cached["w_active"], cached.get("wp_idx", 1)
        )

        # Removed legacy State B.1 tangent override. 
        # All modes now smoothly track their active route via LOS.
        psi_guidance_ref = cached["psi_wp"]

        # 4. Control Layer (Eq. 3.42: psi_cmd = psi_ref + psi_ca)
        # Note: psi_ca is now inherently 0.0 from the DecisionLayer for unified geometries
        target_speed = float(u_nominal * cached.get("p_ca", 1.0))
        
        u_c, tau_c, psi_cmd = Autopilot.compute_control(
            x_os, psi_guidance_ref, cached["psi_ca"], target_speed
        )

        # 5. Vessel Dynamics Integration via RK4 (10 Hz)
        next_internal_state = VesselDynamics.rk4(internal_state, np.array([tau_c, u_c]), dt=dt)

        telemetry = {
            "time": cached.get("time", 0.0) + dt,
            "x_os": x_os,
            "u_c": u_c,
            "tau_c": tau_c,
            "psi_cmd": psi_cmd,
            "dcpa": cached["dcpa"],
            "tcpa": cached["tcpa"],
            "active_state": cached["state"],
        }
        cached["time"] = telemetry["time"]
        
        return next_internal_state, cached, telemetry